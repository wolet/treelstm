--[[

  Sentiment classification using a Binary Tree-LSTM.

--]]

local TreeLSTMSentiment = torch.class('treelstm.TreeLSTMSentiment')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

function TreeLSTMSentiment:__init(config)
  self.mem_dim		 = config.mem_dim	    or 150
  self.learning_rate	 = config.learning_rate	    or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.1
  self.batch_size	 = config.batch_size	    or 25
  self.reg		 = config.reg		    or 1e-4
  self.structure	 = config.structure	    or 'constituency'
  self.fine_grained	 = (config.fine_grained == nil) and true or config.fine_grained
  self.dropout		 = (config.dropout == nil) and true or config.dropout
  self.patience 	 = config.patience	    or 5
  self.sorted    	 = config.sorted
  self.psp               = config.psp               or ''
  self.msp               = config.msp               or ''
  self.best_dev_score    = config.best_dev_score    or 0
  self.n_observed        = config.n_observed        or 0
  self.progress          = config.progress          or 0
  self.regimen           = config.regimen           or 'vanilla'
  self.keep_log          = config.keep_log          or false
  self.episode           = config.episode           or 10000
  self.log_trn_acc       = config.log_trn_acc       or {}
  self.log_trn_lss       = config.log_trn_lss       or {}
  self.log_val_acc       = config.log_val_acc       or {}
  self.log_val_lss       = config.log_val_lss       or {}
  self.log_bucket        = config.log_bucket        or {}

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  self.in_zeros = torch.zeros(self.emb_dim)
  self.num_classes = self.fine_grained and 5 or 3

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()

  local treelstm_config = {
    in_dim  = self.emb_dim,
    mem_dim = self.mem_dim,
    output_module_fn = function() return self:new_sentiment_module() end,
    criterion = self.criterion,
  }

  if self.structure == 'dependency' then
    self.treelstm = treelstm.ChildSumTreeLSTM(treelstm_config)
  elseif self.structure == 'constituency' then
    self.treelstm = treelstm.BinaryTreeLSTM(treelstm_config)
  else
    error('invalid parse tree type: ' .. self.structure)
  end

  self.params, self.grad_params = self.treelstm:getParameters()
end

function TreeLSTMSentiment:new_sentiment_module()
  local sentiment_module = nn.Sequential()
  if self.dropout then
    sentiment_module:add(nn.Dropout())
  end
  sentiment_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    :add(nn.LogSoftMax())
  return sentiment_module
end

function TreeLSTMSentiment:train(dataset, dev_dataset, best_dev_model)

   self.treelstm:training()
   local zeros = torch.zeros(self.mem_dim)
   local indices = torch.randperm(dataset.size)
   if self.sorted then
      indices, _ = torch.sort(indices)
   end

   prev_10K = math.floor(self.n_observed/self.episode)
   for i = 1, dataset.size, self.batch_size do

      xlua.progress(math.floor(i/self.batch_size), math.floor(dataset.size/self.batch_size))
      local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

      local feval = function(x)
	 self.grad_params:zero()
	 self.emb:zeroGradParameters()

	 local loss = 0
	 for j = 1, batch_size do
	    local idx = indices[i + j - 1]
	    local sent = dataset.sents[idx]
	    local tree = dataset.trees[idx]

	    local inputs = self.emb:forward(sent)
	    local _, tree_loss = self.treelstm:forward(tree, inputs)
	    loss = loss + tree_loss
	    local input_grad = self.treelstm:backward(tree, inputs, {zeros, zeros})
	    self.emb:backward(sent, input_grad)
	 end

	 loss = loss / batch_size
	 self.grad_params:div(batch_size)
	 self.emb.gradWeight:div(batch_size)

	 -- regularization
	 loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
	 self.grad_params:add(self.reg, self.params)
	 return loss, self.grad_params
      end

      optim.adagrad(feval, self.params, self.optim_state)
      self.emb:updateParameters(self.emb_learning_rate)

      self.n_observed = self.n_observed + batch_size
      if prev_10K < math.floor(self.n_observed/self.episode) then -- 10K training instances are observed
      	 prev_10K = math.floor(self.n_observed/self.episode)

     	 dev_predictions, dev_loss = self:predict_dataset(dev_dataset)
     	 dev_score = accuracy(dev_predictions, dev_dataset.labels)

	 if self.keep_log then self:write_predictions(dev_predictions, prev_10K * self.episode) end
     	 printf('%d instances observed | current & best dev score: %.4f| %.4f\n', self.n_observed, dev_score, self.best_dev_score)

      	 if dev_score > self.best_dev_score then
	    self.best_dev_score = dev_score
      	    self.progress = -1
	    best_dev_model.params:copy(self.params)
      	    best_dev_model.emb.weight:copy(self.emb.weight)
      	    printf('best model is updated... \n')
      	 end
      	 self.progress = self.progress + 1
      	 printf('no improvement since %d instances %d %d\n',self.progress * self.episode, self.progress, self. patience)
      	 if self.progress >= self.patience then
      	    printf('stopping training...\n',self.patience)
	    return true
      	 end
      end
   end
   return false
end

function TreeLSTMSentiment:write_predictions(predictions, n_10K)
   local predictions_file = torch.DiskFile(self.psp .. '.dev.' .. n_10K, 'w')
   print('writing predictions to ' .. self.psp .. '.dev.' .. n_10K)
   for i = 1, predictions:size(1) do
      predictions_file:writeInt(predictions[i])
   end
   predictions_file:close()

end

function TreeLSTMSentiment:predict(tree, sent, eval_mode)

  local prediction
  local inputs = self.emb:forward(sent)
  _, loss = self.treelstm:forward(tree, inputs)
  local output = tree.output
  if self.fine_grained then
    prediction = argmax(output)
  else
    prediction = (output[1] > output[3]) and 1 or 3
  end
  self.treelstm:clean(tree)
  return prediction, loss
end

function TreeLSTMSentiment:predict_dataset(predict_dataset, eval_mode)
  local predictions = torch.Tensor(predict_dataset.size)
  local total_loss = 0
  for i = 1, predict_dataset.size do
     predictions[i], instance_loss = self:predict(predict_dataset.trees[i], predict_dataset.sents[i])
     total_loss = total_loss + instance_loss
  end
  return predictions, total_loss / predict_dataset.size
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function TreeLSTMSentiment:print_config()
  local num_params = self.params:size(1)
  local num_sentiment_params = self:new_sentiment_module():getParameters():size(1)
  printf('%-25s = %s\n',   'fine grained sentiment', tostring(self.fine_grained))
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sentiment_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %s\n',   'dropout', tostring(self.dropout))
  printf('%-25s = %s\n',   'use sorted training data', tostring(self.sorted))
  printf('%-25s = %s\n',   'patience for progress', tostring(self.patience))
  printf('%-25s = %s\n',   'prediction save path', tostring(self.psp))
  printf('%-25s = %s\n',   'model save path', tostring(self.msp))
  printf('%-25s = %s\n',   'CL regimen', tostring(self.regimen))
  printf('%-25s = %s\n',   'keep log progress', tostring(self.keep_log))
  printf('%-25s = %s\n',   'frequency of dev evaluation(episode)', tostring(self.episode))

end

function TreeLSTMSentiment:save(path)
  local config = {
    batch_size	      = self.batch_size,
    dropout	      = self.dropout,
    emb_learning_rate = self.emb_learning_rate,
    emb_vecs	      = self.emb.weight:float(),
    fine_grained      = self.fine_grained,
    learning_rate     = self.learning_rate,
    mem_dim	      = self.mem_dim,
    reg		      = self.reg,
    structure	      = self.structure,
    patience          = self.patience,
    sorted            = self.sorted,
    psp               = self.psp,
    msp               = self.msp,
    best_dev_score    = self.best_dev_score,
    n_observed        = self.n_observed,
    progress          = self.progress,
    regimen           = self.regimen,
    keep_log          = self.keep_log,
    episode           = self.episode,
    log_trn_acc       = self.log_trn_acc,
    log_trn_lss       = self.log_trn_lss,
    log_val_acc       = self.log_val_acc,
    log_val_lss       = self.log_val_lss,
    log_bucket        = self.log_bucket
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function TreeLSTMSentiment.load(path)
  local state = torch.load(path)
  local model = treelstm.TreeLSTMSentiment.new(state.config)
  model.params:copy(state.params)
  return model
end
