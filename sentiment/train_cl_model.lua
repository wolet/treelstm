--[[

  Tree-LSTM training script for sentiment classication on the Stanford
  Sentiment Treebank

--]]

require('..')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

-- read command line arguments
local args = lapp [[
Training script for sentiment classification on the SST dataset.
  -m,--model      (default constituency)           Model architecture: [constituency, lstm, bilstm]
  -l,--layers     (default 1)                      Number of layers (ignored for Tree-LSTM)
  -d,--dim        (default 150)                    LSTM memory dimension
  -e,--epochs     (default 10)                     Number of training epochs
  -b,--binary                                      Train and evaluate on binary sub-task
  -p,--patience   (default 5)                      # of epochs before stopping
  -r,--regimen    (default onepass)                CL regiment: [onepass, babystep]
  -s,--save_path  (default ./)                     parent directory for trained_models and predictions
  -t,--train_root (default data/sst/8000/buckets/) buckets root directory
  -k,--keep_log                                    log the progress on validation data
  -f,--frequency  (default 10000)                  frequency of development evaluation
]]

local model_name, model_class, model_structure
if args.model == 'constituency' then
  model_name = 'Constituency Tree LSTM'
  model_class = treelstm.TreeLSTMSentiment
elseif args.model == 'dependency' then
  model_name = 'Dependency Tree LSTM'
  model_class = treelstm.TreeLSTMSentiment
elseif args.model == 'lstm' then
  model_name = 'LSTM'
  model_class = treelstm.LSTMSentiment
elseif args.model == 'bilstm' then
  model_name = 'Bidirectional LSTM'
  model_class = treelstm.LSTMSentiment
end
model_structure = args.model
prefix = args.save_path
regimen = args.regimen
header(model_name .. ' for Sentiment Classification')

-- binary or fine-grained subtask
local fine_grained = not args.binary

-- directory containing dataset files
local data_dir = 'data/sst/'
local train_root = args.train_root

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')
local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
print('loading datasets')
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local dependency = (args.model == 'dependency')
print('trying to load data from' .. train_root)
local train_dataset = treelstm.read_sentiment_dataset(train_root .. '0/', vocab, fine_grained, dependency)
local dev_dataset = treelstm.read_sentiment_dataset(dev_dir, vocab, fine_grained, dependency)
local test_dataset = treelstm.read_sentiment_dataset(test_dir, vocab, fine_grained, dependency)


-- printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)


-- create predictions and models directories if necessary
if lfs.attributes(prefix .. treelstm.predictions_dir) == nil then
   os.execute("mkdir -p " .. prefix .. treelstm.predictions_dir)
--   lfs.mkdir(prefix .. treelstm.predictions_dir)
end

if lfs.attributes(prefix ..treelstm.models_dir) == nil then
   os.execute("mkdir -p " .. prefix .. treelstm.models_dir)
--   lfs.mkdir(prefix .. treelstm.models_dir)
end

-- get paths
local file_idx = 1
local subtask = fine_grained and '5class' or '2class'
local predictions_save_path, model_save_path
while true do
   predictions_save_path = string.format(prefix .. treelstm.predictions_dir .. '/sent-%s.%s.%dl.%dd.%d.pred', args.model, subtask, args.layers, args.dim, file_idx)
   model_save_path = string.format(prefix .. treelstm.models_dir .. '/sent-%s.%s.%dl.%dd.%d.th', args.model, subtask, args.layers, args.dim, file_idx)
  if lfs.attributes(predictions_save_path) == nil and lfs.attributes(model_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

-- initialize model
local model = model_class{
  emb_vecs = vecs,
  structure = model_structure,
  fine_grained = fine_grained,
  num_layers = args.layers,
  mem_dim = args.dim,
  patience = args.patience,
  sorted = false, -- should it be optional?
  psp = predictions_save_path,
  msp = model_save_path,
  regimen = regimen,
  keep_log = args.keep_log,
  episode = args.frequency
}

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()


local train_start = sys.clock()
local best_dev_model = model
n_buckets = 10


-- train model
header('Training model')
btable = {}
for bucket = 1, n_buckets do
   local b_start = sys.clock()

   table.insert(btable, bucket)
   for epoch = 1, num_epochs do
      local start = sys.clock()
      stop = model:train(train_dataset, dev_dataset, best_dev_model)
      printf('-- finished in %.2fs epoch %d|%d for bucket %s|%d \n', sys.clock() - start ,epoch, num_epochs, table.concat(btable, ",") , n_buckets)
      if stop then break end
   end
   model.progress = 0
   printf('-- finished bucket %s in %.2fs\n', bucket , sys.clock() - b_start)
   if bucket == n_buckets then break end

   if regimen == 'babystep' then
      printf("merging bucket %d and %d\n", bucket, bucket+1)
      train_dataset = treelstm.merge_sentiment_dataset(train_dataset, train_root .. bucket .. '/', vocab, fine_grained, dependency)
   else
      train_dataset = treelstm.read_sentiment_dataset(train_root .. bucket .. '/', vocab, fine_grained, dependency)
      btable = {}
   end
end

printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', model.best_dev_score)
local test_predictions = best_dev_model:predict_dataset(test_dataset)
printf('-- test score: %.4f\n', accuracy(test_predictions, test_dataset.labels))

-- write predictions to disk
local predictions_file = torch.DiskFile(predictions_save_path .. '.test', 'w')
print('writing predictions to ' .. predictions_save_path .. '.text')
for i = 1, test_predictions:size(1) do
  predictions_file:writeInt(test_predictions[i])
end
predictions_file:close()

-- write model to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)
