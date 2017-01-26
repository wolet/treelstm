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
  -m,--model        (default constituency)    Model architecture: [constituency, lstm, bilstm]
  -l,--load_model  (default ./)                   model name with th extension
  -t,--test        (default data/sst/test/)       prediction data directory
  -o,--output_file (default  ./model.prediction)  output file name
  -e,--evaluate                                   evaluate accuracy
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
load_model = args.load_model
output_file = args.output_file
prefix = args.prefix
evaluate = args.evaluate

-- directory containing dataset files
local data_dir = 'data/sst/'
local test_dir = args.test

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


-- print information
header('model configuration')
local model = model_class.load(load_model)
model:print_config()
print(model.log_trn_acc)
print(model.log_trn_lss)
print(model.log_val_acc)
print(model.log_val_lss)

-- load datasets
print('loading datasets')
local dependency = (args.model == 'dependency')
print('trying to load data from' .. test_dir)
local test_dataset = treelstm.read_sentiment_dataset(test_dir, vocab, model.fine_grained, dependency)
printf('num instances  = %d\n', test_dataset.size)


-- evaluate
header('Running Model on Files under' .. test_dir)
printf('-- using model with dev score = %.4f\n', model.best_dev_score)
model.treelstm:training()
local test_predictions, _ = model:predict_dataset(test_dataset)
if evaluate then
   printf('-- score on output file: %.4f\n', accuracy(test_predictions, test_dataset.labels))
end

-- write predictions to disk
local predictions_file = torch.DiskFile(output_file, 'w')
print('writing predictions to ' .. output_file)
for i = 1, test_predictions:size(1) do
  predictions_file:writeInt(test_predictions[i])
end
predictions_file:close()
