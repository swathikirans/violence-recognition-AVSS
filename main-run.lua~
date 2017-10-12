--[[
  Program to evaluate the performance of a model described in the paper:
  
  Sudhakaran, Swathikiran, and Oswald Lanz. "Learning to Detect Violent Videos using Convolutional Long 
  Short-Term Memory." arXiv preprint arXiv:1709.06531 (2017).
--]]

unpack = unpack or table.unpack

require 'torch'
require 'nn'
require 'rnn'
require 'cunn'
require 'paths'
require 'cutorch'
require 'cudnn'
require 'image'
require 'ConvLSTM'

-- Options
cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-FightsDataset','./movie_dataset/dataset_fights_norm','Fight Videos')
cmd:option('-noFightsDataset','./movie_dataset/dataset_nofights_norm','Non-Fight Videos')
cmd:option('-model','./models.model_movies','Model')
cmd:option('-nSeq',10,'Length of the sequence')
cmd:text()

params = cmd:parse(arg) -- Parse the arguments
cutorch.setDevice(1)

model = torch.load(params.model) -- Load the model
model:cuda()
print('Model loaded')
dataF = torch.load(params.FightsDataset)
dataNF = torch.load(params.noFightsDataset)
labelF = torch.Tensor(dataF:size(1)):fill(1)
labelNF = torch.Tensor(dataNF:size(1)):fill(2)
testDataset = torch.cat(dataF, dataNF, 1)
testLabel = torch.cat(labelF, labelNF, 1)
numSamples = testDataset:size(1)
print('Dataset loaded')

local function main()
  print('Calculating test accuracy...')
  model:evaluate()
  numCorr = 0
  testBatch = torch.Tensor(params.nSeq-1, 16, 3, 224, 224)
  label_test = torch.Tensor(16)
  for i1 = 1, numSamples do
    print(i1 .. '/' .. numSamples)
    inputTableTest = {}   
    data = testDataset[i1]
    label_test[1] = testLabel[i1]
--    Set up the data
    for j = 1, params.nSeq-1 do
      img_test1 = image.scale(data[j], 224, 224)
      img_test2 = image.scale(data[j+1], 224, 224)
      testBatch[{{j},{1}}] = img_test1 - img_test2
    end  
    for i = 1, params.nSeq-1 do
      table.insert(inputTableTest, testBatch[i]:cuda())
    end
    output = model:forward(inputTableTest) -- Forward pass
    _,max_ind = torch.max(torch.exp(output),2) -- Compute the output probability  
    if max_ind[1][1] == label_test[1] then
      numCorr = numCorr + 1
    end    
    model:forget()
  end
  print('Accuracy  = ' .. numCorr/numSamples*100 .. '%')
  print ('Testing done')
end

main()
