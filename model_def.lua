unpack = unpack or table.unpack

require 'torch'
require 'nn'
require 'rnn'
require 'cunn'
require 'paths'
require 'cutorch'
require 'cudnn'
require 'image'
require 'optim'
require 'loadcaffe'
require 'ConvLSTM'


cutorch.setDevice(1)

-- Options
-------------------------------------------------------------------------------
opt = {}
-- Model parameters:
opt.inputSizeW = 224   -- width of each input patch or image
opt.inputSizeH = 224   -- width of each input patch or image
opt.kernelSize = 3
opt.padding   = torch.floor(opt.kernelSize/2)
opt.stride = 1
opt.nSeq      = 20

-- Model definition
--------------------------------------------------------------------------------

--prototxt = '/home/sudhakaran/prog/convLSTM/siamLSTM_test1/VGG_ILSVRC_16_layers_deploy.prototxt'
--binary = '/home/sudhakaran/prog/convLSTM/siamLSTM_test1/VGG_ILSVRC_16_layers.caffemodel'

prototxt = 'bvlc_alexnet.prototxt'
binary = 'bvlc_alexnet.caffemodel'
alexnet = loadcaffe.load(prototxt, binary, 'cudnn')

for i1 = 1, 9 do
  alexnet:remove()
end

lstm_mod = nn.Sequential()
lstm_mod:add(nn.ConvLSTM(256, 256, opt.nSeq-1, opt.kernelSize, opt.kernelSize, opt.stride, opt.batchSize))
lstm_mod = w_init(lstm_mod, 'xavier')

class_layer1 = nn.Sequential()
class_layer1:add(nn.SpatialMaxPooling(2,2))
class_layer1:add(nn.Reshape(256*3*3))
class_layer1:add(nn.Linear(256*3*3, 1000))
class_layer1:add(nn.BatchNormalization(1000))
class_layer1:add(cudnn.ReLU())
class_layer1:add(nn.Linear(1000, 256))
class_layer1:add(cudnn.ReLU())
class_layer1:add(nn.Linear(256, 10))
class_layer1:add(cudnn.ReLU())
class_layer1:add(nn.Linear(10, 2))
class_layer1:add(cudnn.LogSoftMax())

encoder = nn.Sequential()
encoder:add(alexnet)
encoder:add(lstm_mod)
encoder = nn.Sequencer(encoder)

model = nn.Sequential()
model:add(encoder)
model:add(nn.SelectTable(-1))
model:add(class_layer1)
