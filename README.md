# BenchMark
Based on https://github.com/victoresque/pytorch-template

## How to train a model?
```bash
python train.py -c path_to_config.json
```

## How to test a model?
```bash
python test.py --resume path_to_model.pth
```

## How to extract parameters from model?
```bash
python parameters_extractor.py --model path_to_model.pth
```

## How to seprate the weights?
In this template the weights are saved together with config & other objects in the `pth` file, it can be seprated using the following command.
```bash
python checkpoint_separator.py --model path_to_model.pth
```
The resulting weights file will be saved next to the model.

## How to create config.json?
config.json is a json file that descrips the expermit to run, sevral examples are avalibe under config_json.
### **Keys dictionary**

#### **arch**
The architecure can be defined using this key. The value should be a dictionary with two parameters `type` & `args`. types can be `{VGG_net,QuantVGG,vgg16,alexnet}`.\
VGG_net: Local implementation of VGG.\
QuantVGG: Quantized VGG using xilinx-brevitas.\
vgg16 & alexnet: Are the respective models taken from Pytorch.\
For the required arguments check respective class defination.

#### **data_loader**
to do


#### **test_data_loader**
to do


#### **loss**
to do


#### **metrics**
to do


#### **lr_scheduler**
to do


#### **trainer**
to do


#### **extract**
to do


#### **extractor**
to do

## To Do
-No need to create a new folder when resuming traning
-Generic model initialization from PyTorch