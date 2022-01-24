# BenchMark
Based on https://github.com/victoresque/pytorch-template

## How to train a model?
```bash
python train.py -c path_to_config.json
```
If `'n_gpu'` in config.json is more than 1 the model will be wrapped with `torch.nn.DataParallel()`
## How to train using DDP?
```bash
python train_ddp.py --resume path_to_model.pth
```
In addition to using `train_ddp.py`, `dist_backend` & `dist_url` options should be defined in config.json, other wise default values defined in train_ddp.py will be used.

## How to test a model?
```bash
python test.py --resume path_to_model.pth
```

## How to fuse `nn.BatchNorm2d` with `QuantConv2d` in a model?
`fuser.py` can be used to do so.
```bash
python fuser.py -r /path/to/pth/file
```
* `--bs` Controls the batch size. 

## How to extract parameters from model?
```bash
python parameters_extractor.py --model path_to_model.pth
```
To fuse `BatchNorm2d` into the preceding `Conv2d`, add `-f` option to the previous command. 

## How to seprate the weights?
In this template the weights are saved together with config & other objects in the `pth` file, it can be seprated using the following command.
```bash
python checkpoint_separator.py --model path_to_model.pth
```
The resulting weights file will be saved next to the model.

## How to create config.json?
Config.json is a json file that descrips the experiment to run, sevral examples are avalibe under config_json.
### **Keys dictionary**

#### **name**
Name of the experment, a timestamp will be appended to it using `'_%d_%m_%H%M%S'` as format. This can be modified in `parse_config.json` 
#### **n_gpu**
Total number of gpus to use. If set to `-1` it will use all the available gpus
#### **dist_backend** 
Distributed backend to use. For more info check [relative PyTorch docs](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group).
#### **dist_url**
URL specifying how to initialize the process group. For more info check [relative PyTorch docs](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group).
#### **arch**
The architecure can be defined using this key. The value should be a dictionary with two parameters `type` & `args`. Types can be `{VGG_net,QuantVGG_pure,vgg16,alexnet}`.\
`VGG_net`: Local implementation of VGG.\
`QuantVGG_pure`: Quantized VGG using xilinx-brevitas.\
`vgg16` & `alexnet`: Are the respective models taken from Pytorch.\
\
Example of `QuantVGG_pure` 
```json
"arch": {
        "type": "QuantVGG_pure",
        "args": {
            "VGG_type":"D", 
            "batch_norm":false, 
            "bit_width":16, 
            "num_classes":1000,
            "pretrained_model":"Path to pretrained model or use pytorch to initialize with pytorch's version of the model"
        }
    }
```
Example of `VGG_net`
```json
"arch": {
        "type": "VGG_net",
        "args": {
            "in_channels":3, 
            "num_classes":1000,
            "VGG_type":"D", 
            "batch_norm":true
        }
    }
```
Example of `vgg16`
```json
"arch": {
        "type": "vgg16",
        "args": {
            "pretrained":true, 
            "progress":true
        }
    }
```

#### **data_loader**
Example of `CIFAR_data_loader`
```json
    "data_loader": {
        "type": "CIFAR_data_loader",
        "args":{
            "data_dir": "/Path/to/Data_set",
            "batch_size": 512,
            "download": true,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 5,
            "flavor": 100,
            "training": true
        }
    }
```
Example of `ImageNet_data_loader`
```json
    "data_loader": {
        "type": "ImageNet_data_loader",
        "args":{
            "data_dir": "/Path/to/Data_set",
            "batch_size": 512,
            "shuffle": true,
            "num_workers": 5,
            "pin_memory":true,
            "training": true
        }
    }
```


#### **test_data_loader**
Example of CIFER `test_data_loader`
```json
    "data_loader": {
        "type": "CIFAR_data_loader",
        "args":{
            "data_dir": "/Path/to/Data_set",
            "batch_size": 512,
            "download": true,
            "shuffle": true,
            "validation_split": 0.0,
            "num_workers": 5,
            "flavor": 100,
            "training": false
        }
    }
```

#### **loss**
to do


#### **metrics**
Metrics used to evaluate the model, currently defined metrics are `accuracy` & `top_k_acc`. Default value for `k` in `top_k_acc` is 5. New metrics can be defined under `model/metric.py`.

#### **lr_scheduler**
Example of StepLR
```json
"lr_scheduler": {
    "type": "StepLR",
    "args": {
        "step_size": 30,
        "gamma": 0.1,
        "verbose": true
    }
}
```
Example of MultiStepLR
```json
"lr_scheduler": {
    "type": "MultiStepLR",
    "args": {
        "milestones": [60,120,160],
        "gamma": 0.2,
        "verbose": true
    }
}
```
#### **trainer**
Example of trainer config
```json
"trainer": {
    "epochs": 200,
    "save_dir": "/Where/to/save/train_result",
    "save_period": 200,
    "verbosity": 2,                 // 0: quiet, 1: per epoch, 2: full
    "monitor": "max val_accuracy", 
    "early_stop": -1,
    "tensorboard": true
}
```



#### **extract**
If set the model parameters will be extracted during testing 
```json
"extract": true,
```

#### **extractor**
Configrations to be used in the resulting config.h file
```json
"extractor": {
    "PE": 1,
    "SIMD": 1,
    "DATAWIDTH": 64,
    "SEQUENCE_LENGTH": 120000,
    "CLASS_LABEL_BITS": 1,
    "MUL_BITS": 16,
    "MUL_INT_BITS": 8,
    "ACC_BITS": 16,
    "ACC_INT_BITS": 8,
    "IA_BITS": 8,
    "IA_INT_BITS": 4
}
```
## To Do
-Generic model initialization from PyTorch\
## Notes
* If you get NCCL errors rerun with `export NCCL_DEBUG=WARN`
* If you get `RuntimeError: CUDA error: all CUDA-capable devices are busy or unavailable` while training with DDP try setting `pin_memory` to false in `config.json`