import torch
import torch.nn as nn
import torchvision.models as models
import brevitas.nn as qnn
from .common import print_config
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8WeightPerTensorFloat
from brevitas.quant import Int8Bias, Int8BiasPerTensorFloatInternalScaling, Int16Bias, IntBias
from .vgg import *

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
BIAS_QUANTIZER = Int8Bias
WEIGHT_QUANTIZER = Int8WeightPerTensorFixedPoint
class QuantVGG_pure(nn.Module):

    def __init__(self, VGG_type='A', batch_norm=False, bit_width=8, num_classes=1000, pretrained_model=None):
        super(QuantVGG_pure, self).__init__()
        self.inp_quant = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.features = make_layers(cfgs[VGG_type], batch_norm, bit_width)
        self.avgpool = qnn.QuantAdaptiveAvgPool2d((7, 7))        
        self.classifier = nn.Sequential(
            qnn.QuantLinear(512 * 7 * 7, 4096, bias=True, weight_bit_width=bit_width, bias_quant=BIAS_QUANTIZER,weight_quant=WEIGHT_QUANTIZER, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True),
            qnn.QuantDropout(),
            qnn.QuantLinear(4096, 4096, bias=True, weight_bit_width=bit_width, bias_quant=BIAS_QUANTIZER,weight_quant=WEIGHT_QUANTIZER, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True),
            qnn.QuantDropout(),
            qnn.QuantLinear(4096, num_classes, bias=False, weight_bit_width=bit_width),
        )
        self.classifier[0].cache_inference_quant_bias=True
        self.classifier[3].cache_inference_quant_bias=True
        self.classifier[6].cache_inference_quant_bias=True

        print_config()

        if pretrained_model == None:
            self._initialize_weights()
        else:
            if pretrained_model == 'pytorch':
                print("Initializing with pretrained model from PyTorch")
                pre_model=models.vgg16(pretrained=True) # use pytorch's pretrained model
            else:
                pre_model=VGG_net(VGG_type=VGG_type,batch_norm=batch_norm,num_classes=num_classes)
                loaded_model=torch.load(pretrained_model)['state_dict']
                if next(iter(loaded_model.keys())).startswith('module'): # check if model was trained using DataParallel, keys() return 'odict_keys' which does not support indexing
                    pre_model=torch.nn.DataParallel(pre_model)           # if model is trained w/ DataParallel it's warraped under module
                    pre_model.load_state_dict(loaded_model)
                    unwrapped_sd=pre_model.module.state_dict()
                    pre_model=VGG_net(VGG_type=VGG_type,batch_norm=batch_norm,num_classes=num_classes)
                    pre_model.load_state_dict(unwrapped_sd)
                else:
                    pre_model.load_state_dict(loaded_model)
            self._initialize_custom_weights(pre_model)

    def forward(self, x):
        x = self.inp_quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        print("Initializing model")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if isinstance(m.bias,torch.nn.parameter.Parameter):
                    nn.init.constant_(m.bias, 0)

    def _initialize_custom_weights(self,old_model):
        print("Initializing model with custom weights & bias")
        for n, o in zip(self.features,old_model.features):
            if isinstance(n,nn.Conv2d):
                n.weight.data=o.weight.data
                if n.bias != None and o.bias != None:
                    n.bias.data=o.bias.data
            elif isinstance(n, nn.BatchNorm2d):
                nn.init.constant_(n.weight, 1)
                nn.init.constant_(n.bias, 0)
        for l in self.classifier:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, 0, 0.01)
                if isinstance(l.bias,torch.nn.parameter.Parameter):
                    nn.init.constant_(l.bias, 0)
        print("Initialization Done")

def make_layers(cfg, batch_norm, bit_width):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [qnn.QuantMaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = qnn.QuantConv2d(in_channels, v, kernel_size=3, stride=1, padding=1, groups=1, weight_bit_width=bit_width, bias_quant=BIAS_QUANTIZER,weight_quant=WEIGHT_QUANTIZER, return_quant_tensor=True)
            conv2d.cache_inference_quant_out = True
            conv2d.cache_inference_quant_bias = True
            act = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)