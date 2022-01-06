import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models
import brevitas
import brevitas.nn as qnn
import logging
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8WeightPerTensorFloat
from brevitas.quant import Int8Bias, Int8BiasPerTensorFloatInternalScaling, Int8BiasPerTensorFixedPointInternalScaling, Int16Bias, IntBias
from .vgg import *
from utils import is_master,get_logger

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
BIAS_QUANTIZER = Int8Bias
WEIGHT_QUANTIZER = Int8WeightPerTensorFixedPoint
ACT_QUANTIZER = Int8ActPerTensorFixedPoint
RETURN_QUANT_TENSOR = True
class QuantVGG_pure(nn.Module):

    def __init__(self, VGG_type='A', batch_norm=False, bit_width=8, num_classes=1000, pretrained_model=None):
        super(QuantVGG_pure, self).__init__()
        self.logger = get_logger(name=("{}{}".format(__name__,dist.get_rank()) if dist.is_initialized() else __name__))
        self.inp_quant = qnn.QuantIdentity(bit_width=bit_width, act_quant=ACT_QUANTIZER, return_quant_tensor=RETURN_QUANT_TENSOR)
        self.features = make_layers(cfgs[VGG_type], batch_norm, bit_width)
        self.avgpool = qnn.QuantAdaptiveAvgPool2d((7, 7))        
        self.classifier = nn.Sequential(
            qnn.QuantLinear(512 * 7 * 7, 4096, 
                            bias=True, 
                            weight_bit_width=bit_width,
                            bias_quant=BIAS_QUANTIZER,
                            weight_quant=WEIGHT_QUANTIZER,
                            return_quant_tensor=RETURN_QUANT_TENSOR),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=RETURN_QUANT_TENSOR),
            qnn.QuantDropout(),
            qnn.QuantLinear(4096, 4096, 
                            bias=True, 
                            weight_bit_width=bit_width,
                            bias_quant=BIAS_QUANTIZER,
                            weight_quant=WEIGHT_QUANTIZER,
                            return_quant_tensor=RETURN_QUANT_TENSOR),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=RETURN_QUANT_TENSOR),
            nn.Dropout(),
            qnn.QuantLinear(4096, num_classes, 
                            bias=False, 
                            weight_bit_width=bit_width,
                            return_quant_tensor=False),
        )
        self.classifier[0].cache_inference_quant_bias=True
        self.classifier[3].cache_inference_quant_bias=True
        self.classifier[6].cache_inference_quant_bias=True

        if is_master():
            print_config(self.logger)

        if pretrained_model == None:
            self._initialize_weights()
        else:
            pre_model=None
            if pretrained_model == 'pytorch':
                self.logger.info("Initializing with pretrained model from PyTorch")
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
        self.logger.info("Initialization Done")

    def forward(self, x):
        x = self.inp_quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        self.logger.info("Initializing model")
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
        self.logger.info("Initializing model with custom weights & bias")
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


def make_layers(cfg, batch_norm, bit_width):
    layers = []
    in_channels = 3
    assert not(batch_norm & RETURN_QUANT_TENSOR), "nn.BatchNorm2d does not accept Quant tensor"
    for v in cfg:
        if v == 'M':
            layers += [qnn.QuantMaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = qnn.QuantConv2d(in_channels, v, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1, 
                                    groups=1, 
                                    weight_bit_width=bit_width, 
                                    bias_quant=BIAS_QUANTIZER,
                                    weight_quant=WEIGHT_QUANTIZER, 
                                    return_quant_tensor=RETURN_QUANT_TENSOR)
            conv2d.cache_inference_quant_out = True
            conv2d.cache_inference_quant_bias = True
            act = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=RETURN_QUANT_TENSOR)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)

def print_config(logger):
    logger.info("Brevitas version: {}".format(brevitas.__version__))
    logger.info("BIAS_QUANTIZER: {}".format(BIAS_QUANTIZER))
    logger.info("WEIGHT_QUANTIZER: {}".format(WEIGHT_QUANTIZER))
    logger.info("ACT_QUANTIZER: {}".format(ACT_QUANTIZER))