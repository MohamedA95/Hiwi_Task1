import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8WeightPerTensorFloat
from brevitas.quant import Int8Bias, Int8BiasPerTensorFloatInternalScaling, Int16Bias, IntBias

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
BIAS_QUANTIZER = Int8Bias

class QuantVGG_pure(nn.Module):

    def __init__(self, VGG_type='A', batch_norm=False, bit_width=8, num_classes=1000):
        super(QuantVGG_pure, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.features = make_layers(cfgs[VGG_type], batch_norm, bit_width)
        self.avgpool = qnn.QuantAdaptiveAvgPool2d((7, 7))        
        self.classifier = nn.Sequential(
            qnn.QuantLinear(512 * 7 * 7, 4096, bias=True, weight_bit_width=bit_width, bias_quant=BIAS_QUANTIZER,act_quant=Int8ActPerTensorFloat, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True),
            qnn.QuantDropout(),
            qnn.QuantLinear(4096, 4096, bias=True, weight_bit_width=bit_width, bias_quant=BIAS_QUANTIZER, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True),
            qnn.QuantDropout(),
            qnn.QuantLinear(4096, num_classes, bias=False, weight_bit_width=bit_width),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
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


def make_layers(cfg, batch_norm, bit_width):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [qnn.QuantMaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = qnn.QuantConv2d(in_channels, v, kernel_size=3, stride=1, padding=1, groups=1, weight_bit_width=bit_width, bias_quant=BIAS_QUANTIZER, return_quant_tensor=True)
            act = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)