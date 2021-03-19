# BSD 3-Clause License
# Copyright (c) Alessandro Pappalardo 2019,
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Based on the torchvision implementation
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py


import torch
import torch.nn as nn
from brevitas.nn import QuantIdentity
from .common import make_quant_conv2d, make_quant_linear, make_quant_relu
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int8Bias

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class QuantVGG(nn.Module):

    def __init__(self, VGG_type='A', batch_norm=False, bit_width=8, num_classes=1000):
        super(QuantVGG, self).__init__()
        self.inp_quant = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)
        self.features = make_layers(cfgs[VGG_type], batch_norm, bit_width)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            make_quant_linear(512 * 7 * 7, 4096, bias=True, bit_width=bit_width),
            make_quant_relu(bit_width),
            nn.Dropout(),
            make_quant_linear(4096, 4096, bias=True, bit_width=bit_width),
            make_quant_relu(bit_width),
            nn.Dropout(),
            make_quant_linear(4096, num_classes, bias=False, bit_width=bit_width,
                              weight_scaling_per_output_channel=False),
        )
        self.features.cache_inference_quant_out = True
        self.features.cache_inference_quant_bias = True
        self._initialize_weights()

    def forward(self, x):
        x = self.features(self.inp_quant(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
                if(type(m.bias) == torch.nn.parameter.Parameter):
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm, bit_width):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = make_quant_conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, groups=1,
            bias=not batch_norm, bit_width=bit_width,weight_quant=Int8WeightPerTensorFixedPoint,bias_quant=Int8Bias)
            act = make_quant_relu(bit_width)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)