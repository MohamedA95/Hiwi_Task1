import torch.nn as nn
from .common import make_quant_conv2d, make_quant_linear, make_quant_relu
from .quant_vgg import QuantVGG
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,VGG_type='A'):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(cfgs[VGG_type])
        self.fcs= nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )
    def forward(self, inputs):
        inputs = self.conv_layers(inputs)
        inputs = inputs.reshape(inputs.shape[0],-1)
        inputs = self.fcs(inputs)
        return inputs

    def create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels 
        for layer in arch:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            else:
                out_channels = layer
                layers+=[nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(layer),nn.ReLU()]
                in_channels=layer
        return nn.Sequential(*layers)