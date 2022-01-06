import torch
import torch.nn as nn
from base import BaseModel


class LeNet(BaseModel):
    def __init__(self,batchnorm=False):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            *make_conv2d(3,32,batchnorm,3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            *make_conv2d(32,64,batchnorm,3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.classifier = nn.Sequential(
            nn.Linear(64*6*6, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        return out

def make_conv2d(in_channels,out_channels,batchnorm,kernel_size=3):
    layers=[nn.Conv2d(in_channels,out_channels,kernel_size)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    return layers