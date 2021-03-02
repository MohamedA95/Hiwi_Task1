import torch.nn as nn

VGG_types = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,VGG_type='A'):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[VGG_type])
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
            if layer == 'm':
                layers.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            else:
                out_channels = layer
                layers+=[nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=(1,1),padding=(1,1)),nn.BatchNorm2d(layer),nn.ReLU()]
                in_channels=layer
        return nn.Sequential(*layers)

# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
