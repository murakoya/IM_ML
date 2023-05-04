import numpy as np
import torch
import torch.nn as nn

class myCNN(nn.Module):
    '''
    CNN architecture
    
    memo:
    if there is no padding, size of image after Conv2D is (size of input image) - ((kernel size) -1)
    In pooling, if size of input image is odd number, size of image after pooling is int((size of input image)/(kernel size)) + 1
    '''
    def __init__(self):
        super().__init__()
        # layer1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0), # conv_1,out:[126,126,32]
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0), # conv_2,out:[124,124,32]
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0), # conv_3,out:[122,122,64]
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0), # conv_4,out:[120,120,64]
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0), # conv_5,out:[118,118,128]
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0), # conv_6,out:[118,118,64]
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0), # conv_7,out[116,116,128]
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2) # out:[58,58,128]
        )
        
        # layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0), # conv_8,out:[56,56,256]
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0), # conv_9,out:[56,56,128]
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0), # conv_10,out:[54,54,256]
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2) # out:[27,27,256]
        )

        # layer3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0), # conv_11,out:[25,25,512]
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0), # conv_12,out:[25,25,256]
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0), # conv_13,out:[23,23,512]
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, ceil_mode=True) # out:[12,12,512]
        )

        # layer4
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0), # conv_14,out:[10,10,512]
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0), # conv_15,out:[10,10,256]
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0), # conv_16,out:[8,8,512]
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0), # conv_17,out:[8,8,256]
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0), # conv_18,out:[6,6,512]
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        # global average pooling
        self.gap = nn.AvgPool2d(kernel_size=6)
        
        # fully connected layer
        self.fc = nn.Linear(in_features = 512, out_features = 2)
        
    # forward propagation
    def forward(self, x):
        # CNNへの入力は(batch_size, channel, height, width)である必要がある
        # ref, https://tzmi.hatenablog.com/entry/2020/02/16/170928
        #x = x.permute(0,3,1,2)
        x = self.layer1(x) # layer1
        x = self.layer2(x) # layer2
        x = self.layer3(x) # layer3
        x = self.layer4(x) # layer4
        x = self.gap(x)
        x = x.reshape(x.size(0), -1) # converted to column vector
        x = self.fc(x) # fully connected
        return x        
