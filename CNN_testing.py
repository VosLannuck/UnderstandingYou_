
from torch.nn import (Module, Conv2d, Linear,
                      MaxPool2d,)
import torch
from torch import nn


class CNN_Testing(Module):

    def __init__(self, numChannels: int, classes: int):

        super(CNN_Testing, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32,
                                     kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32,
                                     kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(10816, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, classes)
    # Progresses data across layers    

    def forward(self, x: torch.Tensor):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
