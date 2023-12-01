#%%
import torch 
from torch.nn import (Module, Conv2d, Linear,
                      MaxPool2d, ReLU,
                      Flatten, Dropout, LogSoftmax, )
from torch import flatten
from typing import Tuple

#%%a
DEFAULT_CNN_OUT_CHANNEL : int = 20
DEFAULT_CNN_KERNEL_SIZE : Tuple[int, int] = (5,5)
DEFAULT_CNN_POOL_KERNEL_SIZE : Tuple[int, int] = (2,2)
DEFAULT_CNN_POOL_STRIDE : Tuple[int, int] = (2,2)

DEFAULT_IN_FEATURES : int = 800
DEFAULT_OUT_FEATURES : int = 500

class CNN_Baseline(Module):

    def __init__(self, numChannels : int, classes : int):
        super(CNN_Baseline, self).__init__()
        
        self.convLayer_1 : Conv2d = Conv2d(in_channels=numChannels, out_channels=DEFAULT_CNN_OUT_CHANNEL, kernel_size=DEFAULT_CNN_KERNEL_SIZE)
        self.relu_1  : ReLU = ReLU()
        self.maxPool2D_1 : MaxPool2d = MaxPool2d(kernel_size=DEFAULT_CNN_POOL_KERNEL_SIZE,stride=DEFAULT_CNN_POOL_STRIDE )

        self.convLayer_2 : Conv2d = Conv2d(in_channels=DEFAULT_CNN_KERNEL_SIZE, out_channels=50 )
        self.relu_2 : ReLU = ReLU()
        self.maxPool2D_2 : MaxPool2d = MaxPool2d(kernel_size=DEFAULT_CNN_POOL_KERNEL_SIZE,
                                                 stride=DEFAULT_CNN_POOL_STRIDE)

        self.fc : Linear = Linear(in_features=800, out_features=500)
        self.relu_fc : ReLU = ReLU()
        
        ## Softmax a
        self.output : Linear = Linear(in_features=500, out_features=classes)
        self.logSoftmax : LogSoftmax = LogSoftmax(dim=1)

        
    def forward(self, input ) -> torch.Tensor:
        x : torch.Tensor = self.convLayer_1(input)
        x = self.relu_1(x)
        x = self.maxPool2D_1(x)
        
        x = self.convLayer_2(x)
        x = self.relu_2(x)
        x = self.maxPool2D_2(x)
        
        x = flatten(x,start_dim=1)
        x = self.fc(x)
        x = self.relu_fc(x)
        
        x = self.output(x)
        return self.logSoftmax(x)
        
        