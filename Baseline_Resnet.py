import torch 
from torch import nn

from torch.nn import (Conv2d, Linear, MaxPool2d,
                       ReLU, BatchNorm2d, Sequential)

from typing import List


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels,
                                  kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())

        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels,
                                  kernel_size=3, stride=1,
                                  padding=1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BaselineResnet(torch.nn.Module):

    def __init__(self, residualBlock: ResidualBlock,
                 total_residualBlocks: List[int] = [3, 4, 6, 3],
                 numClasses: int = 2):

        super(BaselineResnet, self).__init__()
        self.inChannel: int = 64
        self.conv_1: Sequential() = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                   stride=2, padding=3),
            BatchNorm2d(64),
            ReLU()
        )
        self.max_pool_1: MaxPool2d = MaxPool2d(kernel_size=3, stride=2,
                                               padding=1)
        self.layer_1: Sequential = self._make_layer(residualBlock, 64, 64,
                                                    total_residualBlocks[0],
                                                    stride=1
                                                    )
        self.layer_2: Sequential = self._make_layer(residualBlock, 64, 128,
                                                    total_residualBlocks[1],
                                                    stride=2)
        self.layer_3: Sequential = self._make_layer(residualBlock, 128, 256,
                                                    total_residualBlocks[2],
                                                    stride=2)
        self.layer_4: Sequential = self._make_layer(residualBlock, 256, 512,
                                                    total_residualBlocks[3],
                                                    stride=2
                                                    )
        self.avg_pool: torch.nn.AvgPool2d = torch.nn.AvgPool2d(kernel_size=7,
                                                               stride=1)
        self.fc: torch.nn.Linear = Linear(512, numClasses)

    def _make_layer(self, residualBlock: ResidualBlock, in_channel: int,
                    output_channel: int,
                    total_residualBlock: int,
                    stride: int = 1) -> torch.nn.Sequential:
        downSample: torch.nn.Sequential = None
        self.inChannel = in_channel
        if stride != 1 or self.inChannel != -1:
            downSample = torch.nn.Sequential(
                Conv2d(in_channels=self.inChannel, out_channels=output_channel,
                       stride=stride, kernel_size=1),
                BatchNorm2d(output_channel)
            )
        listOfLayers: List[torch.nn.Module] = []
        listOfLayers.append(residualBlock(self.inChannel, output_channel,
                                          stride, downSample))

        self.inChannel = output_channel  # For preserving another layer input
        for i in range(1, total_residualBlock):
            listOfLayers.append(residualBlock(self.inChannel,
                                              output_channel, ))

        return torch.nn.Sequential(*listOfLayers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv_1(input)
        out = self.max_pool_1(out)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
