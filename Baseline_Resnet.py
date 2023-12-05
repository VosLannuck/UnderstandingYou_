import torch 

from torch.nn import (Conv2d, Linear, MaxPool2d,
                      Flatten, ReLU, BatchNorm2d, Sequential)

from typing import List


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: Sequential = None):

        super(ResidualBlock, self).__init__()
        self.conv_1: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels,
                   stride=stride, padding=1),
            BatchNorm2d(out_channels)
        )

        self.conv_2: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels,
                   stride=stride, padding=1
                   ),
            BatchNorm2d(out_channels)
        )

        self.downSample: torch.nn.Sequential = downsample
        self.relu: ReLU = ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = input
        out: torch.Tensor = self.conv_1(input)
        out = self.conv_2(2)
        if self.downSample:
            # Apply some computation to the residual, if any
            residual = self.downSample(residual)
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
        self.layer_1: Sequential = self._make_layer(residualBlock, 64,
                                                    total_residualBlocks[0],
                                                    stride=1
                                                    )
        self.layer_2: Sequential = self._make_layer(residualBlock, 128,
                                                    total_residualBlocks[1],
                                                    stride=2)
        self.layer_3: Sequential = self._make_layer(residualBlock, 256,
                                                    total_residualBlocks[2],
                                                    stride=2)
        self.layer_4: Sequential = self._make_layer(residualBlock, 512,
                                                    total_residualBlocks[3],
                                                    stride=2
                                                    )
        self.avg_pool: torch.nn.AvgPool2d = torch.nn.AvgPool2d(kernel_size=7,
                                                               stride=1)
        self.fc: torch.nn.Linear = Linear(512, numClasses)

    def _make_layer(self, residualBlock: ResidualBlock, output_channel: int,
                    total_residualBlock: int,
                    stride: int = 1) -> torch.nn.Sequential:
        downSample: torch.nn.Sequential = None
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
        for i in range(0, total_residualBlock-1):
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
        out = Flatten(out)
        return self.fc(out)
