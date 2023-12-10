import torch
from torch.nn import (Module, Conv2d,
                      Linear, MaxPool2d,
                      Flatten,
                      ReLU, Dropout, BatchNorm2d)

VGG_DEFAULT_KERNEL_SIZE: int = 3
VGG_DEFAULT_STRIDE_SIZE: int = 1
VGG_DEFAULT_PADDING: int = 1

VGG_DEFAULT_POOL_SIZE: int = 2
VGG_DEFAULT_POOL_STRIDE: int = 2


class VGG_Baseline_16(Module):

    def __init__(self, numClasses: int = 2):
        super(VGG_Baseline_16, self).__init__()

        self.layer_1 = torch.nn.Sequential(
            Conv2d(
                    in_channels=3, out_channels=64,
                    kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                    stride=VGG_DEFAULT_STRIDE_SIZE,
                    padding=VGG_DEFAULT_PADDING,
                    ),
            BatchNorm2d(64),
            ReLU(),
        )
        self.layer_2: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(
                kernel_size=VGG_DEFAULT_POOL_SIZE,
                stride=VGG_DEFAULT_POOL_STRIDE)
        )

        self.layer_3: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(128),
            ReLU()
        )

        self.layer_4: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,

            ),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_POOL_STRIDE
            ),
        )

        self.layer_5: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=128, out_channels=256,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(256),
            ReLU(),
        )

        self.layer_6: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=256, out_channels=256,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(256),
            ReLU(),
        )

        self.layer_7: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=256, out_channels=256,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_POOL_STRIDE
            )
        )

        self.layer_8: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=256, out_channels=512,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(512),
            ReLU(),
        )

        self.layer_9: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(512),
            ReLU()
        )

        self.layer_10: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_POOL_STRIDE
                )
        )

        self.layer_11: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(512),
            ReLU()
        )

        self.layer_12: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(512),
            ReLU()
        )

        self.layer_13: torch.nn.Sequential = torch.nn.Sequential(
            Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_STRIDE_SIZE,
                padding=VGG_DEFAULT_PADDING,
            ),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(
                kernel_size=VGG_DEFAULT_KERNEL_SIZE,
                stride=VGG_DEFAULT_POOL_STRIDE
            )
        )

        self.fc_layer_1: torch.nn.Sequential = torch.nn.Sequential(
            Dropout(0.5),
            Linear(6*6*512, 4096),
            ReLU()
        )

        self.fc_layer_2: torch.nn.Sequential = torch.nn.Sequential(
            Dropout(0.5),
            Linear(4096, 4096),
            ReLU()
        )

        self.fc_layer_3: torch.nn.Sequential = torch.nn.Sequential(
            Linear(4096, numClasses),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layer_1(input)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)
        out = self.layer_8(out)
        out = self.layer_9(out)
        out = self.layer_10(out)
        out = self.layer_11(out)
        out = self.layer_12(out)
        out = self.layer_13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_layer_1(out)
        out = self.fc_layer_2(out)
        out = self.fc_layer_3(out)

        return out
