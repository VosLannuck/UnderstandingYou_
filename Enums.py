from enum import Enum


class ModelName(Enum):
    vanilla_cnn = 1
    vanilla_vgg16 = 2
    vanilla_resnet = 3
    vanilla_alex = 4
    vit = 5

    pretrained_alex = 6
    pretrained_resnet = 7
    pretrained_vgg = 8

    testing = 9


class ModelMethod(Enum):
    CNN = 1
    ALEXNET = 2
    RESNET = 3
    VGG = 4
    VIT = 5
