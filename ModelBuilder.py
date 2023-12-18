import PretrainedMain

from Enums import ModelName
from torch.nn import Module, Linear

from omegaconf import DictConfig, ListConfig
from CNN_testing import CNN_Testing
from Baseline_CNN import CNN_Baseline
from Baseline_Resnet import BaselineResnet, ResidualBlock
from Baseline_VGG import VGG_Baseline_16
from Baseline_AlexNet import AlexNet
from typing import Union

from torchvision import models
from torchvision.models.swin_transformer import SwinTransformer


def parseToModelName(config: Union[DictConfig, ListConfig],
                     modelstr: str) -> ModelName:
    if (modelstr == config.constant.vanilla_cnn):
        return ModelName.vanilla_cnn
    elif (modelstr == config.constant.vanilla_alex):
        return ModelName.vanilla_alex
    elif (modelstr == config.constant.vanilla_resnet):
        return ModelName.vanilla_resnet
    elif (modelstr == config.constant.vanilla_vgg16):
        return ModelName.vanilla_vgg16
    elif (modelstr == config.constant.pre_vgg16):
        return ModelName.vanilla_vgg16
    elif (modelstr == config.constant.pre_alex):
        return ModelName.vanilla_alex
    elif (modelstr == config.constant.pre_resnet):
        return ModelName.vanilla_resnet
    else:
        return ModelName.testing


def preserveModel(modelName: ModelName,
                  device: str,
                  num_classes: int,
                 ) -> Module:
    if (modelName == ModelName.vanilla_cnn):
        cnnModel: CNN_Baseline = CNN_Baseline(3, classes=num_classes).to(device)
        return cnnModel
    elif (modelName == ModelName.vanilla_alex):
        alexNetModel: AlexNet = AlexNet(num_classes=num_classes).to(device)
        return alexNetModel
    elif (modelName == ModelName.vanilla_vgg16):
        vggModel: VGG_Baseline_16 = VGG_Baseline_16(numClasses=num_classes).to(device)
        return vggModel
    elif (modelName == ModelName.vanilla_resnet):
        resnetModel: BaselineResnet = BaselineResnet(ResidualBlock,
                                                     numClasses=num_classes).to(device)
        return resnetModel
    elif (modelName == ModelName.vit):
        vit_model: SwinTransformer = models.swin_v2_s(weights='DEFAULT').to(device)
        vit_model.head = Linear(in_features=vit_model.head.in_features,
                                out_features=num_classes).to(device)
        return vit_model
    elif (modelName == ModelName.testing):
        cnn_testing: CNN_Testing = CNN_Testing(3, classes=num_classes).to(device)
        return cnn_testing
    elif (modelName == ModelName.pretrained_alex):
        alex: Module = PretrainedMain.getTrainableLayer(modelName, is_freeze=True).to(device)
        return alex
    elif (modelName == ModelName.pretrained_vgg):
        vgg: Module = PretrainedMain.getTrainableLayer(modelName, is_freeze=True).to(device)
        return vgg
    elif (modelName == ModelName.pretrained_resnet):
        resnet: Module = PretrainedMain.getTrainableLayer(modelName, is_freeze=True).to(device)
        return resnet
