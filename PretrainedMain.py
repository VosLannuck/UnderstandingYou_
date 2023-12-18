import torch
from torch.nn import Module, Linear
from omegaconf import OmegaConf, ListConfig, DictConfig
from typing import Union, List, Dict, Tuple
from torchvision.models import vgg16, resnet18, alexnet, swin_v2_s
from torchvision.models import VGG16_Weights, ResNet18_Weights, AlexNet_Weights
from torchvision.models.swin_transformer import SwinTransformer

from Enums import ModelName

config: Union[DictConfig, ListConfig] = OmegaConf.load("pretrained.yaml")


def preservePretrained(modelName: ModelName,
                       nc: int = 2) -> Tuple[Module, List[str]]:
    net: Module
    features_layers, fc_layers, opt_layer = [], [], []
    if (modelName == ModelName.pretrained_vgg):
        net: Module = vgg16(weights=VGG16_Weights.DEFAULT)
        net.classifier[6] = Linear(in_features=4096, out_features=2)
        features_layers, fc_layers, opt_layer = defineTrainableParamsForVGG()
    elif (modelName == ModelName.pretrained_alex):
        net: Module = alexnet(weights=AlexNet_Weights.DEFAULT)
        net.classifier[6] = Linear(in_features=4096, out_features=2)
        features_layers, fc_layers, opt_layer = defineTrainableParamsForAlex()
    elif (modelName == ModelName.pretrained_resnet):
        net: Module = resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = Linear(in_features=512, out_features=2)
        features_layers, fc_layers, opt_layer = defineTrainableParamsForResnet()
    else:
        raise Exception("Nah bro you'be playing riddle")
    return net, features_layers, fc_layers, opt_layer


def getTrainableLayer(modelName: ModelName, is_freeze: bool = True) -> Module:
    model, features_layers, fc_layers, opt_layer = preservePretrained(modelName)
    if (is_freeze):
        for name, params in model.named_parameters():
            params.requires_grad = False
            if name in opt_layer:
                params.requires_grad = True
    else:
        for name, params in model.named_parameters():
            print(name)
            if (name in features_layers):
                params.requires_grad = True
            elif name in fc_layers:
                params.requires_grad = True
            else:
                params.requires_grad = False

            if name in opt_layer:
                params.requires_grad = True

    # print([p.requires_grad for p in model.parameters()])
    return model


def defineTrainableParamsForAlex():
    update_features_conv: List[str] = [config.alex.ft.ft_0_w,
                                       config.alex.ft.ft_0_b,
                                       config.alex.ft.ft_3_w,
                                       config.alex.ft.ft_3_b,
                                       config.alex.ft.ft_6_w,
                                       config.alex.ft.ft_6_b,
                                       config.alex.ft.ft_8_w,
                                       config.alex.ft.ft_8_b,
                                       config.alex.ft.ft_10_w,
                                       config.alex.ft.ft_10_b,
                                       ]
    update_fc_layer: List[str] = [
                                  config.alex.cl.cl_1_w,
                                  config.alex.cl.cl_1_b,
                                  config.alex.cl.cl_4_w,
                                  config.alex.cl.cl_4_b,
                                 ]
    update_opt_layer: List[str] = [config.alex.cl.cl_6_w,
                                   config.alex.cl.cl_6_b
                                   ]
    return update_features_conv, update_fc_layer, update_opt_layer


def defineTrainableParamsForVGG() -> List[str]:
    update_features_conv: List[str] = [
                                  config.vgg16.ft.cv_0_w,
                                  config.vgg16.ft.cv_0_b,
                                  config.vgg16.ft.cv_2_w,
                                  config.vgg16.ft.cv_2_b,
                                  config.vgg16.ft.cv_5_w,
                                  config.vgg16.ft.cv_5_b,
                                  config.vgg16.ft.cv_7_w,
                                  config.vgg16.ft.cv_7_b,
                                  config.vgg16.ft.cv_10_w,
                                  config.vgg16.ft.cv_10_b,
                                  config.vgg16.ft.cv_12_w,
                                  config.vgg16.ft.cv_12_b,
                                  config.vgg16.ft.cv_14_w,
                                  config.vgg16.ft.cv_14_b,
                                  config.vgg16.ft.cv_17_w,
                                  config.vgg16.ft.cv_17_b,
                                  config.vgg16.ft.cv_19_w,
                                  config.vgg16.ft.cv_19_b,
                                  config.vgg16.ft.cv_21_w,
                                  config.vgg16.ft.cv_21_b,
                                  config.vgg16.ft.cv_24_w,
                                  config.vgg16.ft.cv_24_b,
                                  config.vgg16.ft.cv_26_w,
                                  config.vgg16.ft.cv_26_b,
                                  config.vgg16.ft.cv_28_b,
                                  config.vgg16.ft.cv_28_w,
                                 ]

    update_fc_layer: List[str] = [config.vgg16.cl.cl_0_w,
                                  config.vgg16.cl.cl_0_b,
                                  config.vgg16.cl.cl_3_w,
                                  config.vgg16.cl.cl_3_b,
                                  ]

    update_opt_layer: List[str] = [config.vgg16.cl.cl_6_w,
                                   config.vgg16.cl.cl_6_b]

    return update_features_conv, update_fc_layer, update_opt_layer


def defineTrainableParamsForResnet():
    update_features_conv: List[str] = [config.res.ft.cv_1_w,
                                       config.res.ft.cv_l1_01_w,
                                       config.res.ft.cv_l1_02_w,
                                       config.res.ft.cv_l1_11_w,
                                       config.res.ft.cv_l1_12_w,
                                       config.res.ft.cv_l2_01_w,
                                       config.res.ft.cv_l2_02_w,
                                       config.res.ft.cv_l2_11_w,
                                       config.res.ft.cv_l2_12_w,
                                       config.res.ft.cv_l3_01_w,
                                       config.res.ft.cv_l3_02_w,
                                       config.res.ft.cv_l3_11_w,
                                       config.res.ft.cv_l3_12_w,
                                       config.res.ft.cv_l4_01_w,
                                       config.res.ft.cv_l4_02_w,
                                       config.res.ft.cv_l4_11_w,
                                       config.res.ft.cv_l4_12_w,
                                       ]
    update_fc_layer = []
    update_opt_layer = [config.res.fc.fc_w,
                        config.res.fc.fc_b]
    return update_features_conv, update_fc_layer, update_opt_layer

#model, _,_,_ = preservePretrained(ModelName.vit)
#print(model.train())

#getTrainableLayer(ModelName.pretrained_vgg, is_freeze=False)
