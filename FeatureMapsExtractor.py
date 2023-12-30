#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import json 
import torch.nn as nn

from matplotlib.pyplot import Axes
from torch.nn import (Conv2d, Module, Linear)
from torch import Tensor
from enum import Enum
from PIL import Image
from torchvision import models, transforms, utils
from typing import List, Tuple

from CNN_testing import CNN_Testing
from Baseline_CNN import CNN_Baseline
from Baseline_AlexNet import AlexNet
from Baseline_VGG import VGG_Baseline_16
from Baseline_Resnet import ResidualBlock, BaselineResnet 
from typing import Union
from Enums import ModelName, ModelMethod
from omegaconf import OmegaConf, DictConfig, ListConfig
from torchvision.models import Swin_T_Weights
from torchvision.models.swin_transformer import SwinTransformer

config: Union[DictConfig, ListConfig] = OmegaConf.load("params.yaml")
NUM_CLASSES: int = 2

transform: transforms.Compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.2224, 0.225])
])


def openImage(path: str) -> Image:
    image: Image = Image.open(path)
    return image


def loadPretrained(modelMethod: ModelMethod, path: str = None) -> torch.nn.Module:
    model: torch.nn.Module
    if (modelMethod == ModelMethod.ALEXNET):
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.classifier[6] = Linear(in_features=4096,
                                     out_features=NUM_CLASSES,
                                     )
    elif (modelMethod == ModelMethod.RESNET):
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = Linear(in_features=512, out_features=NUM_CLASSES)
    elif (modelMethod == ModelMethod.VGG):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = Linear(in_features=4096,
                                     out_features=NUM_CLASSES,
                                     )
    elif (modelMethod == ModelMethod.VIT):
        model: SwinTransformer = models.swin_v2_t(weights="DEFAULT")
        model.head = Linear(in_features=model.head.in_features,
                                out_features=NUM_CLASSES)
    if path:
        model.load_state_dict(torch.load(path))
    return model


def loadModelFromPath(modelMethod: ModelMethod, path: str) -> nn.Module:
    model: nn.Module = CNN_Testing(numChannels=3, classes=NUM_CLASSES)
    if (modelMethod == ModelMethod.CNN):
        model = CNN_Baseline(numChannels=3, classes=NUM_CLASSES)
    elif (modelMethod == ModelMethod.ALEXNET):
        model = AlexNet(num_classes=NUM_CLASSES)
    elif (modelMethod == ModelMethod.VGG):
        model = VGG_Baseline_16(numClasses=NUM_CLASSES)
    elif (modelMethod == ModelMethod.RESNET):
        model = BaselineResnet(ResidualBlock, numClasses=NUM_CLASSES)
    elif (modelMethod == ModelMethod.VIT):
        model = CNN_Testing(numChannels=NUM_CLASSES)
    model.load_state_dict(torch.load(path))
    return model



def loadModel(modelMethod: ModelMethod,
              type_model: str = "pretrained",
              path: str = None) -> Module:
    model: Module
    if (type_model.lower() == "local"):
        model = loadModelFromPath(modelMethod, path)
    else:
        model = loadPretrained(modelMethod)
    return model


def extractFeatureMaps(model: Module) -> Tuple[List[Tensor], List[Module]]:
    model_weights: List[Tensor] = []
    conv_layers: List[Conv2d] = []
    layer_counter: int = 1
    model_children: List[Module] = list(model.children())

    for i in range(len(model_children)):
        if (type(model_children[i]) is Conv2d):
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
            layer_counter += 1
        elif (type(model_children[i]) is (nn.Sequential)):
            for j in range(len(model_children[i])):
                layer: Module = model_children[i][j]
                if (len(list(layer.children())) > 1):
                    for child in layer.children():
                        if (type(child) is Conv2d):
                            model_weights.append(child.weight)
                            conv_layers.append(child)
                            layer_counter += 1
                else:
                    if (type(layer) is Conv2d):
                        model_weights.append(layer.weight)
                        conv_layers.append(layer)
                        layer_counter += 1
    print("Total Conv Layer: ", len(conv_layers))
    return model_weights, conv_layers


def applyImageTransformation(image_path: str, device: str) -> Tensor:
    image: Image = openImage(image_path)
    imageTensor: Tensor = transform(image)
    imageTensor: Tensor
    imageTensor = imageTensor.unsqueeze(0)
    imageTensor.to(device)
    return imageTensor


def generateFeatureMapsFromInput(image: Tensor,
                                 conv_layers: List[Conv2d]
                                 ) -> Tuple[List[Tensor], List[Module]]:
    outputs_conv: List[Tensor] = []
    layers_info: List[Module] = []

    for layer in conv_layers:
        image = layer(image)
        outputs_conv.append(image)
        layers_info.append(layer)

    return outputs_conv, layers_info


def flattenFilters(outputs_conv: List[Tensor]) -> List[Tensor]:
    flattenedFilter: List[Tensor] = []

    for feature_map in outputs_conv:
        totalFilters: int = feature_map.shape[1]
        feature_map = torch.squeeze(feature_map)
        feature_map = torch.sum(feature_map, dim=0)
        feature_map = feature_map / totalFilters
        flattenedFilter.append(torch.detach(feature_map).cpu().numpy())
    return flattenedFilter


def plotFilters(filters: List[Tensor], layers_info):
    fig: plt.Figure = plt.figure(figsize=(20, 15))
    for i in range(len(filters) ):
        axes: Axes = fig.add_subplot(9, 4, i+1)
        _ = plt.imshow(filters[i])
        axes.axis("off")
        axes.set_title("%s -%s" % (layers_info[i].__class__.__name__, len(filters)- i), fontdict={"fontsize": 30})


def extractResultFeaturemaps(model: Module, device: str = "cpu",
                             image_path="dog.jpg"):
    model = model.to(device)
    model_w, conv_layers = extractFeatureMaps(model)
    image = applyImageTransformation(image_path, device)
    outputs_conv, layers_info = generateFeatureMapsFromInput(image, conv_layers)
    filters = flattenFilters(outputs_conv)
    plotFilters(filters, layers_info)

def prepareImageForPrediction(path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(path)
    img = transform(img).unsqueeze(0)

    return img

def run(modelName: str, type_model: str = "pretrained", path: str= None,
        device: str = "cpu", image_path: str = "dog.jpg"):
    modelName = modelName.lower()
    model: Module
    if (modelName == "resnet"):
        model = loadModel(ModelMethod.RESNET, type_model, path)
    elif (modelName == "alex"):
        model = loadModel(ModelMethod.ALEXNET, type_model, path)
    elif (modelName == "vgg"):
        model = loadModel(ModelMethod.VGG, type_model, path)
    elif (modelName == "cnn"):
        model = loadModel(ModelMethod.CNN, type_model, path)
    with torch.no_grad():
        model = model.to(device)
        img_real = prepareImageForPrediction(image_path)
        model_weights, conv_layers = extractFeatureMaps(model)
        preds = model(img_real)
        torch.nn.functional.softmax(preds, dim=1)
        _, predicted = torch.max(preds, 1)
        
        image = applyImageTransformation(image_path, device)
        outputs_conv, layers_info = generateFeatureMapsFromInput(image,
                                                                 conv_layers)
        filters: List[Tensor] = flattenFilters(outputs_conv)
        plotFilters(filters, layers_info)
        torch.cuda.empty_cache()


#if __name__ == "__main__":
#    run("cnn", type_model="local", path=config.fme.cnn_testing_path_best)
