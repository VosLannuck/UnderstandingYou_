import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import json 
import torch.nn as nn

from matplotlib.pyplot import Axes
from torch.nn import (Conv2d, Module)
from torch import Tensor
from enum import Enum
from PIL import Image
from torchvision import models, transforms, utils
from typing import List, Tuple

class ModelName(Enum):
    ALEXNET: int = 1
    RESNET: int = 2
    RESNET_18: int = 3
    VGG: int = 4


transform: transforms.Compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.2224, 0.225])
])


def openImage(path: str) -> Image:
    image: Image = Image.open(path)
    return image


def loadPretrained(modelName: ModelName) -> torch.nn.Module:
    model: torch.nn.Module
    if (modelName == ModelName.ALEXNET):
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    elif (modelName == ModelName.RESNET):
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif (modelName == ModelName.VGG):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif (modelName == ModelName.RESNET_18):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return model


def loadModelFromPath(modelName: str):
    ...

def loadModel(modelName: ModelName, type_model: str = "pretrained") -> Module:
    model: Module
    if (type_model.lower() == "local"):
        model = loadModelFromPath(modelName)
    else:
        model = loadPretrained(modelName)
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
        elif (type(model_children[i]) is nn.Sequential):
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
    print("Total Conv Layer: ", layer_counter)
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


def plotFilters(filters: List[Tensor]):
    fig: plt.Figure = plt.figure(figsize=(20, 15))
    for i in range(len(filters)):
        axes: Axes = fig.add_subplot(5, 4, i+1)
        _ = plt.imshow(filters[i])
        axes.axis("off")
        axes.set_title("Just layer", fontdict={"fontsize": 30})


def run(modelName: str, type_model: str = "pretrained",
        device: str = "cpu", image_path: str = "dog.jpg"):
    modelName = modelName.lower()
    model: Module
    if (modelName == "resnet"):
        model = loadModel(ModelName.RESNET_18)
    elif (modelName == "alex"):
        model = loadModel(ModelName.ALEXNET)
    elif (modelName == "vgg"):
        model = loadModel(ModelName.VGG)
    model = model.to(device)
    model_weights, conv_layers = extractFeatureMaps(model)
    image = applyImageTransformation(image_path, device)
    outputs_conv, layers_info = generateFeatureMapsFromInput(image,
                                                             conv_layers)
    filters: List[Tensor] = flattenFilters(outputs_conv)
    plotFilters(filters)


# if __name__ == "__main__":
#    run("resnet")
