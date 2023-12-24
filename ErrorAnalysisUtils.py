import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import FeatureMapsExtractor as fme

from Enums import ModelName, ModelMethod
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import List, Tuple


def showRandomImages(listImages: List[str],
                     listLabels: List[str],
                     n: int = 15, seed: int = 0):

    randIndexes: np.array = np.random.randint(0, n,
                                              size=len(listImages))

    f, axes = plt.subplots(n // 5, 5, figsize=(15, 6))
    axes = axes.flatten()  # Turn it to 1D
    for i, ax in enumerate(axes):
        img_path: str = listImages[randIndexes[i]]
        label: str = listLabels[randIndexes[i]]
        ax.imshow(plt.imread(img_path))
        ax.set_title(label)
        ax.set_axis_off()
    plt.show()


def showRandomIncorrectlyClassified(listImages: List[str],
                                    listLabels: List[str],
                                    listPredicted: List[str],
                                    title: str = "Resnet Predicting Validation data",
                                    n: int = 10,
                                    seed: int = 0):
    np.random.seed(seed)
    randIndexes: np.array = np.random.randint(0, n,
                                              size=len(listImages))
    f, axes = plt.subplots(n // 5, 5, figsize=(25, 15))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img_path: str = listImages[randIndexes[i]]
        label: str = listLabels[randIndexes[i]]
        pred: str = listPredicted[randIndexes[i]]
        ax.imshow(img_path)
        ax.set_title("True: %s Predicted %s" % (label, pred))
        ax.set_axis_off()
    f.suptitle(title)
    plt.show()


def extractListImagesAndLabels(df: pd.DataFrame, image_col_name: str,
                               label_col_name: str) -> Tuple[List, List]:
    listImages: List[str] = df[image_col_name].values
    listLabels: List[str] = df[label_col_name].values

    return listImages, listLabels


def predictLoader(model: Module,
                  dataLoader: DataLoader,
                  device: str = "cpu"):

    predictions: List[int] = []
    targets: List[int] = []
    images: List[torch.Tensor] = []
    model.eval()
    for data, target in dataLoader:
        target: torch.Tensor = target.type(torch.LongTensor)
        data: torch.Tensor = data.to(device)
        target = target.to(device)
        predicted = model(data)

        predictions.append(torch.max(predicted, dim=1)[1])
        targets.append(target)
        images.append(data)

    return images, targets, predictions


def changeToNormalImage(listImages: Tuple[torch.Tensor]):
    normalImages = []
    for indx, image in enumerate(listImages):
        image = np.array(image.permute(1, 2, 0).numpy(), np.float32)
        image = image * (1 / image.max())  # Make it to 0 - 1
        normalImages.append(image)
    return normalImages


def getFalsePrediction(listTargets: List[torch.Tensor],
                       listPredicts: List[torch.Tensor],
                       listImages: List[torch.Tensor],
                       batch_indx: int = 0,
                       ):
    allFalseImages, allFalsePredictions, allFalseTargets = [], [], []
    for indx in range(len(listTargets)):
        npTargets: np.ndarray = listTargets[indx].numpy()
        npPredicts: np.ndarray = listPredicts[indx].numpy()
        listImages_tf: Tuple[torch.Tensor] = torch.unbind(listImages[indx])
        npImages: np.ndarray = np.array(changeToNormalImage(listImages_tf))
        falseIndexes: np.ndarray = np.where(npTargets != npPredicts)
        falseImages = npImages[falseIndexes]
        falsePredictions = npPredicts[falseIndexes]
        falseTargets = npTargets[falseIndexes]

        for falseImg, falseTarg, falsePred in zip(falseImages, falseTargets, falsePredictions):
            allFalseImages.append(falseImg)
            allFalseTargets.append(falseTarg)
            allFalsePredictions.append(falsePred)
    return allFalseImages, allFalseTargets, allFalsePredictions


def runAlgorithm(config,
                 model: Module,
                 modelName: ModelName,
                 loader: DataLoader,
                 device: str = "cpu",
                 loader_type_validation: bool = True):
    title: str = ""
    pre_title: str = "Validation Result" if loader_type_validation else "Testing Result"
    title = pre_title + " - Model - " + modelName.name
    imgs, targets, preds = predictLoader(model, loader, device=device)
    f_imgs, f_targets, f_preds = getFalsePrediction(imgs, targets, preds)
    showRandomIncorrectlyClassified(f_imgs, f_targets,
                                    f_preds, n=len(f_imgs),
                                    title=title)
