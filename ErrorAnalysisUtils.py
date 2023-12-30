import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import FeatureMapsExtractor as fme
from Enums import ModelName, ModelMethod
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def showRandomImages(listImages: List[str],
                     listLabels: List[str],
                     n: int = 5, seed: int = 0):

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
                                    listLabels: List[int],
                                    listPredicted: List[int],
                                    listRawPreds: List[int],
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
        label: int = listLabels[randIndexes[i]]
        pred: int = listPredicted[randIndexes[i]]
        raw_value: int = listRawPreds[randIndexes[i]]
        ax.imshow(img_path)
        ax.set_title("True: %s ; Predicted: %s ; conf: %s" % (label, pred, raw_value))
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
    predicted_raw_max: List[float] = []
    model.eval()
    predicted_model_n_smoking: np.array = np.array([])
    predicted_model_smoking: np.array = np.array([])

    for data, target in dataLoader:
        target: torch.Tensor = target.type(torch.LongTensor)
        data: torch.Tensor = data.to(device)
        target = target.to(device)
        predicted = model(data)
        #  print(predicted)
        raw, prs_val_indx = torch.max(predicted, dim=1)
        #  print(raw)
        #  print("\n\n\n")
        predicted_model_smoking  = np.concatenate([predicted_model_smoking,
                                                   predicted[:, 1].detach().numpy()])
        predicted_model_n_smoking = np.concatenate([predicted_model_n_smoking,
                                                    predicted[:, 0].detach().numpy()])

        predicted_raw_max.append(raw)
        predictions.append(prs_val_indx)
        targets.append(target)
        images.append(data)

    return images, targets, predictions, predicted_model_smoking, predicted_model_n_smoking, predicted_raw_max


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
                       listRawPredicts,
                       batch_indx: int = 0,
                       ):
    allFalseImages, allFalsePredictions, allFalseTargets = [], [], []
    allFalseRawPredicts = []
    for indx in range(len(listTargets)):
        npTargets: np.ndarray = listTargets[indx].numpy()
        npPredicts: np.ndarray = listPredicts[indx].numpy()
        npRawPredicts: np.ndarray = listRawPredicts[indx].detach().numpy()
        listImages_tf: Tuple[torch.Tensor] = torch.unbind(listImages[indx])
        npImages: np.ndarray = np.array(changeToNormalImage(listImages_tf))
        falseIndexes: np.ndarray = np.where(npTargets != npPredicts)
        falseImages = npImages[falseIndexes]
        falsePredictions = npPredicts[falseIndexes]
        falseTargets = npTargets[falseIndexes]
        falseRawPredicts = npRawPredicts[falseIndexes]

        for falseImg, falseTarg, falsePred, falseRaw in zip(falseImages, falseTargets,
                                                  falsePredictions, falseRawPredicts):
            allFalseImages.append(falseImg)
            allFalseTargets.append(falseTarg)
            allFalsePredictions.append(falsePred)
            allFalseRawPredicts.append(falseRaw)

    npArrFalseTargets = np.array(allFalseTargets)
    indx_false_predict_smoking = np.where(npArrFalseTargets != 1)
    indx_false_predict_n_smoking = np.where(npArrFalseTargets != 0)


    print("Total Missclassification for smoking: ", len(npArrFalseTargets[indx_false_predict_smoking]), " from total ", len(allFalseImages), " data")
    print("Total Missclassification for Not Smoking: ", len(npArrFalseTargets[indx_false_predict_n_smoking]), "from total  ", len(allFalseImages), " data")
    return allFalseImages, allFalseTargets, allFalsePredictions, allFalseRawPredicts


def plotClassificationReport(targets: List[torch.Tensor],
                             preds:[torch.Tensor]):

    all_targets: List[torch.Tensor] = np.array([])
    all_preds: List[torch.Tensor] = np.array([])
    for indx,_ in enumerate(targets):
        all_targets = np.concatenate((all_targets,targets[indx]), axis=0)
        all_preds = np.concatenate((all_preds, preds[indx]), axis=0)
    print(classification_report(all_targets, all_preds))

    return all_targets, all_preds


def plotHistPlotComparasionPrediction(not_smoke, smoke):
    sns.histplot(x=not_smoke, label="not_smoke")
    sns.histplot(x=smoke, label="smoke")

    plt.title("Prediction Confidence")
    plt.legend()
    plt.show()


def runAlgorithm(config,
                 model: Module,
                 modelName: ModelName,
                 loader: DataLoader,
                 device: str = "cpu",
                 loader_type_validation: bool = True):
    print(f"Result for : ${modelName.name}")
    title: str = ""
    pre_title: str = "Validation Result" if loader_type_validation else "Testing Result"
    title = pre_title + " - Model - " + modelName.name
    imgs, targets, preds, predicted_smoking_conf, predicted_n_smoking_conf, predicted_raw = predictLoader(model, loader, device=device)
    plotClassificationReport(targets,
                             preds)

    plotHistPlotComparasionPrediction(predicted_smoking_conf, predicted_n_smoking_conf)
    f_imgs, f_targets, f_preds, f_raw_preds = getFalsePrediction(targets, preds, imgs, predicted_raw)
    showRandomIncorrectlyClassified(f_imgs, f_targets,
                                    f_preds, f_raw_preds,
                                    title=title,
                                    n=len(f_imgs))
