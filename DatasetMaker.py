"""
    Dataset preparation file.
    Created by : Vos
    Date : Today
"""


import torch
from torch.utils.data import (Dataset, DataLoader)
import pandas as pd
import os
from PIL import Image
from torchvision.transforms import v2
from typing import List, Dict


def getImgTransformation(resize_img: int,
                         transformation_type: str = "train"):
    if (transformation_type == "test"):
        img_test_transform: v2.Compose = v2.Compose([
            v2.Resize(resize_img),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return img_test_transform

    img_transformation: v2.Compose = v2.Compose([
        v2.Resize((resize_img, resize_img)),
        v2.RandomResizedCrop(size=(resize_img, resize_img), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.RandomErasing(p=0.5, scale=(0.1, 0.15)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    return img_transformation


class SmokerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame,
                 img_size: int,
                 transform_type: str = "train"):
        self.df: pd.DataFrame = dataframe
        self.transformations: v2.Compose = getImgTransformation(img_size, transform_type)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path: str = self.df.iloc[index, 0]
        img: Image = Image.open(image_path).convert("RGB")
        transformed_img: torch.Tensor = self.transformations(img)
        class_id: int = self.df.iloc[index, -1]
        return transformed_img, class_id


class MainSmokerDataset():
    def __init__(self, trainData: str, validationData: str,
                 testingData: str = None, numClasses: int = 2,
                 classes: List[str] = ["notsmoking", "smoking"],
                 columns_df: List[str] = []
                 ):
        self.trainDataPath: str = trainData
        self.validationDataPath: str = validationData
        self.testingDataPath: str = None
        self.classes: List[str] = classes
        self.c: torch.Tensor
        if (testingData is not None and type(testingData) is str):
            self.testingDataPath: str = testingData
        # Train Dataframe preps
        self.trainDataFrame: pd.DataFrame = pd.DataFrame(columns=columns_df)
        self.validationDataFrame: pd.DataFrame = pd.DataFrame(columns=columns_df)
        self.testDataFrame: pd.DataFrame = pd.DataFrame(columns=columns_df)
        # Dataset torch for loader
        self.trainDataset: Dataset = None
        self.validationDataset: Dataset = None
        self.testDataset: Dataset = None
        # DataLoader torch for training
        self.trainDataLoader: DataLoader = None
        self.validationDataLoader: DataLoader = None
        self.testDataLoader: DataLoader = None
        # This just default column names
        self.columnsDf: List[str] = columns_df

    def makeDataFrame(self, targetPath: str):
        print(targetPath)
        image_list: List[str] = os.listdir(targetPath)
        localDataFrame: pd.DataFrame = pd.DataFrame(columns=self.columnsDf)
        for image in image_list:
            fileName: str = os.path.splitext(image)[0].split("/")[-1]
            if fileName[0:len(self.classes[0])] == self.classes[0]:
                dictData: Dict[str, str] = {self.columnsDf[0]: os.path.join(targetPath, image),
                                            self.columnsDf[1]: self.classes[0],
                                            self.columnsDf[2]: 0
                                            }
                newData: pd.DataFrame = pd.DataFrame(dictData, index=[1])
                localDataFrame = pd.concat([localDataFrame, newData],
                                           ignore_index=True)
            elif fileName[0:len(self.classes[1])] == self.classes[1]:
                dictData: Dict[str, str] = {self.columnsDf[0]: os.path.join(targetPath, image),
                                            self.columnsDf[1]: self.classes[1],
                                            self.columnsDf[2]: 1
                                            }
                newData: pd.DataFrame = pd.DataFrame(dictData, index=[1])
                localDataFrame = pd.concat([localDataFrame, newData],
                                           ignore_index=True)
        # Assigning df
        if (targetPath == self.trainDataPath):
            self.trainDataFrame = localDataFrame
        elif (targetPath == self.validationDataPath):
            self.validationDataFrame = localDataFrame
        elif (targetPath == self.testingDataPath):
            self.testDataFrame = localDataFrame
    # Make Dataset instance

    def makeDataset(self, img_size: int):
        self.trainDataset: SmokerDataset = SmokerDataset(self.trainDataFrame,
                                                         img_size)
        self.validationDataset: SmokerDataset = SmokerDataset(self.validationDataFrame,
                                                              img_size)
        self.testDataset: SmokerDataset = SmokerDataset(self.testDataFrame,
                                                        img_size)

    def makeDataLoader(self, batchSize: int):
        self.trainDataLoader = DataLoader(self.trainDataset,
                                          batch_size=batchSize,
                                          shuffle=True)
        self.validationDataLoader = DataLoader(self.validationDataset,
                                               batch_size=batchSize,
                                               )
        self.testDataLoader = DataLoader(self.testDataset)
