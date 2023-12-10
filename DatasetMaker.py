#%%
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
from glob import glob

from torchvision.transforms import v2
from typing import List, Dict
from torch.nn.functional import one_hot
IMG_SIZE_TESTING: int = 64
RESIZE_IMG: int = IMG_SIZE_TESTING # 224
BATCH_SIZE: int = 16

DEFAULT_IMG_TRAIN_TRANSFORMATION: v2.Compose = v2.Compose([
    v2.Resize((RESIZE_IMG, RESIZE_IMG)),
    v2.RandomResizedCrop(size=(RESIZE_IMG, RESIZE_IMG), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.RandomErasing(p=0.5, scale=(0.1, 0.15)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DEFAULT_IMG_TEST_TRANSFORMATION: v2.Compose = v2.Compose([
    v2.Resize(RESIZE_IMG),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SmokerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame,
                 transforms: v2.Compose = DEFAULT_IMG_TRAIN_TRANSFORMATION):
        self.df: pd.DataFrame = dataframe
        self.transformations: v2.Compose = transforms

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

    def makeDataset(self):
        self.trainDataset: SmokerDataset = SmokerDataset(self.trainDataFrame)
        self.validationDataset: SmokerDataset = SmokerDataset(self.validationDataFrame)
        self.testDataset: SmokerDataset = SmokerDataset(self.testDataFrame)

    def makeDataLoader(self, batchSize: int):
        self.trainDataLoader = DataLoader(self.trainDataset,
                                          batch_size=batchSize,
                                          shuffle=True)
        self.validationDataLoader = DataLoader(self.validationDataset,
                                               batch_size=batchSize,
                                               )
        self.testDataLoader = DataLoader(self.testDataset)
