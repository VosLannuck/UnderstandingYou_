#%%
import torch

from Baseline_CNN import CNN_Baseline
from sklearn.metrics import classification_report
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import AdamW

import argparse
import time
import mlflow

from typing import Dict
#%%
INIT_LR : float = 1e-3
BATCH_SIZE : int = 64
EPOCH_SIZE : int = 10

TRAIN_SPLIT : float = 0.75
VAL_SPLIT : float = 1 - TRAIN_SPLIT

device : str = torch.device("cuda" if torch.cuda.is_available() else "cpua")

arg : argparse.ArgumentParser = argparse.ArgumentParser()
arg.add_argument("-m", "--model", type=str, required=True, help="Path to output trained model")
arg.add_argument("-p", "--plot", type=str, required=True, help="Path to output loss/accracy plot")
arg.add_argument("-b", "--batch", type=int, default=BATCH_SIZE )
arg.add_argument("-e", "--epoch", type=int, default=EPOCH_SIZE)
arg.add_argument("-lr","--learning_rate", type=int, default=INIT_LR)

args = vars(arg.parse_args())
print(args)


# %%
