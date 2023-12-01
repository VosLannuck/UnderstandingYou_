#%%

import torch
from torch.nn import (Module, )
from torch.utils.data import (Dataset, DataLoader)

import pandas as pd
import numpy as np
import os

class MainSmokerDataset(Dataset):
    def __init__(self, trainData : str, validationData : str):
        super(MainSmokerDataset).__init__()
        
        self.trainDataPath : str = trainData
        self.validationDatPath : str = validationData
        
        

