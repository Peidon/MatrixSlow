import os
from abc import ABC

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class IrisDataset(Dataset, ABC):
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder()
