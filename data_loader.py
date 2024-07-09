import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
# include number of columns in the dataset from 0 to n,
# if total number of columns are 5 then number of classes should be 4
numberOfClasses = 4
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        # print(self.data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, :numberOfClasses].values, dtype=torch.float32)
        # print(self.data.iloc[idx, 2])
        class_name = torch.tensor(self.data.iloc[idx, numberOfClasses], dtype=torch.long)
       # print(features.shape)

        if self.transform:
            features = self.transform(features)

        return features, class_name


class CustomDataset_Norm(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Assuming the last column is the label and the rest are features
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values

        # Compute mean and std for normalization
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)

        # Normalize the features
        self.features = (self.features - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label
