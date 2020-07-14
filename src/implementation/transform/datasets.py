import numpy as np
import torch
from torch.utils.data import Dataset


class TACDatasetRegressionRouge(Dataset):

    def __init__(self, dataset, data):
        self.dataset = dataset
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i = self.data[idx][1]
        e = self.dataset[topic_id]['documents']['embs'][i]
        y = torch.tensor(self.data[idx][2], dtype=torch.float)
        
        return e, y


class TACDatasetRegression(Dataset):

    def __init__(self, dataset, data):
        self.dataset = dataset
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i  = self.data[idx][1]
        d  = self.dataset[topic_id]['documents']['embs']
        si = self.dataset[topic_id][f'summary_{i}']['embs']
        a  = self.dataset[topic_id]['documents']['aux']
        ai = self.dataset[topic_id][f'summary_{i}']['aux']
        y  = torch.tensor(self.data[idx][2], dtype=torch.float)
        
        return d, si, a, ai, y


class TACDatasetClassification(Dataset):

    def __init__(self, dataset, data):
        self.dataset = dataset
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i  = self.data[idx][1]
        j  = self.data[idx][2]
        d  = self.dataset[topic_id]['documents']['embs']
        si = self.dataset[topic_id][f'summary_{i}']['embs']
        sj = self.dataset[topic_id][f'summary_{j}']['embs']
        a  = self.dataset[topic_id]['documents']['aux']
        ai = self.dataset[topic_id][f'summary_{i}']['aux']
        aj = self.dataset[topic_id][f'summary_{j}']['aux']
        y  = torch.tensor(self.data[idx][3], dtype=torch.float)
        
        return d, si, sj, a, ai, aj, y


def repeat_mean(x: list, M: int) -> np.array:
    ''' Repeats the mean of a tensor several times along the vertical axis. '''
    x = np.mean(x, 0)
    x = np.tile(x, (M, 1))
    return x


def pad(x: list, M: int) -> (np.array, np.array):
    ''' Pads a tensor with zero-valued vectors along the vertical axis and
    creates a mask to designate the original tensor's vectors. '''
    m, n = len(x), len(x[0])
    p = np.zeros((M, n), dtype=np.float)
    p[:m] = x
    mask = np.zeros(M, dtype=np.bool)
    mask[:m] = True
    return p, mask


def pad_h(x: list, M: int) -> (np.array, np.array):
    ''' Pads a tensor with zero-valued vectors along the vertical axis and
    creates the histogram for the original tensor's vectors. '''
    m, n = len(x), len(x[0])
    p = np.zeros((M, n), dtype=np.float)
    p[:m] = x
    hist = np.zeros(M, dtype=np.float)
    hist[:m] = 1.0 / m
    return p, hist
