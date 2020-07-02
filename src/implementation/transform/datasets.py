import os
import numpy as np
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_documents
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize

from src.util.helpers import *


class TACDatasetRegression(Dataset):

    def __init__(self, embedding_method, dataset_id, data, transform=None):
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i = self.data[idx][1]
        x = (load_embedded_item(self.embedding_method, self.dataset_id, topic_id, 'document_embs'),
             load_embedded_item(self.embedding_method, self.dataset_id, topic_id, f'summary_{i}_embs'))
        y = float(self.data[idx][2])
        
        if self.transform is not None:
            return self.transform((x, y))
        return (x, y)


class TACDatasetRegressionRouge(Dataset):

    def __init__(self, embedding_method, dataset_id, data, transform=None):
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i = int(self.data[idx][1])
        document_embs = load_embedded_item(self.embedding_method, self.dataset_id, topic_id, 'document_embs')
        s = (document_embs[i],)
        y = float(self.data[idx][2])
        
        if self.transform is not None:
            return self.transform((s, y))
        return (s, y)


class TACDatasetClassification(Dataset):

    def __init__(self, embedding_method, dataset_id, data, transform=None):
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i1 = self.data[idx][1]
        i2 = self.data[idx][2]
        x = (load_embedded_item(self.embedding_method, self.dataset_id, topic_id, 'document_embs'),
                load_embedded_item(self.embedding_method, self.dataset_id, topic_id, f'summary_{i1}_embs'),
                load_embedded_item(self.embedding_method, self.dataset_id, topic_id, f'summary_{i2}_embs'))
        y = int(self.data[idx][3])
        
        if self.transform is not None:
            return self.transform((x, y))
        return (x, y)


class TACDatasetLoadedClassification(Dataset):

    def __init__(self, dataset, data):
        self.dataset = dataset
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i1 = self.data[idx][1]
        i2 = self.data[idx][2]
        d = self.dataset[topic_id]['document_embs']
        s1 = self.dataset[topic_id][f'summary_{i1}_embs']
        s2 = self.dataset[topic_id][f'summary_{i2}_embs']
        m1 = self.dataset[topic_id][f'mask_{i2}']
        m2 = self.dataset[topic_id][f'mask_{i2}']
        y = torch.tensor(float(self.data[idx][3]), dtype=torch.float)
        
        return d, s1, s2, m1, m2, y


class Normalize():
    def __call__(self, sample):
        x, y = sample
        return (tuple(normalize(x_i, axis=1) for x_i in x), y)


class ToTensor():
    def __call__(self, sample):
        x, y = sample
        return (tuple(torch.tensor(x_i, dtype=torch.float) for x_i in x), torch.tensor(y, dtype=torch.float))


def repeat_mean(x: torch.tensor, M: int) -> torch.tensor:
    ''' Repeats the mean of a tensor several times along the vertical axis. '''
    x = torch.mean(x, axis=0)
    x = x.repeat(M, 1)
    return x

def pad(x: torch.tensor, M: int) -> (torch.tensor, torch.tensor):
    ''' Pads a tensor with zero-valued vectors along the vertical axis and
    creates a mask to designate the original tensor's vectors. '''
    m, n = x.shape

    p = torch.zeros(size=(M, n), dtype=torch.float)
    p[:m,:] = x

    mask = torch.zeros(size=(M, ), dtype=torch.bool)
    mask[:m] = True
    mask = mask.view(-1, 1)

    return p, mask

class Expand(): 
    def __init__(self, M):
        self.M = M

    def __call__(self, sample):
        (d, s1, s2), y = sample

        d = repeat_mean(d, self.M)
        s1, m1 = pad(s1, self.M)
        s2, m2 = pad(s2, self.M)

        return d, s1, s2, m1, m2, y
