import os
import numpy as np
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_documents
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize

from src.util.helpers import *


class TACDatasetRegression(Dataset):

    def __init__(self, embedding_method, dataset_id, data):
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.data = data

        np.random.shuffle(data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i = self.data[idx][1]
        x = (load_embedded_item(self.embedding_method, self.dataset_id, topic_id, 'document_embs'),
             load_embedded_item(self.embedding_method, self.dataset_id, topic_id, f'summary_{i}_embs'))
        y = float(self.data[idx][2])
        
        return (x, y)


class TACDatasetClassification(Dataset):

    def __init__(self, embedding_method, dataset_id, data):
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.data = data

        np.random.shuffle(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        topic_id = self.data[idx][0]
        
        i1 = self.data[idx][1]
        i2 = self.data[idx][2]
        x = (load_embedded_item(self.embedding_method, self.dataset_id, topic_id, 'document_embs'),
             load_embedded_item(self.embedding_method, self.dataset_id, topic_id, f'summary_{i1}_embs'),
             load_embedded_item(self.embedding_method, self.dataset_id, topic_id, f'summary_{i2}_embs'))
        y = float(self.data[idx][3])
        
        return (x, y)


class Normalize():
    def __call__(self, sample):
        x, y = sample
        return (tuple(normalize(x_i, axis=1) for x_i in x), y)


class ToTensor():
    def __call__(self, sample):
        x, y = sample
        return (tuple(torch.tensor(x_i, dtype=torch.float) for x_i in x), torch.tensor(y, dtype=torch.float))
