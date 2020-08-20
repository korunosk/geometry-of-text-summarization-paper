import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from src.config import RANDOM_STATE


def stratified_sampling(data, test_size=0.3):
    # np.random.shuffle(data)
    # data = data[:int(0.1*data.shape[0])]
    
    train, test = train_test_split(data, test_size=test_size, random_state=RANDOM_STATE, stratify=data[:,0])
    return train, test


def cross_validation_sampling(data, n_splits=3):
    # np.random.shuffle(data)
    # data = data[:int(0.1*data.shape[0])]
    
    topic_ids = pd.unique(data[:,0])
    for i, (topic_ids_train_idx, topic_ids_test_idx) in enumerate(KFold(n_splits=n_splits).split(topic_ids)):
        topic_ids_train, topic_ids_test = topic_ids[topic_ids_train_idx], topic_ids[topic_ids_test_idx]
        train = data[np.isin(data[:,0], topic_ids_train)]
        test = data[np.isin(data[:,0], topic_ids_test)]
        yield i, train, test
