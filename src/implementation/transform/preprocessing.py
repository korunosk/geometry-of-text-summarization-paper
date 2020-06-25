import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from more_itertools import unique_everseen

from src.config import *


def stratified_sampling(data, test_size=0.3):
    train, test = train_test_split(data, test_size=test_size, random_state=RANDOM_STATE, stratify=data[:,0])
    return train, test


def leave_n_out(data, test_size=0.3):
    topics = pd.unique(data[:,0])
    n = int(test_size * len(topics))
    train_topics = topics[:-n]
    test_topics = topics[-n:]
    train = data[np.isin(data[:,0], train_topics)]
    test = data[np.isin(data[:,0], test_topics)]
    return train, test
