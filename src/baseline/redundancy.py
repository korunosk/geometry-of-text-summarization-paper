import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from scipy.spatial.distance import cdist


def average_pairwise_distance(summary_embs: np.array) -> float:
    ''' Calculates the average pairwise distance between summary embeddings '''
    return np.mean(cdist(summary_embs, summary_embs, metric='euclidean'))


def semantic_volume(summary_embs: np.array) -> float:
    ''' Calculates the semantic volume of the summary embeddings '''
    try:
        return ConvexHull(summary_embs).volume
    except QhullError as e:
        return 0


def semantic_spread(summary_embs: np.array) -> float:
    ''' Calculates the semantic spread of the summary embeddings '''
    if summary_embs.shape[0] == 1:
        return 0
    return np.linalg.det(summary_embs @ summary_embs.T)