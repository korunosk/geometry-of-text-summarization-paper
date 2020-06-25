import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

import torch
from torch.utils.tensorboard import SummaryWriter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import *


def make_pytorch_projector(log_dir, embeddings, global_step):
    ''' Exports PyTorch projector '''
    writer = SummaryWriter(log_dir)
    writer.add_embedding(embeddings['mat'],
                         metadata=embeddings['labels'],
                         tag=embeddings['tag'],
                         global_step=global_step)
    writer.close()


def project_pca(embs, t, n_components=2):
    ''' Projects embeddings using PCA '''
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pts = pca.fit_transform(embs)
    return pts[:t], pts[t:]


def project_tsne(embs, t):
    ''' Projects embeddings using t-SNE '''
    tsne = TSNE(n_components=2, perplexity=30, n_iter=5000, verbose=1, random_state=RANDOM_STATE)
    pts = tsne.fit_transform(embs)
    return pts[:t], pts[t:]