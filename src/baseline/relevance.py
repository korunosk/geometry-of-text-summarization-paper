import numpy as np
from scipy.spatial.distance import cdist
import ot


def word_mover_distance(document_embs: np.array, summary_embs: np.array) -> float:
    ''' Calculates the Word Mover distance between the document and summary distributions. '''
    return ot.emd2([], [], cdist(document_embs, summary_embs))


def lex_rank(document_embs: np.array, summary_embs: np.array, lr_scores: np.array) -> float:
    ''' Calculates the LexRank score of the summary based on the documents LexRank score. '''
    return np.sum(lr_scores[np.argmax(cdist(summary_embs, document_embs, metric='cosine'), axis=1)])