from sklearn.utils.extmath import randomized_svd
from src.util.helpers import (
    read_sentences,
    make_vectorizer
)
from src.util.loaders import save_embeddings
from src.config import RANDOM_STATE


if __name__ == '__main__':

    sentences = read_sentences()
    vectorizer = make_vectorizer(sentences)

    X = vectorizer.transform(sentences)

    U, Sigma, VT = randomized_svd(X, n_components=300, random_state=RANDOM_STATE)

    V = VT.T
    bigrams = vectorizer.get_feature_names()
    
    embedding_method = 'LSA'
    item_id = 'tac-300d'
    vocab = '\n'.join(bigrams)
    embs = V
    save_embeddings(embedding_method, item_id, vocab, embs)
        