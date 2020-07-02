from gensim.parsing.preprocessing import *

from bert_serving.client import BertClient

from .loaders import *
from .helpers import *


FILTERS = [
    lambda s: s.lower(),
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
]


def encode_sentences(documents, encode):
    document_embs = []
    for document in documents:
        sentence_embs = []
        for sentence in document:
            sentence_embs.extend(encode(sentence))
        document_embs.append(sentence_embs)
    return document_embs


def make_encoder_lsa():
    embedding_method = 'LSA'
    item_id = 'tac-300d'
    vocab, embs = load_embeddings(embedding_method, item_id)
    
    sentences = read_sentences()
    vectorizer = make_vectorizer(sentences)

    def encode(sentence):
        X = vectorizer.transform([sentence])
        return [ list(embs[j]) for i, j in zip(*X.nonzero()) for _ in range(X[i, j]) ]

    return lambda documents: encode_sentences(documents, encode)


def make_encoder_glove():
    embedding_method = 'GloVe'
    item_id = 'glove.42B.300d'
    vocab, embs = load_embeddings(embedding_method, item_id)

    def encode(sentence):
        words = preprocess_string(sentence, FILTERS)
        return [ list(embs[vocab[w]]) for w in words if w in vocab ]
    
    return lambda documents: encode_sentences(documents, encode)


def make_encoder_fasttext():
    embedding_method = 'fasttext'
    item_id = 'crawl-300d-2M'
    vocab, embs = load_embeddings(embedding_method, item_id)

    def encode(sentence):
        words = preprocess_string(sentence, FILTERS)
        return [ list(embs[vocab[w]]) for w in words if w in vocab ]
    
    return lambda documents: encode_sentences(documents, encode)


def make_encoder_bert_word():
    def extract(document_embs_list, words_list):
        valid_embs = []
        for document_embs, words in zip(document_embs_list, words_list):
            for i in range(1, len(words)-1):
                if words[i] in STOPWORDS or words[i] in PUNCTUATION:
                    continue
                valid_embs.append(document_embs[i])
        return valid_embs
    
    def encode_sentences(documents):
        bc = BertClient(port=5557, port_out=5558, output_fmt='list')
        n = np.cumsum([0] + list(map(len, documents)))
        document_embs_list, words_list = bc.encode(list(chain(*documents)), show_tokens=True)
        bc.close()
        return [ extract(document_embs_list[b:e], words_list[b:e]) for b, e in zip(n[:-1], n[1:]) ]
    
    return encode_sentences


def make_encoder_bert_sent():
    def encode_sentences(documents):
        bc = BertClient(port=5557, port_out=5558, output_fmt='list')
        n = np.cumsum([0] + list(map(len, documents)))
        document_embs_list = bc.encode(list(chain(*documents)))
        bc.close()
        return [ document_embs_list[b:e] for b, e in zip(n[:-1], n[1:]) ]
    
    return encode_sentences


ENCODERS = [
    make_encoder_lsa,
    make_encoder_glove,
    make_encoder_fasttext,
    make_encoder_bert_word,
    make_encoder_bert_sent
]
