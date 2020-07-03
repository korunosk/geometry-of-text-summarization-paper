import torch
from transformers import BartTokenizer, BertTokenizer
from transformers import BartModel, BertModel

from gensim.parsing.preprocessing import *

from bert_serving.client import BertClient

from .loaders import *
from .helpers import *


FILTERS = [
    lambda s: s.lower(),
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    remove_stopwords,
    strip_short
]


def encode_sentences(documents, encode):
    document_embs = []
    for document in documents:
        sentence_embs = []
        for sentence in document:
            sentence_embs.extend(encode(sentence))
        document_embs.append(sentence_embs)
    return document_embs


def make_encoder_lsa(**kwargs):
    embedding_method = 'LSA'
    item_id = 'tac-300d'
    vocab, embs = load_embeddings(embedding_method, item_id)
    
    sentences = read_sentences()
    vectorizer = make_vectorizer(sentences)

    def encode(sentence):
        X = vectorizer.transform([sentence])
        return [ list(embs[j]) for i, j in zip(*X.nonzero()) for _ in range(X[i, j]) ]

    return lambda documents: encode_sentences(documents, encode)


def make_encoder_glove(**kwargs):
    embedding_method = 'GloVe'
    item_id = 'glove.42B.300d'
    vocab, embs = load_embeddings(embedding_method, item_id)

    def encode(sentence):
        words = preprocess_string(sentence, FILTERS)
        return [ list(embs[vocab[w]]) for w in words if w in vocab ]
    
    return lambda documents: encode_sentences(documents, encode)


def make_encoder_fasttext(**kwargs):
    embedding_method = 'fasttext'
    item_id = 'crawl-300d-2M'
    vocab, embs = load_embeddings(embedding_method, item_id)

    def encode(sentence):
        words = preprocess_string(sentence, FILTERS)
        return [ list(embs[vocab[w]]) for w in words if w in vocab ]
    
    return lambda documents: encode_sentences(documents, encode)


def make_encoder_bert_word(**kwargs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.eval()

    def encode(sentence):
        words = preprocess_string(sentence, FILTERS)
        if not words:
            return []
        inputs = tokenizer(words, is_pretokenized=True, return_tensors='pt')
        hidden_states = model(**inputs, output_hidden_states=True)[2]
        return hidden_states[kwargs['layer']].squeeze()[1:-1].data.tolist()

    return lambda documents: encode_sentences(documents, encode)


def make_encoder_bert_sent(**kwargs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.eval()

    def encode(sentence):
        words = preprocess_string(sentence, FILTERS)
        if not words:
            return []
        inputs = tokenizer(words, is_pretokenized=True, return_tensors='pt')
        hidden_states = model(**inputs, output_hidden_states=True)[2]
        return hidden_states[-2].squeeze()[1:-1].mean(axis=0).data.tolist()

    return lambda documents: encode_sentences(documents, encode)


def make_encoder_bart_word(**kwargs):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartModel.from_pretrained('facebook/bart-large')
    model = model.eval()

    def encode(sentence):
        words = preprocess_string(sentence, FILTERS)
        if not words:
            return []
        inputs = tokenizer(words, is_pretokenized=True, return_tensors='pt')
        hidden_states = model(**inputs, output_hidden_states=True)[3]
        return hidden_states[kwargs['layer']].squeeze()[1:-1].data.tolist()

    return lambda documents: encode_sentences(documents, encode)


ENCODERS = [
    make_encoder_lsa,
    make_encoder_glove,
    make_encoder_fasttext,
    make_encoder_bert_word,
    make_encoder_bert_sent,
    make_encoder_bart_word
]
