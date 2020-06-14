import os
import json
import orjson
import numpy as np

from src.config import *


def load_rouge_scores(dataset_id):
    fname = os.path.join(BASE_DATA_DIR, dataset_id) + '_rouge.json'
    with open(fname, mode='r') as fp:
        return orjson.loads(fp.read())


def save_rouge_scores(dataset_id, scores):
    fname = os.path.join(BASE_DATA_DIR, dataset_id) + '_rouge.json'
    with open(fname, mode='w') as fp:
        json.dump(scores, fp, indent=4)


def load_embeddings(embedding_method, item_id):
    fname = os.path.join(EMBEDDINGS_DIR, embedding_method, item_id)
    with open(fname + '.vocab', mode='r') as fp:
        vocab = { line.strip(): i for i, line in enumerate(fp.readlines()) }
    embs = np.load(fname + '.npy', allow_pickle=True)
    return vocab, embs


def save_embeddings(embedding_method, item_id, vocab, embs):
    directory = os.path.join(EMBEDDINGS_DIR, embedding_method)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, item_id)
    with open(fname + '.vocab', mode='w') as fp:
        fp.write(vocab)
    np.save(fname + '.npy', embs)


def load_dataset(dataset_id):
    fname = os.path.join(BASE_DATA_DIR, dataset_id) + '.json'
    with open(fname, mode='r') as fp:
        return orjson.loads(fp.read())


def load_embedded_topic(embedding_method, dataset_id, topic_id):
    fname = os.path.join(EMBEDDINGS_DIR, embedding_method, dataset_id, topic_id) + '_encoded.json'
    with open(fname, mode='r') as fp:
        return orjson.loads(fp.read())


def save_embedded_topic(embedding_method, dataset_id, topic_id, topic):
    directory = os.path.join(EMBEDDINGS_DIR, embedding_method, dataset_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, topic_id) + '_encoded.json'
    with open(fname, mode='w') as fp:
        json.dump(topic, fp, indent=4)


def load_embedded_item(embedding_method, dataset_id, topic_id, item_id):
    fname = os.path.join(EMBEDDINGS_DIR, embedding_method, dataset_id, topic_id, item_id) + '.npy'
    return np.load(fname)


def save_embedded_item(embedding_method, dataset_id, topic_id, item_id, item):
    directory = os.path.join(EMBEDDINGS_DIR, embedding_method, dataset_id, topic_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, item_id) + '.npy'
    np.save(fname, item)


def load_train_data(dataset_id, item_id):
    fname = os.path.join(DATA_DIR, dataset_id, item_id) + '.npy'
    return np.load(fname)


def save_train_data(dataset_id, item_id, item):
    directory = os.path.join(DATA_DIR, dataset_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, item_id) + '.npy'
    np.save(fname, item)
