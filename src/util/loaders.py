import os
import json
import orjson
import numpy as np
import torch
from src.config import (
    BASE_DATA_DIR,
    EMBEDDINGS_DIR,
    DATA_DIR,
    MODELS_DIR
)


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


def load_embedded_topic(embedding_method, dataset_id, layer, topic_id):
    directory = os.path.join(EMBEDDINGS_DIR, embedding_method, dataset_id)
    if layer:
        directory = os.path.join(directory, str(layer))
    fname = os.path.join(directory, topic_id) + '_encoded.json'
    with open(fname, mode='r') as fp:
        return orjson.loads(fp.read())


def save_embedded_topic(embedding_method, dataset_id, layer, topic_id, topic):
    directory = os.path.join(EMBEDDINGS_DIR, embedding_method, dataset_id)
    if layer:
        directory = os.path.join(directory, str(layer))
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, topic_id) + '_encoded.json'
    with open(fname, mode='w') as fp:
        json.dump(topic, fp, indent=4)


def load_embedded_item(embedding_method, dataset_id, layer, topic_id, item_id):
    directory = os.path.join(EMBEDDINGS_DIR, embedding_method, dataset_id)
    if layer:
        directory = os.path.join(directory, str(layer))
    directory = os.path.join(directory, topic_id)
    fname = os.path.join(directory, item_id) + '.npy'
    return np.load(fname, allow_pickle=True)


def save_embedded_item(embedding_method, dataset_id, layer, topic_id, item_id, item):
    directory = os.path.join(EMBEDDINGS_DIR, embedding_method, dataset_id)
    if layer:
        directory = os.path.join(directory, str(layer))
    directory = os.path.join(directory, topic_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, item_id) + '.npy'
    np.save(fname, item)


def load_train_data(dataset_id, item_id):
    fname = os.path.join(DATA_DIR, dataset_id, item_id) + '.npy'
    return np.load(fname, allow_pickle=True)


def save_train_data(dataset_id, item_id, item):
    directory = os.path.join(DATA_DIR, dataset_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, item_id) + '.npy'
    np.save(fname, item)


def load_model(embedding_method, dataset_id, layer, model_id, Model, config):
    directory = os.path.join(MODELS_DIR, embedding_method, dataset_id)
    if layer:
        directory = os.path.join(directory, str(layer))
    fname = os.path.join(directory, model_id) + '.pt'
    model = Model(config)
    model.load_state_dict(torch.load(fname))
    model.eval()
    return model


def save_model(embedding_method, dataset_id, layer, model_id, model):
    directory = os.path.join(MODELS_DIR, embedding_method, dataset_id)
    if layer:
        directory = os.path.join(directory, str(layer))
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, model_id) + '.pt'
    torch.save(model.state_dict(), fname)
