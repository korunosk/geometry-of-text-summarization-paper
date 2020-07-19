from .trainer import ModelTrainer
from src.implementation.transform.preprocessing import stratified_sampling
from src.util.loaders import (
    load_train_data,
    save_model
)


def train_model_1(embedding_method, dataset_id, layer):
    data = load_train_data(dataset_id, 'regression_rouge')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    trainer = ModelTrainer(embedding_method, dataset_id, layer)
    model = trainer.train_nn_rouge_reg_model(train, val)

    save_model(embedding_method, dataset_id, layer, 'nn_rouge_reg_model', model)


def train_model_2(embedding_method, dataset_id, layer):
    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    trainer = ModelTrainer(embedding_method, dataset_id, layer)
    model = trainer.train_nn_wavg_pr_model(train, val)

    save_model(embedding_method, dataset_id, layer, 'nn_wavg_pr_model', model)


def train_model_3(embedding_method, dataset_id, layer):
    data = load_train_data(dataset_id, 'regression')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    trainer = ModelTrainer(embedding_method, dataset_id, layer)
    model = trainer.train_lin_sinkhorn_reg_model(train, val)

    save_model(embedding_method, dataset_id, layer, 'lin_sinkhorn_reg_model', model)


def train_model_4(embedding_method, dataset_id, layer):
    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    trainer = ModelTrainer(embedding_method, dataset_id, layer)
    model = trainer.train_lin_sinkhorn_pr_model(train, val)

    save_model(embedding_method, dataset_id, layer, 'lin_sinkhorn_pr_model', model)


def train_model_5(embedding_method, dataset_id, layer):
    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    trainer = ModelTrainer(embedding_method, dataset_id, layer)
    model = trainer.train_nn_sinkhorn_pr_model(train, val)

    save_model(embedding_method, dataset_id, layer, 'nn_sinkhorn_pr_model', model)


def train_model_6(embedding_method, dataset_id, layer):
    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    trainer = ModelTrainer(embedding_method, dataset_id, layer)
    model = trainer.train_cond_lin_sinkhorn_pr_model(train, val)

    save_model(embedding_method, dataset_id, layer, 'cond_lin_sinkhorn_pr_model', model)


PROCEDURES = [
    train_model_1,
    train_model_2,
    train_model_3,
    train_model_4,
    train_model_5,
    train_model_6
]
