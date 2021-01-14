from .trainer import ModelTrainer
from src.implementation.transform.preprocessing import cross_validation_sampling
from src.util.loaders import (
    load_train_data,
    save_model
)


def train_nn_rouge_reg_model(embedding_method, dataset_id, layer, device_id):
    data = load_train_data(dataset_id, 'regression_rouge')
    for i, train, val in cross_validation_sampling(data):
        print(len(train), len(val))

        print(f'Model {i + 1}')

        trainer = ModelTrainer(embedding_method, dataset_id, layer, device_id)
        model = trainer.train_nn_rouge_reg_model(train, val)

        save_model(embedding_method, dataset_id, layer, f'nn_rouge_reg_model_{i}', model)


def train_nn_wavg_pr_model(embedding_method, dataset_id, layer, device_id):
    data = load_train_data(dataset_id, 'classification')
    for i, train, val in cross_validation_sampling(data):
        print(len(train), len(val))

        print(f'Model {i + 1}')

        trainer = ModelTrainer(embedding_method, dataset_id, layer, device_id)
        model = trainer.train_nn_wavg_pr_model(train, val)

        save_model(embedding_method, dataset_id, layer, f'nn_wavg_pr_model_{i}', model)


def train_lin_sinkhorn_reg_model(embedding_method, dataset_id, layer, device_id):
    data = load_train_data(dataset_id, 'regression')
    for i, train, val in cross_validation_sampling(data):
        print(len(train), len(val))

        print(f'Model {i + 1}')

        trainer = ModelTrainer(embedding_method, dataset_id, layer, device_id)
        model = trainer.train_lin_sinkhorn_reg_model(train, val)

        save_model(embedding_method, dataset_id, layer, f'lin_sinkhorn_reg_model_{i}', model)


def train_lin_sinkhorn_pr_model(embedding_method, dataset_id, layer, device_id):
    data = load_train_data(dataset_id, 'classification')
    for i, train, val in cross_validation_sampling(data):
        print(len(train), len(val))

        print(f'Model {i + 1}')

        trainer = ModelTrainer(embedding_method, dataset_id, layer, device_id)
        model = trainer.train_lin_sinkhorn_pr_model(train, val)

        save_model(embedding_method, dataset_id, layer, f'lin_sinkhorn_pr_model_{i}', model)


def train_nn_sinkhorn_pr_model(embedding_method, dataset_id, layer, device_id):
    data = load_train_data(dataset_id, 'classification')
    for i, train, val in cross_validation_sampling(data):
        print(len(train), len(val))

        print(f'Model {i + 1}')

        trainer = ModelTrainer(embedding_method, dataset_id, layer, device_id)
        model = trainer.train_nn_sinkhorn_pr_model(train, val)

        save_model(embedding_method, dataset_id, layer, f'nn_sinkhorn_pr_model_{i}', model)


def train_cond_lin_sinkhorn_pr_model(embedding_method, dataset_id, layer, device_id):
    data = load_train_data(dataset_id, 'classification')
    for i, train, val in cross_validation_sampling(data):
        print(len(train), len(val))

        print(f'Model {i + 1}')

        trainer = ModelTrainer(embedding_method, dataset_id, layer, device_id)
        model = trainer.train_cond_lin_sinkhorn_pr_model(train, val)

        save_model(embedding_method, dataset_id, layer, f'cond_lin_sinkhorn_pr_model_{i}', model)

        import torch
        torch.cuda.empty_cache()


def train_cond_nn_wavg_pr_model(embedding_method, dataset_id, layer, device_id):
    data = load_train_data(dataset_id, 'classification')
    for i, train, val in cross_validation_sampling(data):
        print(len(train), len(val))

        print(f'Model {i + 1}')

        trainer = ModelTrainer(embedding_method, dataset_id, layer, device_id)
        model = trainer.train_cond_nn_wavg_pr_model(train, val)

        save_model(embedding_method, dataset_id, layer, f'cond_nn_wavg_pr_model_{i}', model)

        import torch
        torch.cuda.empty_cache()


PROCEDURES = [
    train_nn_rouge_reg_model,
    train_nn_wavg_pr_model,
    train_lin_sinkhorn_reg_model,
    train_lin_sinkhorn_pr_model,
    train_nn_sinkhorn_pr_model,
    train_cond_lin_sinkhorn_pr_model,
    train_cond_nn_wavg_pr_model
]
