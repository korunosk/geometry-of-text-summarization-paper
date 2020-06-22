import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

from src.implementation.transform.datasets import *
from src.implementation.transform.models import *
from src.implementation.transform.preprocessing import *
from src.util.loaders import *
from src.config import *
from src.config_models import *

torch.manual_seed(RANDOM_STATE)
cuda = torch.device('cuda:1')


def train_model_1(embedding_method, dataset_id):
    config = CONFIG_MODELS['NNRougeRegModel']

    data = load_train_data(dataset_id, 'regression_rouge')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    transform = transforms.Compose([ToTensor()])
    dataset = TACDatasetRegressionRouge(embedding_method, dataset_id, train)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = NNRougeRegModel(config).to(device=cuda)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    loss = []

    for batch in data_loader:
        (s, ), y = transform(batch)

        y_hat = model(s.to(device=cuda))
        y = -torch.log(y + 0.00001).to(device=cuda)

        L = criterion(y_hat, y)

        L.backward()

        optimizer.step()

        optimizer.zero_grad()

        loss.append(L.item())

        print(f'Train Loss: {loss[-1]:.4f}')

    save_model(embedding_method, dataset_id, 'nn_rouge_reg_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_2(embedding_method, dataset_id):
    config = CONFIG_MODELS['NNWAvgPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    transform = transforms.Compose([ToTensor()])
    dataset = TACDatasetClassification(embedding_method, dataset_id, train)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x: x)

    model = NNWAvgPRModel(config).to(device=cuda)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    loss = []

    for batch in data_loader:

        for sample in batch:
            (d, s1, s2), y = transform(sample)
            
            y_hat = model(d.to(device=cuda),
                        s1.to(device=cuda),
                        s2.to(device=cuda))
            
            L = criterion(y_hat, y.to(device=cuda))
            
            L.backward()
            
            loss.append(L.item())
            
        optimizer.step()

        optimizer.zero_grad()

        print(f'Train Loss: {loss[-1]:4f}')

    save_model(embedding_method, dataset_id, 'nn_wavg_pr_model', model)

    n = config['batch_size']
    loss = [sum(loss[i:i+n])/n for i in range(0,len(loss),n)]

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_2_batch(embedding_method, dataset_id):
    config = CONFIG_MODELS['NNWAvgPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    transform = transforms.Compose([ToTensor(), Expand(20)])
    dataset = TACDatasetClassification(embedding_method, dataset_id, train, transform=transform)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = NNWAvgPRBatchModel(config).to(device=cuda)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    loss = []

    for i, batch in enumerate(data_loader):
        d, s1, s2, m1, m2, y = batch

        y_hat = model(d.to(device=cuda),
                    s1.to(device=cuda),
                    s2.to(device=cuda),
                    m1.to(device=cuda),
                    m2.to(device=cuda))

        y_hat = y_hat.squeeze()

        L = criterion(y_hat, y.to(device=cuda))

        L.backward()

        optimizer.step()

        optimizer.zero_grad()

        loss.append(L.item())

        if i % 100 == 0:
            print(f'Train Loss: {loss[-1]:.4f}')

    save_model(embedding_method, dataset_id, 'nn_wavg_pr_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_3(embedding_method, dataset_id):
    config = CONFIG_MODELS['LinSinkhornRegModel']

    data = load_train_data(dataset_id, 'regression')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    transform = transforms.Compose([Normalize(), ToTensor()])
    dataset = TACDatasetRegression(embedding_method, dataset_id, train)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x: x)

    model = LinSinkhornRegModel(config).to(device=cuda)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    loss = []

    optimizer.zero_grad()

    for batch in data_loader:
        
        for sample in batch:
            (d, s), y = transform(sample)

            y_hat = model(d.to(device=cuda),
                          s.to(device=cuda))
            
            L = criterion(y_hat, y.to(device=cuda))
            
            L.backward()
            
            loss.append(L.item())
            
        optimizer.step()
        
        optimizer.zero_grad()
        
        print(f'Train Loss: {loss[-1]:.4f}')

    save_model(embedding_method, dataset_id, 'lin_sinkhorn_reg_model', model)

    # n = config['batch_size']
    # loss = [sum(loss[i:i+n])/n for i in range(0,len(loss),n)]

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_4(embedding_method, dataset_id):
    pass


def train_model_5(embedding_method, dataset_id):
    pass
