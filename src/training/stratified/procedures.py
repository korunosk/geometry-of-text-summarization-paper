from collections import defaultdict
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


def train_model_1_batch(embedding_method, dataset_id, **kwargs):
    config = CONFIG_MODELS['NNRougeRegModel']

    data = load_train_data(dataset_id, 'regression_rouge')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    transform = transforms.Compose([ToTensor()])
    dataset = TACDatasetRegressionRouge(embedding_method, dataset_id, train, **kwargs)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = NNRougeRegModel(config).to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    loss = []

    for batch in data_loader:
        (s, ), y = transform(batch)

        y_hat = model(s.to(device=device))
        y = -torch.log(y + 0.00001).to(device=device)

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


def train_model_2(embedding_method, dataset_id, **kwargs):
    config = CONFIG_MODELS['NNWAvgPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    transform = transforms.Compose([ToTensor()])
    dataset = TACDatasetClassification(embedding_method, dataset_id, train, **kwargs)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x: x)

    model = NNWAvgPRModel(config).to(device=device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    loss = []

    for batch in data_loader:

        for sample in batch:
            (d, s1, s2), y = transform(sample)
            
            y_hat = model(d.to(device=device),
                          s1.to(device=device),
                          s2.to(device=device))
            
            L = criterion(y_hat, y.to(device=device))
            
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


def train_model_2_batch(embedding_method, dataset_id, **kwargs):
    def evaluate(model, dataset, val):
        dataset_val = TACDatasetLoadedClassification(dataset, val)
        data_loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE_VAL, shuffle=True)

        auc = 0.0

        for i, batch in enumerate(data_loader_val):
            d, s1, s2, m1, m2, y = batch

            y_hat = model(d.to(device=device),
                          s1.to(device=device),
                          s2.to(device=device),
                          m1.to(device=device),
                          m2.to(device=device))

            y_hat = y_hat.squeeze()

            y_hat = y_hat > 0.5

            auc += torch.sum(y_hat == y.to(device=device)).type(torch.float)
        
        return auc.cpu().numpy() / len(dataset_val)

    config = CONFIG_MODELS['NNWAvgPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    # transform = transforms.Compose([ToTensor(), Expand(20)])
    # dataset_train = TACDatasetClassification(embedding_method, dataset_id, train, transform=transform, **kwargs)

    dataset = defaultdict(defaultdict)
    for topic_id in TOPIC_IDS[dataset_id]:
        topic = load_embedded_topic(embedding_method, dataset_id, topic_id, **kwargs)
        document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
        document_embs = torch.tensor(document_embs, dtype=torch.float)
        summary_embs = torch.tensor(summary_embs, dtype=torch.float)
        dataset[topic_id]['document_embs'] = repeat_mean(document_embs, 150)
        for i, idx in enumerate(indices):
            summary_embs_, mask_ = pad(summary_embs[idx[0]:idx[1]], 150)
            dataset[topic_id]['summary_{}_embs'.format(summary_ids[i])] = summary_embs_
            dataset[topic_id]['mask_{}'.format(summary_ids[i])] = mask_
    
    dataset_train = TACDatasetLoadedClassification(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = NNWAvgPRBatchModel(config).to(device=device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            d, s1, s2, m1, m2, y = batch

            y_hat = model(d.to(device=device),
                          s1.to(device=device),
                          s2.to(device=device),
                          m1.to(device=device),
                          m2.to(device=device))

            y_hat = y_hat.squeeze()

            L = criterion(y_hat, y.to(device=device))

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())

            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')
        
        with torch.no_grad():
            print(f'AUC: {evaluate(model, dataset, val):.4f}')

    save_model(embedding_method, dataset_id, 'nn_wavg_pr_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_3(embedding_method, dataset_id, **kwargs):
    config = CONFIG_MODELS['LinSinkhornRegModel']

    data = load_train_data(dataset_id, 'regression')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    transform = transforms.Compose([Normalize(), ToTensor()])
    dataset = TACDatasetRegression(embedding_method, dataset_id, train, **kwargs)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x: x)

    model = LinSinkhornRegModel(config).to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    loss = []

    optimizer.zero_grad()

    for batch in data_loader:
        
        for sample in batch:
            (d, s), y = transform(sample)

            y_hat = model(d.to(device=device),
                          s.to(device=device))
            
            L = criterion(y_hat, y.to(device=device))
            
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


def train_model_3_batch(embedding_method, dataset_id, **kwargs):
    pass


def train_model_4_batch(embedding_method, dataset_id, **kwargs):
    pass


def train_model_5_batch(embedding_method, dataset_id, **kwargs):
    pass


PROCEDURES = [
    train_model_1_batch,
    train_model_2_batch,
    train_model_3_batch,
    train_model_4_batch,
    train_model_5_batch
]
