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


def accuracy(forward, dataset, val, batch_size_val=BATCH_SIZE_VAL):
    dataset_val = TACDatasetClassification(dataset, val)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=True)

    auc = 0.0

    for i, batch in enumerate(data_loader_val):
        *_, y = batch

        y_hat = forward(batch)
        
        y_hat = (y_hat > 0.5).type(torch.bool)

        auc += torch.sum(y_hat == y.type(torch.bool).to(DEVICE1)).type(torch.float)
    
    return auc.cpu().numpy() / len(dataset_val)


def load_dataset(embedding_method, dataset_id, layer, transform_documents, transform_summary):
    dataset = defaultdict(defaultdict)
    for topic_id in TOPIC_IDS[dataset_id]:
        topic = load_embedded_topic(embedding_method, dataset_id, layer, topic_id)
        document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
        dataset[topic_id]['documents'] = transform_documents(document_embs)
        for i, idx in enumerate(indices):
            dataset[topic_id]['summary_{}'.format(summary_ids[i])] = \
                transform_summary(summary_embs[idx[0]:idx[1]])
    return dataset
     

def train_model_1(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['NNRougeRegModel']

    data = load_train_data(dataset_id, 'regression_rouge')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    def transform_documents(document_embs):
        return {
            'embs': torch.tensor(document_embs, dtype=torch.float)
        }
    
    def transform_summary(summary_embs):
        return None

    dataset = load_dataset(embedding_method, dataset_id, layer, transform_documents, transform_summary)

    dataset_train = TACDatasetRegressionRouge(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = NNRougeRegModel(config).to(DEVICE1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            e, y = batch

            y_hat = model(e.to(DEVICE1))

            L = criterion(y_hat, -torch.log(y + 1e-8).to(DEVICE1))

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())

            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')

    save_model(embedding_method, dataset_id, layer, 'nn_rouge_reg_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_2(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['NNWAvgPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    def transform_documents(document_embs):
        document_embs = repeat_mean(document_embs, 15)
        return {
            'embs': torch.tensor(document_embs, dtype=torch.float),
            'aux': torch.tensor([], dtype=torch.bool)
        }
    
    def transform_summary(summary_embs):
        summary_embs, mask = pad(summary_embs, 15)
        return {
            'embs': torch.tensor(summary_embs, dtype=torch.float),
            'aux': torch.tensor(mask, dtype=torch.bool)
        }

    dataset = load_dataset(embedding_method, dataset_id, layer, transform_documents, transform_summary)

    dataset_train = TACDatasetClassification(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = NNWAvgPRModel(config).to(DEVICE1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def forward(batch):
        d, si, sj, m, mi, mj, y = batch

        return model(
            d.to(DEVICE1),
            si.to(DEVICE1),
            sj.to(DEVICE1),
            mi.to(DEVICE1),
            mj.to(DEVICE1)
        )

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            d, si, sj, m, mi, mj, y = batch

            y_hat = forward(batch)

            L = criterion(y_hat, y.to(DEVICE1))

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())

            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')
        
        with torch.no_grad():
            print(f'AUC: {accuracy(forward, dataset, val):.4f}')

    save_model(embedding_method, dataset_id, layer, 'nn_wavg_pr_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_3(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['LinSinkhornRegModel']

    data = load_train_data(dataset_id, 'regression')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    def transform_documents(document_embs):
        document_embs, hist = pad_h(document_embs, 650)
        return {
            'embs': torch.tensor(document_embs, dtype=torch.float),
            'aux': torch.tensor(hist, dtype=torch.float)
        }
    
    def transform_summary(summary_embs):
        summary_embs, hist = pad_h(summary_embs, 15)
        return {
            'embs': torch.tensor(summary_embs, dtype=torch.float),
            'aux': torch.tensor(hist, dtype=torch.float)
        }

    dataset = load_dataset(embedding_method, dataset_id, layer, transform_documents, transform_summary)
    
    dataset_train = TACDatasetRegression(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = LinSinkhornRegModel(config).to(DEVICE1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def forward(batch):
        d, si, h, hi, y = batch

        return model(
            d.to(DEVICE1),
            si.to(DEVICE1),
            h.to(DEVICE1),
            hi.to(DEVICE1),
        )

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            d, si, h, hi, y = batch
            
            y_hat = forward(batch)
            
            L = criterion(y_hat, y.to(DEVICE1))
            
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())
            
            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')

    save_model(embedding_method, dataset_id, layer, 'lin_sinkhorn_reg_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_4(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['LinSinkhornPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    def transform_documents(document_embs):
        document_embs, hist = pad_h(document_embs, 650)
        return {
            'embs': torch.tensor(document_embs, dtype=torch.float),
            'aux': torch.tensor(hist, dtype=torch.float)
        }
    
    def transform_summary(summary_embs):
        summary_embs, hist = pad_h(summary_embs, 15)
        return {
            'embs': torch.tensor(summary_embs, dtype=torch.float),
            'aux': torch.tensor(hist, dtype=torch.float)
        }

    dataset = load_dataset(embedding_method, dataset_id, layer, transform_documents, transform_summary)
    
    dataset_train = TACDatasetClassification(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = LinSinkhornPRModel(config).to(DEVICE1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def forward(batch):
        d, si, sj, h, hi, hj, y = batch

        return model(
            d.to(DEVICE1),
            si.to(DEVICE1),
            sj.to(DEVICE1),
            h.to(DEVICE1),
            hi.to(DEVICE1),
            hj.to(DEVICE1)
        )

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            d, si, sj, h, hi, hj, y = batch
            
            y_hat = forward(batch)
            
            L = criterion(y_hat, y.to(DEVICE1))
            
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())
            
            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')
        
        with torch.no_grad():
            print(f'AUC: {accuracy(forward, dataset, val, 256):.4f}')

    save_model(embedding_method, dataset_id, layer, 'lin_sinkhorn_pr_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_5(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['NNSinkhornPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    def transform_documents(document_embs):
        document_embs, hist = pad_h(document_embs, 650)
        return {
            'embs': torch.tensor(document_embs, dtype=torch.float),
            'aux': torch.tensor(hist, dtype=torch.float)
        }
    
    def transform_summary(summary_embs):
        summary_embs, hist = pad_h(summary_embs, 15)
        return {
            'embs': torch.tensor(summary_embs, dtype=torch.float),
            'aux': torch.tensor(hist, dtype=torch.float)
        }

    dataset = load_dataset(embedding_method, dataset_id, layer, transform_documents, transform_summary)
    
    dataset_train = TACDatasetClassification(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = NNSinkhornPRModel(config).to(DEVICE1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def forward(batch):
        d, si, sj, h, hi, hj, y = batch

        return model(
            d.to(DEVICE1),
            si.to(DEVICE1),
            sj.to(DEVICE1),
            h.to(DEVICE1),
            hi.to(DEVICE1),
            hj.to(DEVICE1)
        )

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            d, si, sj, h, hi, hj, y = batch
            
            y_hat = forward(batch)
            
            L = criterion(y_hat, y.to(DEVICE1))
            
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())
            
            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')
        
        with torch.no_grad():
            print(f'AUC: {accuracy(forward, dataset, val, 256):.4f}')

    save_model(embedding_method, dataset_id, layer, 'nn_sinkhorn_pr_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


PROCEDURES = [
    train_model_1,
    train_model_2,
    train_model_3,
    train_model_4,
    train_model_5
]
