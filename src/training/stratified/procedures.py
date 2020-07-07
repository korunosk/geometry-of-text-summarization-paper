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
    dataset_val = TACDatasetLoadedClassification(dataset, val)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=True)

    auc = 0.0

    for i, batch in enumerate(data_loader_val):
        *_, y = batch

        y_hat = forward(batch).squeeze()
        
        y_hat = (y_hat > 0.5).type(torch.bool)

        auc += torch.sum(y_hat == y.type(torch.bool).to(DEVICE1)).type(torch.float)
    
    return auc.cpu().numpy() / len(dataset_val)


def load_dataset(embedding_method, dataset_id, layer, transform_documents, transform_summaries):
    def transform_documents(document_embs):
        document_embs_, hist_ = pad_h(document_embs, 650)
        return {
            'embs': document_embs_,
            'aux': hist_
        }
    def transform_summaries(summary_embs):
        summary_embs_, hist_ = pad_h(summary_embs, 15)
        return {
            'embs': summary_embs_,
            'aux': hist_
        }
    dataset = defaultdict(defaultdict)
    for topic_id in TOPIC_IDS[dataset_id]:
        topic = load_embedded_topic(embedding_method, dataset_id, layer, topic_id)
        document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
        document_embs = torch.tensor(document_embs, dtype=torch.float)
        summary_embs = torch.tensor(summary_embs, dtype=torch.float)
        dataset[topic_id]['documents'] = transform_documents(document_embs)
        for i, idx in enumerate(indices):
            dataset[topic_id]['summary_{}'.format(summary_ids[i])] = \
                transform_summaries(summary_embs[idx[0]:idx[1]])
    return dataset
     


def train_model_1(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['NNRougeRegModel']

    data = load_train_data(dataset_id, 'regression_rouge')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    transform = transforms.Compose([ToTensor()])
    dataset = TACDatasetRegressionRouge(embedding_method, dataset_id, layer, train)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = NNRougeRegModel(config).to(DEVICE1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for batch in data_loader:
            (s, ), y = transform(batch)

            y_hat = model(s.to(DEVICE1))

            L = criterion(y_hat, -torch.log(y + 1e-8).to(DEVICE1))

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())

            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')

    save_model(embedding_method, dataset_id, 'nn_rouge_reg_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_2(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['NNWAvgPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    dataset = defaultdict(defaultdict)
    for topic_id in TOPIC_IDS[dataset_id]:
        topic = load_embedded_topic(embedding_method, dataset_id, layer, topic_id)
        document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
        document_embs = torch.tensor(document_embs, dtype=torch.float)
        summary_embs = torch.tensor(summary_embs, dtype=torch.float)
        dataset[topic_id]['document_embs'] = repeat_mean(document_embs, 15)
        dataset[topic_id]['aux'] = torch.tensor([])
        for i, idx in enumerate(indices):
            summary_embs_, mask_ = pad(summary_embs[idx[0]:idx[1]], 15)
            dataset[topic_id]['summary_{}_embs'.format(summary_ids[i])] = summary_embs_
            dataset[topic_id]['aux_{}'.format(summary_ids[i])] = mask_
    
    dataset_train = TACDatasetLoadedClassification(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = NNWAvgPRModel(config).to(DEVICE1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def forward(batch):
        d, s1, s2, m, m1, m2, _ = batch

        return model(
            d.to(DEVICE1),
            s1.to(DEVICE1),
            s2.to(DEVICE1),
            m1.to(DEVICE1),
            m2.to(DEVICE1)
        )

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            *_, y = batch

            y_hat = forward(batch).squeeze()

            L = criterion(y_hat, y.to(DEVICE1))

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())

            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')
        
        with torch.no_grad():
            print(f'AUC: {accuracy(forward, dataset, val):.4f}')

    save_model(embedding_method, dataset_id, 'nn_wavg_pr_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_3(embedding_method, dataset_id, layer):
    pass


def train_model_4(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['LinSinkhornPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    dataset = defaultdict(defaultdict)
    for topic_id in TOPIC_IDS[dataset_id]:
        topic = load_embedded_topic(embedding_method, dataset_id, topic_id)
        document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
        document_embs = torch.tensor(document_embs, dtype=torch.float)
        summary_embs = torch.tensor(summary_embs, dtype=torch.float)
        document_embs_, hist_ = pad_h(document_embs, 650)
        dataset[topic_id]['document_embs'] = document_embs_
        dataset[topic_id]['aux'] = hist_
        for i, idx in enumerate(indices):
            summary_embs_, hist_ = pad_h(summary_embs[idx[0]:idx[1]], 15)
            dataset[topic_id]['summary_{}_embs'.format(summary_ids[i])] = summary_embs_
            dataset[topic_id]['aux_{}'.format(summary_ids[i])] = hist_
    
    dataset_train = TACDatasetLoadedClassification(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = LinSinkhornPRModel(config).to(DEVICE1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def forward(batch):
        d, s1, s2, h, h1, h2, _ = batch

        return model(
            d.to(DEVICE1),
            s1.to(DEVICE1),
            s2.to(DEVICE1),
            h.to(DEVICE1),
            h1.to(DEVICE1),
            h2.to(DEVICE1)
        )

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            *_, y = batch
            
            y_hat = forward(batch).squeeze()
            
            L = criterion(y_hat, y.to(DEVICE1))
            
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())
            
            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')
        
        with torch.no_grad():
            print(f'AUC: {accuracy(forward, dataset, val, 256):.4f}')

    save_model(embedding_method, dataset_id, 'lin_sinkhorn_pr_model', model)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,1,1)
    # plot_loss(ax, loss)
    # plt.show()


def train_model_5(embedding_method, dataset_id, layer):
    config = CONFIG_MODELS['NNSinkhornPRModel']

    data = load_train_data(dataset_id, 'classification')
    train, val = stratified_sampling(data)
    print(len(train), len(val))

    dataset = defaultdict(defaultdict)
    for topic_id in TOPIC_IDS[dataset_id]:
        topic = load_embedded_topic(embedding_method, dataset_id, topic_id)
        document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
        document_embs = torch.tensor(document_embs, dtype=torch.float)
        summary_embs = torch.tensor(summary_embs, dtype=torch.float)
        document_embs_, hist_ = pad_h(document_embs, 650)
        dataset[topic_id]['document_embs'] = document_embs_
        dataset[topic_id]['aux'] = hist_
        for i, idx in enumerate(indices):
            summary_embs_, hist_ = pad_h(summary_embs[idx[0]:idx[1]], 15)
            dataset[topic_id]['summary_{}_embs'.format(summary_ids[i])] = summary_embs_
            dataset[topic_id]['aux_{}'.format(summary_ids[i])] = hist_
    
    dataset_train = TACDatasetLoadedClassification(dataset, train)
    data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

    model = NNSinkhornPRModel(config).to(DEVICE1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def forward(batch):
        d, s1, s2, h, h1, h2, _ = batch

        return model(
            d.to(DEVICE1),
            s1.to(DEVICE1),
            s2.to(DEVICE1),
            h.to(DEVICE1),
            h1.to(DEVICE1),
            h2.to(DEVICE1)
        )

    loss = []

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch + 1}')

        for i, batch in enumerate(data_loader_train):
            *_, y = batch
            
            y_hat = forward(batch).squeeze()
            
            L = criterion(y_hat, y.to(DEVICE1))
            
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss.append(L.item())
            
            if i % 10 == 0:
                print(f'\tTrain Loss: {loss[-1]:.4f}')
        
        with torch.no_grad():
            print(f'AUC: {accuracy(forward, dataset, val, 256):.4f}')

    save_model(embedding_method, dataset_id, 'nn_sinkhorn_pr_model', model)

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
