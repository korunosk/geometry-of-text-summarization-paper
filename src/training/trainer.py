import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
from torch.utils.data import DataLoader
from collections import defaultdict
from src.implementation.transform.datasets import *
from src.implementation.transform.models import *
from src.util.helpers import extract_topic_data
from src.util.loaders import (
    load_embedded_topic
)
from src.config import (
    TOPIC_IDS,
    BATCH_SIZE_VAL,
    DEVICES
)
from src.config_models import CONFIG_MODELS


DEVICE = DEVICES[0]


class ModelTrainer():

    def accuracy(self, forward, dataset_val, batch_size_val=BATCH_SIZE_VAL):
        data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=True)

        auc = 0.0

        for i, batch in enumerate(data_loader_val):
            *_, y = batch

            y_hat = forward(batch)
            
            y_hat = (y_hat > 0.5).type(torch.bool)

            auc += torch.sum(y_hat == y.type(torch.bool).to(DEVICE)).type(torch.float)
        
        return auc.cpu().numpy() / len(dataset_val)
    
    def rmse(self, forward, dataset_val, batch_size_val=BATCH_SIZE_VAL):
        data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=True)

        mse = 0.0

        for i, batch in enumerate(data_loader_val):
            *_, y = batch

            y_hat = forward(batch)
            
            mse += torch.sum(torch.pow(y_hat - y.to(DEVICE), 2))
        
        return np.sqrt(mse.cpu().numpy() / len(dataset_val))

    def load_dataset(self, transform_documents, transform_summary):
        dataset = defaultdict(defaultdict)
        for topic_id in TOPIC_IDS[self.dataset_id]:
            topic = load_embedded_topic(self.embedding_method, self.dataset_id, self.layer, topic_id)
            document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
            dataset[topic_id]['documents'] = transform_documents(document_embs)
            for i, idx in enumerate(indices):
                dataset[topic_id]['summary_{}'.format(summary_ids[i])] = \
                    transform_summary(summary_embs[idx[0]:idx[1]])
        return dataset

    def __init__(self, embedding_method, dataset_id, layer):
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.layer = layer

    def train_nn_rouge_reg_model(self, train, val):
        def transform_documents(document_embs):
            return {
                'embs': torch.tensor(document_embs, dtype=torch.float)
            }
        
        def transform_summary(summary_embs):
            return None
        
        config = CONFIG_MODELS['NNRougeRegModel']

        dataset = self.load_dataset(transform_documents, transform_summary)

        dataset_train = TACDatasetRegressionRouge(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = NNRougeRegModel(config).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            e, y = batch

            return model(e.to(DEVICE))

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                e, y = batch

                y_hat = forward(batch)

                L = criterion(y_hat, -torch.log(y + 1e-8).to(DEVICE))

                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())

                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            with torch.no_grad():
                dataset_val = TACDatasetRegressionRouge(dataset, val)
                print(f'RMSE: {self.rmse(forward, dataset_val):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_nn_wavg_pr_model(self, train, val):
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
        
        config = CONFIG_MODELS['NNWAvgPRModel']

        dataset = self.load_dataset(transform_documents, transform_summary)

        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = NNWAvgPRModel(config).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, m, mi, mj, y = batch

            return model(
                d.to(DEVICE),
                si.to(DEVICE),
                sj.to(DEVICE),
                mi.to(DEVICE),
                mj.to(DEVICE)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, m, mi, mj, y = batch

                y_hat = forward(batch)

                L = criterion(y_hat, y.to(DEVICE))

                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())

                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            with torch.no_grad():
                dataset_val = TACDatasetClassification(dataset, val)
                print(f'AUC: {self.accuracy(forward, dataset_val):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_lin_sinkhorn_reg_model(self, train, val):
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
        
        config = CONFIG_MODELS['LinSinkhornRegModel']

        dataset = self.load_dataset(transform_documents, transform_summary)
        
        dataset_train = TACDatasetRegression(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = LinSinkhornRegModel(config).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, h, hi, y = batch

            return model(
                d.to(DEVICE),
                si.to(DEVICE),
                h.to(DEVICE),
                hi.to(DEVICE),
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, h, hi, y = batch
                
                y_hat = forward(batch)
                
                L = criterion(y_hat, y.to(DEVICE))
                
                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())
                
                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            with torch.no_grad():
                dataset_val = TACDatasetRegression(dataset, val)
                print(f'RMSE: {self.rmse(forward, dataset_val):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_lin_sinkhorn_pr_model(self, train, val):
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

        config = CONFIG_MODELS['LinSinkhornPRModel']

        dataset = self.load_dataset(transform_documents, transform_summary)
        
        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = LinSinkhornPRModel(config).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, h, hi, hj, y = batch

            return model(
                d.to(DEVICE),
                si.to(DEVICE),
                sj.to(DEVICE),
                h.to(DEVICE),
                hi.to(DEVICE),
                hj.to(DEVICE)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, h, hi, hj, y = batch
                
                y_hat = forward(batch)
                
                L = criterion(y_hat, y.to(DEVICE))
                
                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())
                
                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            with torch.no_grad():
                dataset_val = TACDatasetClassification(dataset, val)
                print(f'AUC: {self.accuracy(forward, dataset_val, 256):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_nn_sinkhorn_pr_model(self, train, val):
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
        
        config = CONFIG_MODELS['NNSinkhornPRModel']

        dataset = self.load_dataset(transform_documents, transform_summary)
        
        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = NNSinkhornPRModel(config).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, h, hi, hj, y = batch

            return model(
                d.to(DEVICE),
                si.to(DEVICE),
                sj.to(DEVICE),
                h.to(DEVICE),
                hi.to(DEVICE),
                hj.to(DEVICE)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, h, hi, hj, y = batch
                
                y_hat = forward(batch)
                
                L = criterion(y_hat, y.to(DEVICE))
                
                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())
                
                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            with torch.no_grad():
                dataset_val = TACDatasetClassification(dataset, val)
                print(f'AUC: {self.accuracy(forward, dataset_val, 256):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_cond_lin_sinkhorn_pr_model(self, train, val):
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

        config = CONFIG_MODELS['CondLinSinkhornPRModel']

        dataset = self.load_dataset(transform_documents, transform_summary)
        
        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = CondLinSinkhornPRModel(config).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, h, hi, hj, y = batch

            return model(
                d.to(DEVICE),
                si.to(DEVICE),
                sj.to(DEVICE),
                h.to(DEVICE),
                hi.to(DEVICE),
                hj.to(DEVICE)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, h, hi, hj, y = batch
                
                y_hat = forward(batch)
                
                L = criterion(y_hat, y.to(DEVICE))
                
                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())
                
                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            with torch.no_grad():
                dataset_val = TACDatasetClassification(dataset, val)
                print(f'AUC: {self.accuracy(forward, dataset_val, 256):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model