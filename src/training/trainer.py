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
from src.config_models import make_config_models
from src.config_transformers import make_config_transformers


SHOULD_EVAL = False


class ModelTrainer():

    def accuracy(self, forward, dataset_val, batch_size_val=BATCH_SIZE_VAL):
        data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=True)

        auc = 0.0

        for i, batch in enumerate(data_loader_val):
            *_, y = batch

            y_hat = forward(batch)
            
            y_hat = (y_hat > 0.5).type(torch.bool)

            auc += torch.sum(y_hat == y.type(torch.bool).to(self.device)).type(torch.float)
        
        return auc.cpu().numpy() / len(dataset_val)
    
    def rmse(self, forward, dataset_val, batch_size_val=BATCH_SIZE_VAL):
        data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=True)

        mse = 0.0

        for i, batch in enumerate(data_loader_val):
            *_, y = batch

            y_hat = forward(batch)
            
            mse += torch.sum(torch.pow(y_hat - y.to(self.device), 2))
        
        return np.sqrt(mse.cpu().numpy() / len(dataset_val))

    def load_dataset(self, transformer):
        max_document_len = 0
        max_summary_len = 0
        dataset = defaultdict(defaultdict)
        for topic_id in TOPIC_IDS[self.dataset_id]:
            topic = load_embedded_topic(self.embedding_method, self.dataset_id, self.layer, topic_id)
            document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
            max_document_len = max(max_document_len, len(document_embs))
            dataset[topic_id]['documents'] = transformer['transform_documents'](document_embs)
            dataset[topic_id]['summaries'] = defaultdict(defaultdict)
            for i, idx in enumerate(indices):
                max_summary_len = max(max_summary_len, len(summary_embs[idx[0]:idx[1]]))
                dataset[topic_id]['summaries'][summary_ids[i]] = \
                    transformer['transform_summary'](summary_embs[idx[0]:idx[1]])
            dataset[topic_id]['pyr_scores'] = pyr_scores
        print(' *** Stats *** ')
        print(f'Maximum document length: {max_document_len}')
        print(f'Maximum summary length:  {max_summary_len}')
        return dataset

    def __init__(self, embedding_method, dataset_id, layer, device_id=0):
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.layer = layer
        self.config_models = make_config_models(embedding_method, dataset_id)
        self.transformers = make_config_transformers(embedding_method, dataset_id, False)
        self.device = DEVICES[device_id]

    def train_nn_rouge_reg_model(self, train, val):
        config = self.config_models['NNRougeRegModel']

        dataset = self.load_dataset(self.transformers['NNRougeRegModel'])

        dataset_train = TACDatasetRegressionRouge(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = NNRougeRegModel(config).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            e, y = batch

            return model(e.to(self.device))

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                e, y = batch

                y_hat = forward(batch)

                L = criterion(y_hat, -torch.log(y + 1e-8).to(self.device))

                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())

                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            if SHOULD_EVAL:
                with torch.no_grad():
                    dataset_val = TACDatasetRegressionRouge(dataset, val)
                    print(f'RMSE: {self.rmse(forward, dataset_val):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_nn_wavg_pr_model(self, train, val):
        config = self.config_models['NNWAvgPRModel']

        dataset = self.load_dataset(self.transformers['NNWAvgPRModel'])

        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = NNWAvgPRModel(config).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, m, mi, mj, y = batch

            return model(
                d.to(self.device),
                si.to(self.device),
                sj.to(self.device),
                mi.to(self.device),
                mj.to(self.device)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, m, mi, mj, y = batch

                y_hat = forward(batch)

                L = criterion(y_hat, y.to(self.device))

                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())

                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            if SHOULD_EVAL:
                with torch.no_grad():
                    dataset_val = TACDatasetClassification(dataset, val)
                    print(f'AUC: {self.accuracy(forward, dataset_val):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_lin_sinkhorn_reg_model(self, train, val):
        config = self.config_models['LinSinkhornRegModel']
        
        dataset = self.load_dataset(self.transformers['LinSinkhornRegModel'])
        
        dataset_train = TACDatasetRegression(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = LinSinkhornRegModel(config).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, h, hi, y = batch

            return model(
                d.to(self.device),
                si.to(self.device),
                h.to(self.device),
                hi.to(self.device),
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, h, hi, y = batch
                
                y_hat = forward(batch)
                
                L = criterion(y_hat, y.to(self.device))
                
                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())
                
                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            if SHOULD_EVAL:
                with torch.no_grad():
                    dataset_val = TACDatasetRegression(dataset, val)
                    print(f'RMSE: {self.rmse(forward, dataset_val):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_lin_sinkhorn_pr_model(self, train, val):
        config = self.config_models['LinSinkhornPRModel']

        dataset = self.load_dataset(self.transformers['LinSinkhornPRModel'])
        
        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = LinSinkhornPRModel(config).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, h, hi, hj, y = batch

            return model(
                d.to(self.device),
                si.to(self.device),
                sj.to(self.device),
                h.to(self.device),
                hi.to(self.device),
                hj.to(self.device)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, h, hi, hj, y = batch
                
                y_hat = forward(batch)
                
                L = criterion(y_hat, y.to(self.device))
                
                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())
                
                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            if SHOULD_EVAL:
                with torch.no_grad():
                    dataset_val = TACDatasetClassification(dataset, val)
                    print(f'AUC: {self.accuracy(forward, dataset_val, 128):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_nn_sinkhorn_pr_model(self, train, val):
        config = self.config_models['NNSinkhornPRModel']

        dataset = self.load_dataset(self.transformers['NNSinkhornPRModel'])
        
        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = NNSinkhornPRModel(config).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, h, hi, hj, y = batch

            return model(
                d.to(self.device),
                si.to(self.device),
                sj.to(self.device),
                h.to(self.device),
                hi.to(self.device),
                hj.to(self.device)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, h, hi, hj, y = batch
                
                y_hat = forward(batch)
                
                L = criterion(y_hat, y.to(self.device))
                
                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())
                
                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')

            if SHOULD_EVAL:
                with torch.no_grad():
                    dataset_val = TACDatasetClassification(dataset, val)
                    print(f'AUC: {self.accuracy(forward, dataset_val, 128):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model

    def train_cond_lin_sinkhorn_pr_model(self, train, val):
        config = self.config_models['CondLinSinkhornPRModel']

        dataset = self.load_dataset(self.transformers['CondLinSinkhornPRModel'])
        
        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = CondLinSinkhornPRModel(config).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, h, hi, hj, y = batch

            return model(
                d.to(self.device),
                si.to(self.device),
                sj.to(self.device),
                h.to(self.device),
                hi.to(self.device),
                hj.to(self.device)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, h, hi, hj, y = batch
                
                y_hat = forward(batch)
                
                L = criterion(y_hat, y.to(self.device))
                
                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())
                
                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            if SHOULD_EVAL:
                with torch.no_grad():
                    dataset_val = TACDatasetClassification(dataset, val)
                    print(f'AUC: {self.accuracy(forward, dataset_val, 128):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model
    
    def train_cond_nn_wavg_pr_model(self, train, val):
        config = self.config_models['CondNNWAvgPRModel']

        dataset = self.load_dataset(self.transformers['CondNNWAvgPRModel'])

        dataset_train = TACDatasetClassification(dataset, train)
        data_loader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        model = CondNNWAvgPRModel(config).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

        def forward(batch):
            d, si, sj, m, mi, mj, y = batch

            return model(
                d.to(self.device),
                si.to(self.device),
                sj.to(self.device),
                mi.to(self.device),
                mj.to(self.device)
            )

        loss = []

        for epoch in range(config['epochs']):
            print(f'Epoch {epoch + 1}')

            for i, batch in enumerate(data_loader_train):
                d, si, sj, m, mi, mj, y = batch

                y_hat = forward(batch)

                L = criterion(y_hat, y.to(self.device))

                L.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(L.item())

                print(f'{100 * float(i + 1) / len(data_loader_train):>6.2f}% complete - Train Loss: {loss[-1]:.4f}')
            
            if SHOULD_EVAL:
                with torch.no_grad():
                    dataset_val = TACDatasetClassification(dataset, val)
                    print(f'AUC: {self.accuracy(forward, dataset_val):.4f}')

        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(1,1,1)
        # plot_loss(ax, loss)
        # plt.show()

        return model