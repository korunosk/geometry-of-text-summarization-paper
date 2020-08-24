import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
from operator import itemgetter
from collections import defaultdict
from scipy.stats import kendalltau
from src.implementation.transform.datasets import *
from src.implementation.transform.models import *
from src.util.helpers import (
    extract_topic_data,
    format_time
)
from src.util.loaders import (
    load_embedded_topic,
    load_model
)
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS,
    TOPIC_IDS,
    DEVICES,
    PLOTS_DIR
)
from src.config_models import CONFIG_MODELS
from src.config_transformers import TRANSFORMERS


DEVICE = DEVICES[0]


TRANSFORMERS['NNRougeRegModel'] = {
    'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, 650)))),
    'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, 15))))
}


class TransformExperimentExecutor():

    def __init__(self, embedding_method, dataset_id, layer):
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.layer = layer

        self.models = [
            load_model(self.embedding_method, self.dataset_id, self.layer, 'nn_rouge_reg_model', NNRougeRegModel, CONFIG_MODELS['NNRougeRegModel']).to(DEVICE),
            load_model(self.embedding_method, self.dataset_id, self.layer, 'nn_wavg_pr_model', NNWAvgPRModel, CONFIG_MODELS['NNWAvgPRModel']).to(DEVICE),
            load_model(self.embedding_method, self.dataset_id, self.layer, 'lin_sinkhorn_reg_model', LinSinkhornRegModel, CONFIG_MODELS['LinSinkhornRegModel']).to(DEVICE),
            load_model(self.embedding_method, self.dataset_id, self.layer, 'lin_sinkhorn_pr_model', LinSinkhornPRModel, CONFIG_MODELS['LinSinkhornPRModel']).to(DEVICE),
            load_model(self.embedding_method, self.dataset_id, self.layer, 'nn_sinkhorn_pr_model', NNSinkhornPRModel, CONFIG_MODELS['NNSinkhornPRModel']).to(DEVICE),
            load_model(self.embedding_method, self.dataset_id, self.layer, 'cond_lin_sinkhorn_pr_model', CondLinSinkhornPRModel, CONFIG_MODELS['CondLinSinkhornPRModel']).to(DEVICE)
        ]

        self.experiments = [{
                'label': 'NNRougeRegModel',
                'transformer': TRANSFORMERS['NNRougeRegModel'], 
                'procedure': lambda dataset: self.experiment(self.models[0], dataset)
            }, {
                'label': 'NNWAvgPRModel',
                'transformer': TRANSFORMERS['NNWAvgPRModel'], 
                'procedure': lambda dataset: self.experiment(self.models[1], dataset)
            }, {
                'label': 'LinSinkhornRegModel',
                'transformer': TRANSFORMERS['LinSinkhornRegModel'], 
                'procedure': lambda dataset: self.experiment(self.models[2], dataset)
            }, {
                'label': 'LinSinkhornPRModel',
                'transformer': TRANSFORMERS['LinSinkhornPRModel'], 
                'procedure': lambda dataset: self.experiment(self.models[3], dataset)
            }, {
                'label': 'NNSinkhornPRModel',
                'transformer': TRANSFORMERS['NNSinkhornPRModel'], 
                'procedure': lambda dataset: self.experiment(self.models[4], dataset)
            }, {
                'label': 'CondLinSinkhornPRModel',
                'transformer': TRANSFORMERS['CondLinSinkhornPRModel'], 
                'procedure': lambda dataset: self.experiment(self.models[5], dataset)
            }]
    
    def load_and_extract(self, transformer):
        dataset = defaultdict(defaultdict)
        for topic_id in TOPIC_IDS[self.dataset_id]:
            topic = load_embedded_topic(self.embedding_method, self.dataset_id, self.layer, topic_id)
            document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
            dataset[topic_id]['documents'] = transformer['transform_documents'](document_embs)
            dataset[topic_id]['summaries'] = defaultdict(defaultdict)
            for i, idx in enumerate(indices):
                dataset[topic_id]['summaries'][summary_ids[i]] = \
                    transformer['transform_summary'](summary_embs[idx[0]:idx[1]])
            dataset[topic_id]['pyr_scores'] = pyr_scores
        return dataset
    
    def experiment(self, model, data):
        si = torch.stack(list(map(itemgetter('embs'), data['summaries'].values())))
        ai = torch.stack(list(map(itemgetter('aux'), data['summaries'].values())))
        d = data['documents']['embs'].unsqueeze(0).repeat(si.shape[0], 1, 1)
        a = data['documents']['aux'].repeat(ai.shape[0], 1)
        y = data['pyr_scores']
        y_hat = model.predict(d.to(DEVICE),
                              si.to(DEVICE),
                              a.to(DEVICE),
                              ai.to(DEVICE)).cpu().tolist()
        return kendalltau(y, y_hat)[0]

    def __execute_experiment(self, transformer, procedure):
        dataset = self.load_and_extract(transformer)
        scores = []
        with torch.no_grad():
            for topic_id in TOPIC_IDS[self.dataset_id]:
                topic = dataset[topic_id]
                scores.append(procedure(topic))
        return np.array(scores)

    def __generate_plots(self):
        topic_ids = TOPIC_IDS[self.dataset_id]
        x = np.arange(len(topic_ids))
        fig = plt.figure(figsize=(17.5,5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x,  self.experiments[0]['values'], '-o', c='tab:blue',   label=f'Model 1: {np.mean( self.experiments[0]["values"]):.2f}')
        ax.plot(x,  self.experiments[1]['values'], '-^', c='tab:orange', label=f'Model 2: {np.mean( self.experiments[1]["values"]):.2f}')
        ax.plot(x, -self.experiments[2]['values'], '-s', c='tab:green',  label=f'Model 3: {np.mean(-self.experiments[2]["values"]):.2f}')
        ax.plot(x, -self.experiments[3]['values'], '-+', c='tab:red',    label=f'Model 4: {np.mean(-self.experiments[3]["values"]):.2f}')
        ax.plot(x, -self.experiments[4]['values'], '-x', c='tab:purple', label=f'Model 5: {np.mean(-self.experiments[4]["values"]):.2f}')
        ax.plot(x, -self.experiments[5]['values'], '-D', c='tab:brown',  label=f'Model 6: {np.mean(-self.experiments[5]["values"]):.2f}')
        ax.hlines(0.5, x[0], x[-1], linestyle='dashed', color='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(topic_ids, rotation=45)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Kendall tau')
        ax.legend(loc='upper right')
        fig.savefig(os.path.join(PLOTS_DIR, f'transform_correlation_{self.dataset_id}_{self.embedding_method}.png'), dpi=fig.dpi, bbox_inches='tight')

    def execute(self):
        result = ''
        print(f'=== Experiment "{self.dataset_id}" - Embeddings "{self.embedding_method}" ===\n')

        for i, experiment in enumerate(self.experiments):
            label = experiment['label']
            transformer = experiment['transformer']
            procedure = experiment['procedure']
            
            print('Executing "{}"\n'.format(label))
            
            start = time.time()
            values = self.__execute_experiment(transformer, procedure)
            end = time.time()
            
            print('   *** Elapsed: {:}\n'.format(format_time(end - start)))
            
            result += '{:30} {:.4}\n'.format(label, np.mean(values))
            
            self.experiments[i]['values'] = values
        
        print('\n=== Results ===\n')
        print(result)

        self.__generate_plots()
