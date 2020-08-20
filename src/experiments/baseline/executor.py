import os
import time
import numpy as np
import ray # Interferes with linalg.det causing semantic_spread to output different results
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from src.external.lexrank import degree_centrality_scores
from typing import Callable
from src.implementation.baseline.redundancy import *
from src.implementation.baseline.relevance import *
from src.implementation.transform.models import * 
from src.util.helpers import (
    extract_topic_data,
    format_time
)
from src.util.loaders import (
    load_model,
    load_embedded_topic
)
from src.util.visualization import (
    project_pca,
    # plot_corr_coeff
)
from src.config import (
    DATASET_IDS,
    TOPIC_IDS,
    PLOTS_DIR,
    DEVICES
)
from src.config_models import CONFIG_MODELS


# models = [
    # load_model(embedding_method, dataset_id, 'nn_rouge_reg_model', NNRougeRegModel, CONFIG_MODELS['NNRougeRegModel']).to(DEVICES[1]),
    # load_model(embedding_method, dataset_id, 'nn_wavg_pr_model', NNWAvgPRModel, CONFIG_MODELS['NNWAvgPRModel']).to(DEVICES[1]),
    # load_model(embedding_method, dataset_id, 'lin_sinkhorn_reg_model', LinSinkhornRegModel, CONFIG_MODELS['LinSinkhornRegModel']).to(DEVICES[1]),
    # load_model(embedding_method, dataset_id, 'lin_sinkhorn_pr_model', LinSinkhornPRModel, CONFIG_MODELS['LinSinkhornPRModel']).to(DEVICES[1]),
    # load_model(embedding_method, dataset_id, 'nn_sinkhorn_pr_model', NNSinkhornPRModel, CONFIG_MODELS['NNSinkhornPRModel']).to(DEVICES[1]),
# ]


# def transform(x):
#     return models[0].transform(
#         torch.tensor(x, dtype=torch.float).to(DEVICES[1])
#     ).data.cpu().tolist()

def transform(x):
    return x


class BaselineMetricsExperimentExecutor():
    
    def __init__(self, embedding_method: str, dataset_id: str, layer: int):
        self.embedding_method = embedding_method
        self.dataset_id       = dataset_id
        self.layer            = layer
        self.topic_ids        = TOPIC_IDS[dataset_id]
        
        # Define list of experiments to execute.
        # Every entry needs to contain a label - the experiment name,
        # and a procedure - the method that will be executed.
        self.experiments = [{
                'label': 'Average Pairwise Distance',
                'procedure': self.experiment_average_pairwise_distance 
            }, {
                'label': 'Semantic Volume',
                'procedure': self.experiment_semantic_volume 
            }, {
                'label': 'Semantic Spread',
                'procedure': self.experiment_semantic_spread 
            }, {
                'label': 'Word Mover Distance',
                'procedure': self.experiment_word_mover_distance 
            }, {
                'label': 'LexRank',
                'procedure': self.experiment_lex_rank 
            }]
    
    @staticmethod
    @ray.remote
    def load_and_extract(embedding_method: str, dataset_id: str, layer: int, topic_id: str) -> tuple:
        topic = load_embedded_topic(embedding_method, dataset_id, layer, topic_id)
        document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)
        document_embs = transform(document_embs)
        summary_embs = transform(summary_embs)
        return document_embs, summary_embs, indices, pyr_scores, summary_ids
    
    @staticmethod
    @ray.remote
    def experiment_average_pairwise_distance(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        summary_embs = np.array(summary_embs)
        metric = lambda i: average_pairwise_distance(summary_embs[i[0]:i[1]])
        return kendalltau(pyr_scores, [metric(i) for i in indices])[0]

    @staticmethod
    @ray.remote
    def experiment_semantic_volume(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        summary_embs = np.array(summary_embs)
        document_pts, summary_pts = project_pca(np.concatenate((document_embs, summary_embs)), document_embs.shape[0])
        metric = lambda i: semantic_volume(summary_pts[i[0]:i[1]])
        return kendalltau(pyr_scores, [metric(i) for i in indices])[0]

    @staticmethod
    @ray.remote
    def experiment_semantic_spread(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        summary_embs = np.array(summary_embs)
        metric = lambda i: semantic_spread(summary_embs[i[0]:i[1]])
        return kendalltau(pyr_scores, [metric(i) for i in indices])[0]

    @staticmethod
    @ray.remote
    def experiment_word_mover_distance(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        summary_embs = np.array(summary_embs)
        metric = lambda i: word_mover_distance(document_embs, summary_embs[i[0]:i[1]])
        return kendalltau(pyr_scores, [metric(i) for i in indices])[0]

    @staticmethod
    @ray.remote
    def experiment_lex_rank(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        summary_embs = np.array(summary_embs)
        lr_scores = degree_centrality_scores(cdist(document_embs, document_embs, metric='cosine'))
        metric = lambda i: lex_rank(document_embs, summary_embs[i[0]:i[1]], lr_scores)
        return kendalltau(pyr_scores, [metric(i) for i in indices])[0]
    
    def __execute_experiment(self, experiment: Callable) -> np.array:
        # Pass 1: Collect the topics
        dataset = [ self.load_and_extract.remote(self.embedding_method, self.dataset_id, self.layer, topic_id)
                       for topic_id in self.topic_ids ]
        # Pass 2: Execute the experiment
        scores  = [ experiment.remote(topic)
                       for topic in dataset ]

        return ray.get(scores)
    
    def __generate_plots(self):
        fig = plt.figure(figsize=(17.5,10))
        # Redundancy
        ax1 = fig.add_subplot(2,1,1)
        plot_corr_coeff(ax1, self.topic_ids, self.experiments[:3])
        ax1.set_xlabel('')
        # Relevance
        ax2 = fig.add_subplot(2,1,2)
        plot_corr_coeff(ax2, self.topic_ids, self.experiments[3:])
        fig.savefig(os.path.join(PLOTS_DIR, f'{self.dataset_id}_{self.embedding_method}.png'), dpi=fig.dpi, bbox_inches='tight')
          
    def execute(self):
        result = ''
        print(f'=== Experiment "{self.dataset_id}" - Embeddings "{self.embedding_method}" ===\n')

        for i, experiment in enumerate(self.experiments):
            label = experiment['label']
            procedure = experiment['procedure']
            
            print('Executing "{}"\n'.format(label))
            
            start = time.time()
            values = np.nan_to_num(self.__execute_experiment(procedure), nan=0)
            end = time.time()
            
            print('   *** Elapsed: {:}\n'.format(format_time(end - start)))
            
            result += '{:30} {:.4}\n'.format(label, np.mean(values))
            
            self.experiments[i]['values'] = values

        print('\n=== Results ===\n')
        print(result)
        
        # self.__generate_plots()
