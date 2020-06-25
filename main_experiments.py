import ray

from src.experiments.baseline.executor import BaselineMetricsExperimentExecutor
from src.config import *


if __name__ == '__main__':

    ex0 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[0], DATASET_IDS[0])
    ex1 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[0], DATASET_IDS[1])
    ex2 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[1], DATASET_IDS[0])
    ex3 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[1], DATASET_IDS[1])
    ex4 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[2], DATASET_IDS[0])
    ex5 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[2], DATASET_IDS[1])
    ex6 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[0])
    ex7 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[1])
    ex8 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[0])
    ex9 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[1])

    ray.init(num_cpus=30)
    ex6.execute()
    ray.shutdown()