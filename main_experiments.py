import ray

from src.experiments.baseline.executor import BaselineMetricsExperimentExecutor
from src.config import *


if __name__ == '__main__':

    ex0 = BaselineMetricsExperimentExecutor(DATASET_IDS[0], EMBEDDING_METHODS[0])
    ex1 = BaselineMetricsExperimentExecutor(DATASET_IDS[1], EMBEDDING_METHODS[0])
    ex2 = BaselineMetricsExperimentExecutor(DATASET_IDS[0], EMBEDDING_METHODS[1])
    ex3 = BaselineMetricsExperimentExecutor(DATASET_IDS[1], EMBEDDING_METHODS[1])
    ex4 = BaselineMetricsExperimentExecutor(DATASET_IDS[0], EMBEDDING_METHODS[2])
    ex5 = BaselineMetricsExperimentExecutor(DATASET_IDS[1], EMBEDDING_METHODS[2])
    ex6 = BaselineMetricsExperimentExecutor(DATASET_IDS[0], EMBEDDING_METHODS[3])
    ex7 = BaselineMetricsExperimentExecutor(DATASET_IDS[1], EMBEDDING_METHODS[3])
    ex8 = BaselineMetricsExperimentExecutor(DATASET_IDS[0], EMBEDDING_METHODS[4])
    ex9 = BaselineMetricsExperimentExecutor(DATASET_IDS[1], EMBEDDING_METHODS[4])

    ray.init(num_cpus=30)
    ex6.execute()
    ray.shutdown()