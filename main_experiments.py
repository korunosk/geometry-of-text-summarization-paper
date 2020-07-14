import ray
from src.experiments.baseline.executor import BaselineMetricsExperimentExecutor
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS
)


if __name__ == '__main__':

    ex0  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[0], DATASET_IDS[0], None)
    ex1  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[0], DATASET_IDS[1], None)
    ex2  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[1], DATASET_IDS[0], None)
    ex3  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[1], DATASET_IDS[1], None)
    ex4  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[2], DATASET_IDS[0], None)
    ex5  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[2], DATASET_IDS[1], None)
    ex6  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[0], 11)
    ex7  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[1], 11)
    ex8  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[0], None)
    ex9  = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[1], None)
    ex10 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[5], DATASET_IDS[0], 11)
    ex11 = BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[5], DATASET_IDS[1], 11)

    ray.init(num_cpus=40)
    ex8.execute()
    ray.shutdown()
