import ray
from src.experiments.baseline.executor import BaselineMetricsExperimentExecutor
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS
)


if __name__ == '__main__':

    device_id = 0

    models = [
        BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[0], DATASET_IDS[0], None, None, device_id),
        BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[0], DATASET_IDS[1], None, None, device_id),
        BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[0], DATASET_IDS[2], None, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[1], DATASET_IDS[0], None, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[1], DATASET_IDS[1], None, None, device_id),
        BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[1], DATASET_IDS[2], None, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[2], DATASET_IDS[0], None, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[2], DATASET_IDS[1], None, None, device_id),
        BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[2], DATASET_IDS[2], None, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[0], 11, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[1], 11, None, device_id),
        BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[2], 11, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[0], None, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[1], None, None, device_id),
        BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[2], None, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[5], DATASET_IDS[0], 11, None, device_id),
        #BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[5], DATASET_IDS[1], 11, None, device_id),
        BaselineMetricsExperimentExecutor(EMBEDDING_METHODS[5], DATASET_IDS[2], 11, None, device_id),
    ]

    ray.init(num_cpus=40)
    for model in models:
        model.execute()
    ray.shutdown()
