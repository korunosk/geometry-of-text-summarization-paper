from src.experiments.transform.executor import TransformExperimentExecutor
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS
)


if __name__ == '__main__':

    device_id = 0

    ex0 = TransformExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[2], 11, False, device_id)

    ex0.execute()