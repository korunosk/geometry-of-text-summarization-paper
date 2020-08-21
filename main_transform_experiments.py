from src.experiments.transform.executor import TransformExperimentExecutor
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS
)


if __name__ == '__main__':

    ex0 = TransformExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[0], None)

    ex0.execute()