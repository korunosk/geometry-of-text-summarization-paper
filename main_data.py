import numpy as np
from src.util.helpers import (
    make_data_regression,
    make_data_regression_rouge,
    make_data_classification
)
from src.util.loaders import save_train_data
from src.config import DATASET_IDS


if __name__ == '__main__':

    for dataset_id in DATASET_IDS:
        print(dataset_id)

        data = np.array(make_data_regression(dataset_id), dtype=np.object)
        save_train_data(dataset_id, 'regression', data)

        data = np.array(make_data_regression_rouge(dataset_id), dtype=np.object)
        save_train_data(dataset_id, 'regression_rouge', data)

        data = np.array(make_data_classification(dataset_id), dtype=np.object)
        save_train_data(dataset_id, 'classification', data)
