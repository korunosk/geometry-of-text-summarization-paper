from src.util.helpers import *
from src.config import *


if __name__ == '__main__':

    for dataset_id in DATASET_IDS:
        print(dataset_id)

        data = make_data_regression(dataset_id)
        save_train_data(dataset_id, 'regression', data)

        data = make_data_regression_rouge(dataset_id)
        save_train_data(dataset_id, 'regression_rouge', data)

        data = make_data_classification(dataset_id)
        save_train_data(dataset_id, 'classification', data)
