import argparse
from src.experiments.transform.executor import TransformExperimentExecutor
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS,
    DEVICES
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Executes transform experiments.')

    parser.add_argument('-l',
                        dest='layer',
                        help='Transformer\'s hidden state',
                        type=int,
                        choices=range(1, 13))
    
    desc = ', '.join([ '{} - {}'.format(i + 1, device) for i, device in enumerate(DEVICES) ])

    parser.add_argument('-d',
                        dest='device_id',
                        help='Device ID: {}'.format(desc),
                        type=int,
                        choices=(range(1, len(DEVICES) + 1)))
    
    parser.add_argument('-cv',
                        dest='crossval',
                        help='Cross-validate',
                        type=bool,
                        nargs='?',
                        const=True)
    
    args = parser.parse_args()

    if args.layer == None:
        raise Exception('Not suitable layer chosen. Use -h for more info.')

    if args.device_id == None:
        raise Exception('Not suitable device chosen. Use -h for more info.')

    layer = args.layer
    device_id = args.device_id - 1
    cv = args.crossval

    models = [
        TransformExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[0], layer, cv, device_id),
        TransformExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[1], layer, cv, device_id),
        TransformExperimentExecutor(EMBEDDING_METHODS[3], DATASET_IDS[2], layer, cv, device_id),
        TransformExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[0],  None, cv, device_id),
        TransformExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[1],  None, cv, device_id),
        TransformExperimentExecutor(EMBEDDING_METHODS[4], DATASET_IDS[2],  None, cv, device_id),
        TransformExperimentExecutor(EMBEDDING_METHODS[5], DATASET_IDS[0], layer, cv, device_id),
        TransformExperimentExecutor(EMBEDDING_METHODS[5], DATASET_IDS[1], layer, cv, device_id),
        TransformExperimentExecutor(EMBEDDING_METHODS[5], DATASET_IDS[2], layer, cv, device_id),
    ]

    for model in models:
        model.execute()
