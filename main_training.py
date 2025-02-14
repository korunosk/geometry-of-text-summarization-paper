
import time
import argparse
import src.training.crossval as crossval
import src.training.stratified as stratified
from scipy.stats import kendalltau
from src.util.helpers import format_time
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS,
    DEVICES
)

assert(len(stratified.PROCEDURES) == len(crossval.PROCEDURES))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Encodes topic items.')

    desc = ', '.join([ '{} - {}'.format(i + 1, embedding_method) for i, embedding_method in enumerate(EMBEDDING_METHODS) ])

    parser.add_argument('-em',
                        dest='embedding_method',
                        help='Embedding method: {}'.format(desc),
                        type=int,
                        choices=range(1, len(EMBEDDING_METHODS) + 1))
    
    desc = ', '.join([ '{} - {}'.format(i + 1, dataset_id) for i, dataset_id in enumerate(DATASET_IDS) ])
    
    parser.add_argument('-did',
                        dest='dataset_id',
                        help='Dataset ID: {}'.format(desc),
                        type=int,
                        choices=range(1, len(DATASET_IDS) + 1))
    
    parser.add_argument('-l',
                        dest='layer',
                        help='Transformer\'s hidden state',
                        type=int,
                        choices=range(1, 13))
    
    desc = ', '.join([ '{} - {}'.format(i + 1, procedure.__name__) for i, procedure in enumerate(stratified.PROCEDURES) ])

    parser.add_argument('-p',
                        dest='procedure',
                        help='Train procedure: {}'.format(desc),
                        type=int,
                        choices=range(1, len(stratified.PROCEDURES) + 1))
    
    parser.add_argument('-cv',
                        dest='crossval',
                        help='Cross-validate',
                        type=bool,
                        nargs='?',
                        const=True)
    
    desc = ', '.join([ '{} - {}'.format(i + 1, device) for i, device in enumerate(DEVICES) ])

    parser.add_argument('-d',
                        dest='device_id',
                        help='Device ID: {}'.format(desc),
                        type=int,
                        choices=(range(1, len(DEVICES) + 1)))
    
    args = parser.parse_args()

    if args.embedding_method == None:
        raise Exception('Not suitable embedding method chosen. Use -h for more info.')
    
    if args.dataset_id == None:
        raise Exception('Not suitable dataset ID chosen. Use -h for more info.')

    layer = None

    if args.embedding_method - 1 in (3, 5):
        if args.layer == None:
            raise Exception('Not suitable layer chosen. Use -h for more info.')
        
        layer = args.layer
    
    if args.procedure == None:
        raise Exception('Not suitable procedure chosen. Use -h for more info.')

    if args.device_id == None:
        raise Exception('Not suitable device chosen. Use -h for more info.')
    
    embedding_method = EMBEDDING_METHODS[args.embedding_method - 1]
    dataset_id = DATASET_IDS[args.dataset_id - 1]
    device_id = args.device_id - 1

    procedure = stratified.PROCEDURES[args.procedure - 1]
    if args.crossval:
        print('Cross-validating!')
        procedure = crossval.PROCEDURES[args.procedure - 1]

    print(embedding_method, dataset_id, procedure.__name__)

    start = time.time()
    procedure(embedding_method, dataset_id, layer, device_id)
    end = time.time()
    
    print('Elapsed: {:}\n'.format(format_time(end - start)))
