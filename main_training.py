
import argparse

from scipy.stats import kendalltau
import src.training.stratified as stratified
import src.training.crossval as crossval 
assert(len(stratified.PROCEDURES) == len(crossval.PROCEDURES))

from src.util.helpers import *
from src.config import *


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Encodes topic items.')

    desc = ', '.join([ '{} - {}'.format(i, embedding_method) for i, embedding_method in enumerate(EMBEDDING_METHODS) ])

    parser.add_argument('-em',
                        dest='embedding_method',
                        help='Embedding method: {}'.format(desc),
                        type=int,
                        choices=range(len(EMBEDDING_METHODS)))
    
    desc = ', '.join([ '{} - {}'.format(i, dataset_id) for i, dataset_id in enumerate(DATASET_IDS) ])
    
    parser.add_argument('-did',
                        dest='dataset_id',
                        help='Dataset ID: {}'.format(desc),
                        type=int,
                        choices=range(len(DATASET_IDS)))
    
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
    
    args = parser.parse_args()

    if args.embedding_method == None:
        raise Exception('Not suitable embedding method chosen. Use -h for more info.')
    
    if args.dataset_id == None:
        raise Exception('Not suitable dataset ID chosen. Use -h for more info.')

    layer = None

    if args.embedding_method in (3, 5):
        if args.layer == None:
            raise Exception('Not suitable layer chosen. Use -h for more info.')
        
        layer = args.layer
    
    if args.procedure == None:
        raise Exception('Not suitable procedure chosen. Use -h for more info.')
    
    embedding_method = EMBEDDING_METHODS[args.embedding_method]
    dataset_id = DATASET_IDS[args.dataset_id]

    procedure = stratified.PROCEDURES[args.procedure - 1]
    if args.crossval:
        print('Cross-validating!')
        procedure = crossval.PROCEDURES[args.procedure - 1]

    print(embedding_method, dataset_id, procedure.__name__)

    start = time.time()
    procedure(embedding_method, dataset_id, layer)
    end = time.time()
    
    print('Elapsed: {:}\n'.format(format_time(end - start)))
