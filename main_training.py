
import argparse

from scipy.stats import kendalltau
import src.training.stratified.procedures as stratified

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
    
    desc = ', '.join([ '{} - {}'.format(i, procedure.__name__) for i, procedure in enumerate(stratified.PROCEDURES) ])

    parser.add_argument('-p',
                        dest='procedure',
                        help='Train procedure: {}'.format(desc),
                        type=int,
                        choices=range(len(stratified.PROCEDURES)))
    
    args = parser.parse_args()

    kwargs = {}
    
    if args.embedding_method == None:
        raise Exception('Not suitable embedding method chosen. Use -h for more info.')
    
    if args.dataset_id == None:
        raise Exception('Not suitable dataset ID chosen. Use -h for more info.')

    if args.embedding_method in (3, 5):
        if args.layer == None:
            raise Exception('Not suitable layer chosen. Use -h for more info.')
        
        kwargs['layer'] = args.layer
    
    if args.procedure == None:
        raise Exception('Not suitable procedure chosen. Use -h for more info.')
    
    embedding_method = EMBEDDING_METHODS[args.embedding_method]
    dataset_id = DATASET_IDS[args.dataset_id]

    stratified.PROCEDURES[args.procedure](embedding_method, dataset_id, **kwargs)
