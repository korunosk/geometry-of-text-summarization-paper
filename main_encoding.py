import argparse

from src.util.encoders import *
from src.util.helpers import *
from src.config import *


if __name__ == '__main__':

    desc = ', '.join([ '{} - {}'.format(i, embedding_method) for i, embedding_method in enumerate(EMBEDDING_METHODS) ])

    parser = argparse.ArgumentParser(description='Encodes topic items.')

    parser.add_argument('-em',
                        dest='embedding_method',
                        help='Embedding method: {}'.format(desc),
                        type=int,
                        choices=range(len(EMBEDDING_METHODS)))
    
    parser.add_argument('-l',
                        dest='layer',
                        help='Transformer\'s hidden state',
                        type=int,
                        choices=range(1, 13))
    
    args = parser.parse_args()

    kwargs = {}
    
    if args.embedding_method == None:
        raise Exception('Not suitable embedding method chosen. Use -h for more info.')
        exit()
    
    if args.embedding_method in (3, 5):
        if args.layer == None:
            raise Exception('Not suitable layer chosen. Use -h for more info.')
            exit()
        
        kwargs['layer'] = args.layer
    
    embedding_method = EMBEDDING_METHODS[args.embedding_method]

    print(embedding_method)

    encode = ENCODERS[args.embedding_method](**kwargs)

    for dataset_id in DATASET_IDS:
        print(dataset_id)
        
        dataset = load_dataset(dataset_id)

        for topic_id in TOPIC_IDS[dataset_id]:
            print('\t{}'.format(topic_id))

            topic = dataset[topic_id]
            topic_embedded = embedd_topic(topic, encode)
            save_embedded_topic(embedding_method, dataset_id, topic_id, topic_embedded, **kwargs)
