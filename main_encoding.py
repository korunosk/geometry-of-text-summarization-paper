import argparse

from src.config import *
from src.encoders import *
from src.helpers import *


if __name__ == '__main__':

    desc = ', '.join([ '{} - {}'.format(i, embedding_method) for i, embedding_method in enumerate(EMBEDDING_METHODS) ])

    parser = argparse.ArgumentParser(description='Encodes topic items.')

    parser.add_argument('-em',
                        dest='embedding_method',
                        help='Embedding method: {}'.format(desc),
                        type=int,
                        choices=range(len(EMBEDDING_METHODS)))
    
    args = parser.parse_args()
    
    if args.embedding_method == None:
        raise Exception('Not suitable embedding method chosen. Use -h for more info.')
        exit()
    
    embedding_method = EMBEDDING_METHODS[args.embedding_method]

    print(embedding_method)

    encode = encoders[args.embedding_method]()

    for dataset_id in DATASET_IDS:
        print(dataset_id)
        
        dataset = load_dataset(dataset_id)

        for topic_id in TOPIC_IDS[dataset_id]:
            print('\t{}'.format(topic_id))

            topic = dataset[topic_id]
            topic_embedded = embedd_topic(topic, encode)
            save_embedded_topic(embedding_method, dataset_id, topic_id, topic_embedded)
