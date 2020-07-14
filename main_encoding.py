import argparse
from src.util.encoders import ENCODERS
from src.util.helpers import embed_topic
from src.util.loaders import (
    load_dataset,
    save_embedded_topic
)
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS,
    TOPIC_IDS
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Encodes topic items.')

    desc = ', '.join([ '{} - {}'.format(i, embedding_method) for i, embedding_method in enumerate(EMBEDDING_METHODS) ])

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

    if args.embedding_method == None:
        raise Exception('Not suitable embedding method chosen. Use -h for more info.')
    
    layer = None

    if args.embedding_method in (3, 5):
        if args.layer == None:
            raise Exception('Not suitable layer chosen. Use -h for more info.')
        
        layer = args.layer
    
    embedding_method = EMBEDDING_METHODS[args.embedding_method]

    print(embedding_method)

    encode = ENCODERS[args.embedding_method](layer)

    for dataset_id in DATASET_IDS:
        print(dataset_id)
        
        dataset = load_dataset(dataset_id)

        for topic_id in TOPIC_IDS[dataset_id]:
            print('\t{}'.format(topic_id))

            topic = dataset[topic_id]
            topic_embedded = embed_topic(topic, encode)
            save_embedded_topic(embedding_method, dataset_id, layer, topic_id, topic_embedded)
