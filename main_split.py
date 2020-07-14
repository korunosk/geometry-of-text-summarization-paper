import argparse
from src.util.helpers import extract_topic_data
from src.util.loaders import (
    load_embedded_topic,
    save_embedded_item
)
from src.config import (
    EMBEDDING_METHODS,
    DATASET_IDS,
    TOPIC_IDS
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Exports embedded items.')

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

    for dataset_id in DATASET_IDS:
        print(dataset_id)
        
        for topic_id in TOPIC_IDS[dataset_id]:
            print('\t{}'.format(topic_id))
            
            topic = load_embedded_topic(embedding_method, dataset_id, layer, topic_id)
            document_embs, summary_embs, indices, pyr_scores, summary_ids = extract_topic_data(topic)

            item_id = 'document_embs'
            item = document_embs
            save_embedded_item(embedding_method, dataset_id, layer, topic_id, item_id, item)

            for i, idx in enumerate(indices):
                item_id = 'summary_{}_embs'.format(summary_ids[i])
                item = summary_embs[idx[0]:idx[1]]
                save_embedded_item(embedding_method, dataset_id, layer, topic_id, item_id, item)
