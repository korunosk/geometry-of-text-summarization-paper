import rouge

from src.util.helpers import *
from src.config import *


if __name__ == '__main__':

    evaluator = rouge.Rouge(metrics=['rouge-n'],
                            max_n=2,
                            length_limit=100,
                            length_limit_type='words')

    for dataset_id in DATASET_IDS:
        print(dataset_id)
        
        scores = {}

        dataset = load_dataset(dataset_id)

        for topic_id in TOPIC_IDS[dataset_id]:
            print('\t{}'.format(topic_id))

            scores[topic_id] = []

            topic = dataset[topic_id]
            documents, summaries, indices, pyr_scores, summary_ids = extract_topic_data(topic)
            reference_summaries = summaries[indices[-4][0]:indices[-1][1]]

            for i in range(len(documents)):
                r = evaluator.get_scores(documents[i], reference_summaries)['rouge-2']['r']
                scores[topic_id].append(r)
        
        save_rouge_scores(dataset_id, scores)
