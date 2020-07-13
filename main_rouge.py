import rouge
from gensim.parsing.preprocessing import preprocess_string
from src.util.encoders import FILTERS
from src.util.helpers import (
    load_dataset,
    extract_topic_data
)
from src.util.loaders import save_rouge_scores
from src.config import (
    DATASET_IDS,
    TOPIC_IDS
)


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
                if not preprocess_string(documents[i]):
                    continue
                r = evaluator.get_scores(documents[i], reference_summaries)['rouge-2']['r']
                scores[topic_id].append(r)
        
        save_rouge_scores(dataset_id, scores)
