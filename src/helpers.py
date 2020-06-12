from operator import itemgetter
from itertools import chain
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from .config import *
from .loaders import *


def extract_topic_data(topic):
    documents = list(chain(*topic['documents']))
    annotations = topic['annotations']
    
    summaries = annotations[0]['text']
    indices = [[0, len(summaries)]]
    pyr_scores = [annotations[0]['pyr_score']]
    summary_ids = [annotations[0]['summ_id']]
    
    for o in annotations[1:]:
        summaries.extend(o['text'])
        start = indices[-1][1]
        indices.append([start, start + len(o['text'])])
        pyr_scores.append(o['pyr_score'])
        summary_ids.append(o['summ_id'])

    return documents, summaries, indices, pyr_scores, summary_ids


def make_annotations(summary_ids, pyr_scores, embeddings):
    # Keep the format for easier parsing of different files
    return [ {
        'summ_id': summary_id,
        'pyr_score': pyr_score,
        'text': embedding
    } for summary_id, pyr_score, embedding in zip(summary_ids, pyr_scores, embeddings) ]


def embedd_topic(topic, encode):
    documents = topic['documents']
    annotations = topic['annotations']

    summaries = list(map(itemgetter('text'), annotations))
    pyr_scores = list(map(itemgetter('pyr_score'), annotations))
    summary_ids = list(map(itemgetter('summ_id'), annotations))

    return {
        'documents': encode(documents),
        'annotations': make_annotations(summary_ids, pyr_scores, encode(summaries))
    }


def read_sentences():
    sentences = []
    for dataset_id in DATASET_IDS:
        dataset = load_dataset(dataset_id)
        for topic_id in TOPIC_IDS[dataset_id]:
            topic = dataset[topic_id]
            documents, summaries, indices, pyr_scores, summary_ids = extract_topic_data(topic)
            sentences.extend(documents)
    return sentences


def make_vectorizer(sentences):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), stop_words=STOPWORDS)
    vectorizer.fit(sentences)
    return vectorizer
