import torch
from src.config import EMBEDDING_METHODS
from src.implementation.transform.datasets import (
    repeat_mean,
    pad,
    pad_h
)


def make_config_transformers(embedding_method, dataset_id, exec):
    PARAMETERS = {
        # LSA
        EMBEDDING_METHODS[0]: {
            'PADDING_DOCUMENTS': None,
            'PADDING_SUMMARY': None
        },
        # GloVe
        EMBEDDING_METHODS[1]: {
            'PADDING_DOCUMENTS': 1200 if dataset_id == 2 else 8850,
            'PADDING_SUMMARY': 110 if dataset_id == 2 else 150
        },
        # fasttext
        EMBEDDING_METHODS[2]: {
            'PADDING_DOCUMENTS': None,
            'PADDING_SUMMARY': None
        },
        # BERT_word
        EMBEDDING_METHODS[3]: {
            'PADDING_DOCUMENTS': 1200 if dataset_id == 2 else 8850,
            'PADDING_SUMMARY': 110 if dataset_id == 2 else 150
        },
        # BERT_sent
        EMBEDDING_METHODS[4]: {
            'PADDING_DOCUMENTS': 650,
            'PADDING_SUMMARY': 15
        },
        # BART_WORD
        EMBEDDING_METHODS[5]: {
            'PADDING_DOCUMENTS': 1200 if dataset_id == 2 else 8850,
            'PADDING_SUMMARY': 110 if dataset_id == 2 else 150
        },
    }

    TRANSFORMERS = {
        'NNRougeRegModel': {
            'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], [torch.tensor(document_embs, dtype=torch.float), None])),
            'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], [None, None]))
        },
        'NNWAvgPRModel': {
            'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, repeat_mean(document_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY'])))),
            'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad(summary_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY']))))
        },
        'LinSinkhornRegModel': {
            'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, PARAMETERS[embedding_method]['PADDING_DOCUMENTS'])))),
            'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY']))))
        },
        'LinSinkhornPRModel': {
            'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, PARAMETERS[embedding_method]['PADDING_DOCUMENTS'])))),
            'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY']))))
        },
        'NNSinkhornPRModel': {
            'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, PARAMETERS[embedding_method]['PADDING_DOCUMENTS'])))),
            'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY']))))
        },
        'CondLinSinkhornPRModel': {
            'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, PARAMETERS[embedding_method]['PADDING_DOCUMENTS'])))),
            'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY']))))
        },
        'CondNNWAvgPRModel': {
            'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, repeat_mean(document_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY'])))),
            'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad(summary_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY']))))
        },
    }

    # For NNRougeRegModel transformers, to get the
    # correlation with human judgements, we use again
    # the Sinkhorn loss.
    if exec:
        TRANSFORMERS['NNRougeRegModel'] = {
            'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, PARAMETERS[embedding_method]['PADDING_DOCUMENTS'])))),
            'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, PARAMETERS[embedding_method]['PADDING_SUMMARY']))))
        }
    
    return TRANSFORMERS
