import torch
from src.implementation.transform.datasets import *


TRANSFORMERS = {
    'NNRougeRegModel': {
        'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], [torch.tensor(document_embs, dtype=torch.float), None])),
        'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], [None, None]))
    },
    'NNWAvgPRModel': {
        'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, repeat_mean(document_embs, 15)))),
        'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad(summary_embs, 15))))
    },
    'LinSinkhornRegModel': {
        'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, 650)))),
        'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, 15))))
    },
    'LinSinkhornPRModel': {
        'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, 650)))),
        'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, 15))))
    },
    'NNSinkhornPRModel': {
        'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, 650)))),
        'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, 15))))
    },
    'CondLinSinkhornPRModel': {
        'transform_documents': lambda document_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(document_embs, 650)))),
        'transform_summary': lambda summary_embs: dict(zip(['embs', 'aux'], map(torch.from_numpy, pad_h(summary_embs, 15))))
    }
}
