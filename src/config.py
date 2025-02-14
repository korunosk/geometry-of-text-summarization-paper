import os
import string
import torch
from gensim.parsing.preprocessing import STOPWORDS

RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)

STOPWORDS = list(STOPWORDS)
PUNCTUATION = list(string.punctuation)

BATCH_SIZE_VAL = 1024

EMBEDDING_METHODS = [
    'LSA',
    'GloVe',
    'fasttext',
    'BERT_word',
    'BERT_sent',
    'BART_word'
]

DATASET_IDS = [
    'TAC2008',
    'TAC2009',
    'CNNDM'
]

TOPIC_IDS = {
    'TAC2008': [
        'D0841', 'D0804', 'D0802', 'D0809', 'D0819',
        'D0825', 'D0828', 'D0826', 'D0843', 'D0829',
        'D0813', 'D0807', 'D0812', 'D0820', 'D0835',
        'D0823', 'D0847', 'D0848', 'D0810', 'D0822',
        'D0845', 'D0844', 'D0839', 'D0814', 'D0824',
        'D0821', 'D0827', 'D0846', 'D0818', 'D0834',
        'D0805', 'D0817', 'D0831', 'D0815', 'D0836',
        'D0806', 'D0808', 'D0837', 'D0803', 'D0830',
        'D0838', 'D0840', 'D0842', 'D0832', 'D0816',
        'D0801', 'D0833', 'D0811'
    ],
    'TAC2009': [
        'D0919', 'D0904', 'D0934', 'D0928', 'D0944',
        'D0917', 'D0926', 'D0921', 'D0930', 'D0907',
        'D0929', 'D0913', 'D0920', 'D0909', 'D0922',
        'D0935', 'D0912', 'D0903', 'D0927', 'D0940',
        'D0902', 'D0925', 'D0910', 'D0931', 'D0943',
        'D0939', 'D0937', 'D0933', 'D0915', 'D0941',
        'D0911', 'D0924', 'D0908', 'D0932', 'D0914',
        'D0916', 'D0905', 'D0923', 'D0936', 'D0938',
        'D0942', 'D0918', 'D0901', 'D0906'
    ],
    'CNNDM': [
        'D01017', 'D10586', 'D11343', 'D01521', 'D02736',
        'D03789', 'D05025', 'D05272', 'D05576', 'D06564',
        'D07174', 'D07770', 'D08334', 'D09325', 'D09781',
        'D10231', 'D10595', 'D11351', 'D01573', 'D02748',
        'D03906', 'D05075', 'D05334', 'D05626', 'D06714',
        'D07397', 'D07823', 'D08565', 'D09393', 'D09825',
        'D10325', 'D10680', 'D11355', 'D01890', 'D00307',
        'D04043', 'D05099', 'D05357', 'D05635', 'D06731',
        'D07535', 'D07910', 'D08613', 'D09502', 'D10368',
        'D10721', 'D01153', 'D00019', 'D03152', 'D04303',
        'D05231', 'D05420', 'D05912', 'D06774', 'D07547',
        'D08001', 'D08815', 'D09555', 'D10537', 'D10824',
        'D01173', 'D01944', 'D03172', 'D04315', 'D05243',
        'D05476', 'D06048', 'D06784', 'D07584', 'D08054',
        'D08997', 'D09590', 'D10542', 'D11049', 'D01273',
        'D02065', 'D03583', 'D04637', 'D05244', 'D05524',
        'D06094', 'D06976', 'D07626', 'D08306', 'D09086',
        'D09605', 'D10563', 'D11264', 'D01492', 'D02292',
        'D03621', 'D04725', 'D05257', 'D05558', 'D06329',
        'D07058', 'D07670', 'D08312', 'D09221', 'D09709'
    ]
}

BASE_DATA_DIR   = '/scratch/korunosk/data_paper'

EMBEDDINGS_DIR  = os.path.join(BASE_DATA_DIR, 'embeddings')
DATA_DIR        = os.path.join(BASE_DATA_DIR, 'data')
MODELS_DIR      = os.path.join(BASE_DATA_DIR, 'models')
PLOTS_DIR       = os.path.join(BASE_DATA_DIR, 'plots')

DEVICES = [
    torch.device('cuda:0'),
    torch.device('cuda:1'),
    torch.device('cpu')
]
