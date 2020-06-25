import os
import string
from gensim.parsing.preprocessing import STOPWORDS

STOPWORDS = list(STOPWORDS)
PUNCTUATION = list(string.punctuation)

RANDOM_STATE = 42

EPOCHS = 10

DATASET_IDS = [
    'TAC2008',
    'TAC2009'
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
    ]
}

EMBEDDING_METHODS = [
    'LSA',
    'GloVe',
    'fasttext',
    'BERT_word',
    'BERT_sent'
]

BASE_DATA_DIR = '/scratch/korunosk/data_paper'

EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, 'embeddings')

DATA_DIR = os.path.join(BASE_DATA_DIR, 'data')

MODELS_DIR = os.path.join(BASE_DATA_DIR, 'models')

PLOTS_DIR = os.path.join(BASE_DATA_DIR, 'plots')
