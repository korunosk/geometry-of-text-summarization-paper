from src.config import EMBEDDING_METHODS


def make_config_models(embedding_method, dataset_id):
    PARAMETERS = {
        EMBEDDING_METHODS[0]: {
            'EMBEDDING_SIZE': 300,
            'BATCH_SIZE': 128,
            'EPOCHS': 5
        }, # LSA
        EMBEDDING_METHODS[1]: {
            'EMBEDDING_SIZE': 300,
            'BATCH_SIZE': 128,
            'EPOCHS': 5
        }, # GloVe
        EMBEDDING_METHODS[2]: {
            'EMBEDDING_SIZE': 300,
            'BATCH_SIZE': 128,
            'EPOCHS': 5
        }, # fasttext
        EMBEDDING_METHODS[3]: {
            'EMBEDDING_SIZE': 768,
            'BATCH_SIZE': 32 if dataset_id == 2 else 4,
            'EPOCHS': 5
        }, # BERT_word
        EMBEDDING_METHODS[4]: {
            'EMBEDDING_SIZE': 768,
            'BATCH_SIZE': 128,
            'EPOCHS': 5
        }, # BERT_sent
        EMBEDDING_METHODS[5]: {
            'EMBEDDING_SIZE': 1024,
            'BATCH_SIZE': 32 if dataset_id == 2 else 4,
            'EPOCHS': 5
        }, # BART_word
    }

    return {
        'NNRougeRegModel': {
            'D': PARAMETERS[embedding_method]['EMBEDDING_SIZE'],
            'p': 2,
            'blur': 0.05,
            'scaling': 0.9,
            'learning_rate': 0.01,
            'batch_size': PARAMETERS[embedding_method]['BATCH_SIZE'],
            'epochs': PARAMETERS[embedding_method]['EPOCHS']
        },
        'NNWAvgPRModel': {
            'D': PARAMETERS[embedding_method]['EMBEDDING_SIZE'],
            'H': 2 * PARAMETERS[embedding_method]['EMBEDDING_SIZE'],
            'scaling_factor': 1,
            'learning_rate': 0.01,
            'batch_size': PARAMETERS[embedding_method]['BATCH_SIZE'],
            'epochs': PARAMETERS[embedding_method]['EPOCHS']
        },
        'LinSinkhornRegModel': {
            'D': PARAMETERS[embedding_method]['EMBEDDING_SIZE'],
            'p': 2,
            'blur': 0.05,
            'scaling': 0.9,
            'learning_rate': 0.01,
            'batch_size': PARAMETERS[embedding_method]['BATCH_SIZE'],
            'epochs': PARAMETERS[embedding_method]['EPOCHS']
        },
        'LinSinkhornPRModel': {
            'D': PARAMETERS[embedding_method]['EMBEDDING_SIZE'],
            'p': 2,
            'blur': 0.05,
            'scaling': 0.9,
            'scaling_factor': 1,
            'learning_rate': 0.01,
            'batch_size': PARAMETERS[embedding_method]['BATCH_SIZE'],
            'epochs': PARAMETERS[embedding_method]['EPOCHS']
        },
        'NNSinkhornPRModel': {
            'D': PARAMETERS[embedding_method]['EMBEDDING_SIZE'],
            'p': 2,
            'blur': 0.05,
            'scaling': 0.9,
            'scaling_factor': 1,
            'learning_rate': 0.01,
            'batch_size': PARAMETERS[embedding_method]['BATCH_SIZE'],
            'epochs': PARAMETERS[embedding_method]['EPOCHS']
        },
        'CondLinSinkhornPRModel': {
            'D': PARAMETERS[embedding_method]['EMBEDDING_SIZE'],
            'p': 2,
            'blur': 0.05,
            'scaling': 0.9,
            'scaling_factor': 1,
            'learning_rate': 0.01,
            'batch_size': PARAMETERS[embedding_method]['BATCH_SIZE'],
            'epochs': PARAMETERS[embedding_method]['EPOCHS']
        },
    }
