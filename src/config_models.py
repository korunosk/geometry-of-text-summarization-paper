CONFIG_MODELS = {
    'NNRougeRegModel': {
        'D': 768,
        'p': 2,
        'blur': 0.05,
        'scaling': 0.9,
        'learning_rate': 0.1,
        'batch_size': 128,
        'epochs': 5
    },
    'NNWAvgPRModel': {
        'D': 768,
        'H': 1536,
        'scaling_factor': 1,
        'learning_rate': 0.1,
        'batch_size': 128,
        'epochs': 5
    },
    'LinSinkhornRegModel': {
        'D': 768,
        'p': 2,
        'blur': 0.05,
        'scaling': 0.9,
        'learning_rate': 0.1,
        'batch_size': 128,
        'epochs': 5
    },
    'LinSinkhornPRModel': {
        'D': 768,
        'p': 2,
        'blur': 0.05,
        'scaling': 0.9,
        'scaling_factor': 1,
        'learning_rate': 0.1,
        'batch_size': 128,
        'epochs': 5
    },
    'NNSinkhornPRModel': {
        'D': 768,
        'p': 2,
        'blur': 0.05,
        'scaling': 0.9,
        'scaling_factor': 1,
        'learning_rate': 0.1,
        'batch_size': 128,
        'epochs': 5
    },
    'CondLinSinkhornPRModel': {
        'D': 768,
        'p': 2,
        'blur': 0.05,
        'scaling': 0.9,
        'scaling_factor': 1,
        'learning_rate': 0.1,
        'batch_size': 128,
        'epochs': 5
    },
}
