CONFIG_MODELS = {
    'NNRougeRegModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'learning_rate': .01,
        'batch_size': 128,
        'epochs': 5
    },
    'NNWAvgPRModel': {
        'D_in': 768,
        'D_out': 768,
        'H': 1536,
        'scaling_factor': 1,
        'learning_rate': .01,
        'batch_size': 128,
        'epochs': 5
    },
    'LinSinkhornRegModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'learning_rate': .01,
        'batch_size': 128,
        'epochs': 5
    },
    'LinSinkhornPRModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'scaling_factor': 1,
        'learning_rate': .01,
        'batch_size': 128,
        'epochs': 5
    },
    'NNSinkhornPRModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'scaling_factor': 1,
        'learning_rate': .01,
        'batch_size': 128,
        'epochs': 5
    }
}
