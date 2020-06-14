CONFIG_MODELS = {
    'NNRougeRegModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'learning_rate': 1e-2,
        'batch_size': 100
    },
    'NNWAvgPRModel': {
        'D_in': 768,
        'D_out': 768,
        'H': 1536,
        'scaling_factor': 1,
        'learning_rate': 1e-4,
        'batch_size': 100
    },
    'LinSinkhornRegModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'learning_rate': 1e-2,
        'batch_size': 100
    },
    'LinSinkhornPRModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'scaling_factor': 1,
        'learning_rate': 1e-2,
        'batch_size': 100
    },
    'NNSinkhornPRModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'scaling_factor': 1,
        'learning_rate': 1e-2,
        'batch_size': 100
    }
}
