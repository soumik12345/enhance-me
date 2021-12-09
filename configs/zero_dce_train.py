import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.experiment_name = "unpaired_low_light_256_resize"  # Experiment Name
    config.image_size = 256  # Image Size
    config.dataset_label = "unpaired"  # Dataset Label
    config.use_mixed_precision = False  # Flag: Use Mixed-precision
    config.apply_resize = True  # Flag: Apply resize
    config.apply_random_horizontal_flip = True  # Flag: Apply Random Horizontal Flip
    config.apply_random_vertical_flip = True  # Flag: Apply Random Vertical Flip
    config.apply_random_rotation = True  # Flag: Apply Random Rotation
    config.val_split = 0.1  # Validation Split
    config.batch_size = 16  # Batch Size
    config.learning_rate = 1e-4  # Learning Rate
    config.epsilon = 1e-3  # Constant for Charbonnier Loss
    config.epochs = 100  # Number of training epochs
    return config
