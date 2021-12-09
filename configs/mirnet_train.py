import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.experiment_name = "lol_dataset_256"  # Experiment Name
    config.image_size = 128  # Image Size
    config.dataset_label = "lol"  # Dataset Label
    config.apply_random_horizontal_flip = True  # Flag: Apply Random Horizontal Flip
    config.apply_random_vertical_flip = True  # Flag: Apply Random Vertical Flip
    config.apply_random_rotation = True  # Flag: Apply Random Rotation
    config.use_mixed_precision = True  # Flag: Use Mixed-precision
    config.val_split = 0.1  # Validation Split
    config.batch_size = 4  # Batch Size
    config.num_recursive_residual_groups = 3  # Number of recursive residual groups in MIRNet
    config.num_multi_scale_residual_blocks = 2  # Number of multi-scale residual blocks in MIRNet
    config.learning_rate = 1e-4  # learning rate
    config.epsilon = 1e-3  # Constant for Charbonnier Loss
    config.epochs = 50  # Number of training epochs
    return config
