import tensorflow as tf
from tensorflow.keras import layers, Input, Model

from .recursive_residual_blocks import recursive_residual_group


def mirnet_model(num_rrg, num_mrb, channels):
    input_tensor = Input(shape=[None, None, 3])
    x1 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_rrg):
        x1 = recursive_residual_group(x1, num_mrb, channels)
    conv = layers.Conv2D(3, kernel_size=(3, 3), padding="same")(x1)
    output_tensor = layers.Add()([input_tensor, conv])
    return Model(input_tensor, output_tensor)
