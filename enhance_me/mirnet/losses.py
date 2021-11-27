import tensorflow as tf
from tensorflow.keras import losses


class CharbonnierLoss(losses.Loss):
    def __init__(self, epsilon: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.sqrt(tf.square(y_true - y_pred) + tf.square(self.epsilon))
        )
