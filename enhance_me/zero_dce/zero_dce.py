import os
import numpy as np
from PIL import Image
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, Model
from wandb.keras import WandbCallback

from .dce_net import build_dce_net
from .dataloader import UnpairedLowLightDataset
from .losses import (
    color_constancy_loss,
    exposure_loss,
    illumination_smoothness_loss,
    SpatialConsistencyLoss,
)
from ..commons import download_lol_dataset, init_wandb


class ZeroDCE(Model):
    def __init__(self, experiment_name=None, wandb_api_key=None, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.experiment_name = experiment_name
        if wandb_api_key is not None:
            init_wandb("zero-dce", experiment_name, wandb_api_key)
            self.using_wandb = True
        else:
            self.using_wandb = False
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )
        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))
        return losses

    def test_step(self, data):
        output = self.dce_model(data)
        return self.compute_losses(data, output)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.dce_model.save_weights(
            filepath, overwrite=overwrite, save_format=save_format, options=options
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )

    def build_datasets(
        self,
        image_size: int = 256,
        dataset_label: str = "lol",
        apply_resize: bool = False,
        apply_random_horizontal_flip: bool = True,
        apply_random_vertical_flip: bool = True,
        apply_random_rotation: bool = True,
        val_split: float = 0.2,
        batch_size: int = 16,
    ) -> None:
        if dataset_label == "lol":
            (self.low_images, _), (self.test_low_images, _) = download_lol_dataset()
        data_loader = UnpairedLowLightDataset(
            image_size,
            apply_resize,
            apply_random_horizontal_flip,
            apply_random_vertical_flip,
            apply_random_rotation,
        )
        self.train_dataset, self.val_dataset = data_loader.get_datasets(
            self.low_images, val_split, batch_size
        )

    def train(self, epochs: int):
        log_dir = os.path.join(
            self.experiment_name,
            "logs",
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        callbacks = [tensorboard_callback]
        if self.using_wandb:
            callbacks += [WandbCallback()]
        history = self.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks,
        )
        return history

    def infer(self, original_image):
        image = keras.preprocessing.image.img_to_array(original_image)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        output_image = self.call(image)
        output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
        output_image = Image.fromarray(output_image.numpy())
        return output_image

    def infer_from_file(self, original_image_file: str):
        original_image = Image.open(original_image_file)
        return self.infer(original_image)
