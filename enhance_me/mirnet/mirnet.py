import os
import numpy as np
from PIL import Image
from typing import List
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import optimizers

from wandb.keras import WandbCallback

from .dataloader import LowLightDataset
from .models import build_mirnet_model
from .losses import CharbonnierLoss
from ..commons import (
    peak_signal_noise_ratio,
    closest_number,
    init_wandb,
    download_lol_dataset,
)


class MIRNet:
    def __init__(self, experiment_name: str, wandb_api_key=None) -> None:
        self.experiment_name = experiment_name
        if wandb_api_key is not None:
            init_wandb("mirnet", experiment_name, wandb_api_key)
            self.using_wandb = True
        else:
            self.using_wandb = False

    def build_datasets(
        self,
        image_size: int = 256,
        dataset_label: str = "lol",
        apply_random_horizontal_flip: bool = True,
        apply_random_vertical_flip: bool = True,
        apply_random_rotation: bool = True,
        val_split: float = 0.2,
        batch_size: int = 16,
    ):
        if dataset_label == "lol":
            (self.low_images, self.enhanced_images), (
                self.test_low_images,
                self.test_enhanced_images,
            ) = download_lol_dataset()
        self.data_loader = LowLightDataset(
            image_size=image_size,
            apply_random_horizontal_flip=apply_random_horizontal_flip,
            apply_random_vertical_flip=apply_random_vertical_flip,
            apply_random_rotation=apply_random_rotation,
        )
        (self.train_dataset, self.val_dataset) = self.data_loader.get_datasets(
            low_light_images=self.low_images,
            enhanced_images=self.enhanced_images,
            val_split=val_split,
            batch_size=batch_size,
        )

    def build_model(
        self,
        num_recursive_residual_groups: int = 3,
        num_multi_scale_residual_blocks: int = 2,
        channels: int = 64,
        learning_rate: float = 1e-4,
        epsilon: float = 1e-3,
    ):
        self.model = build_mirnet_model(
            num_rrg=num_recursive_residual_groups,
            num_mrb=num_multi_scale_residual_blocks,
            channels=channels,
        )
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=CharbonnierLoss(epsilon=epsilon),
            metrics=[peak_signal_noise_ratio],
        )

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.model.save_weights(
            filepath, overwrite=overwrite, save_format=save_format, options=options
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.model.load_weights(
            filepath, by_name=by_name, skip_mismatch=skip_mismatch, options=options
        )

    def train(self, epochs: int):
        log_dir = os.path.join(
            self.experiment_name,
            "logs",
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(self.experiment_name, "weights.h5"),
            save_best_only=True,
            save_weights_only=True,
        )
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor="val_peak_signal_noise_ratio",
            factor=0.5,
            patience=5,
            verbose=1,
            min_delta=1e-7,
            mode="max",
        )
        callbacks = [
            tensorboard_callback,
            model_checkpoint_callback,
            reduce_lr_callback,
        ]
        if self.using_wandb:
            callbacks += [WandbCallback()]
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks,
        )
        return history

    def infer(
        self,
        original_image,
        image_resize_factor: float = 1.0,
        resize_output: bool = False,
    ):
        width, height = original_image.size
        target_width, target_height = (
            closest_number(width // image_resize_factor, 4),
            closest_number(height // image_resize_factor, 4),
        )
        original_image = original_image.resize(
            (target_width, target_height), Image.ANTIALIAS
        )
        image = keras.preprocessing.image.img_to_array(original_image)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        output = self.model.predict(image)
        output_image = output[0] * 255.0
        output_image = output_image.clip(0, 255)
        output_image = output_image.reshape(
            (np.shape(output_image)[0], np.shape(output_image)[1], 3)
        )
        output_image = Image.fromarray(np.uint8(output_image))
        original_image = Image.fromarray(np.uint8(original_image))
        if resize_output:
            output_image = output_image.resize((width, height), Image.ANTIALIAS)
        return output_image

    def infer_from_file(
        self,
        original_image_file: str,
        image_resize_factor: float = 1.0,
        resize_output: bool = False,
    ):
        original_image = Image.open(original_image_file)
        return self.infer(original_image, image_resize_factor, resize_output)
