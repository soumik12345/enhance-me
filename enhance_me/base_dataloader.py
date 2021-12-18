from abc import ABC
from typing import List
import tensorflow as tf

from .commons import read_image
from .augmentation import AugmentationFactory


class PairedDataLoader(ABC):
    def __init__(
        self,
        image_size: int = 256,
        apply_random_horizontal_flip: bool = True,
        apply_random_vertical_flip: bool = True,
        apply_random_rotation: bool = True,
    ) -> None:
        super().__init__()
        self.augmentation_factory = AugmentationFactory(image_size=image_size)
        self.apply_random_horizontal_flip = apply_random_horizontal_flip
        self.apply_random_vertical_flip = apply_random_vertical_flip
        self.apply_random_rotation = apply_random_rotation

    def load_data(self, input_image_path, enhanced_image_path):
        input_image = read_image(input_image_path)
        enhanced_image = read_image(enhanced_image_path)
        input_image, enhanced_image = self.augmentation_factory.random_crop(
            input_image, enhanced_image
        )
        return input_image, enhanced_image

    def configure_dataset(
        self,
        input_images: List[str],
        enhanced_images: List[str],
        batch_size: int = 16,
        is_train: bool = True,
    ):
        dataset = tf.data.Dataset.from_tensor_slices((input_images, enhanced_images))
        dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            self.augmentation_factory.random_crop, num_parallel_calls=tf.data.AUTOTUNE
        )
        if is_train:
            dataset = (
                dataset.map(
                    self.augmentation_factory.random_horizontal_flip,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                if self.apply_random_horizontal_flip
                else dataset
            )
            dataset = (
                dataset.map(
                    self.augmentation_factory.random_vertical_flip,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                if self.apply_random_vertical_flip
                else dataset
            )
            dataset = (
                dataset.map(
                    self.augmentation_factory.random_rotate,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                if self.apply_random_rotation
                else dataset
            )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    def get_datasets(
        self,
        input_images: List[str],
        enhanced_images: List[str],
        val_split: float = 0.2,
        batch_size: int = 16,
    ):
        assert len(input_images) == len(enhanced_images)
        split_index = int(len(input_images) * (1 - val_split))
        train_input_images = input_images[:split_index]
        train_enhanced_images = enhanced_images[:split_index]
        val_input_images = input_images[split_index:]
        val_enhanced_images = enhanced_images[split_index:]
        print(f"Number of train data points: {len(train_input_images)}")
        print(f"Number of validation data points: {len(val_input_images)}")
        train_dataset = self.configure_dataset(
            train_input_images, train_enhanced_images, batch_size, is_train=True
        )
        val_dataset = self.configure_dataset(
            val_input_images, val_enhanced_images, batch_size, is_train=False
        )
        return train_dataset, val_dataset
