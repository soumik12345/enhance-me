import tensorflow as tf
from typing import List

from ..commons import read_image
from ..augmentation import UnpairedAugmentationFactory


class UnpairedLowLightDataset:
    def __init__(
        self,
        image_size: int = 256,
        apply_random_horizontal_flip: bool = True,
        apply_random_vertical_flip: bool = True,
        apply_random_rotation: bool = True,
    ) -> None:
        self.augmentation_factory = UnpairedAugmentationFactory(image_size=image_size)
        self.apply_random_horizontal_flip = apply_random_horizontal_flip
        self.apply_random_vertical_flip = apply_random_vertical_flip
        self.apply_random_rotation = apply_random_rotation

    def _get_dataset(self, images: List[str], batch_size: int, is_train: bool):
        dataset = tf.data.Dataset.from_tensor_slices((images))
        dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
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
        images: List[str],
        val_split: float = 0.2,
        batch_size: int = 16,
    ):
        split_index = int(len(images) * (1 - val_split))
        train_images = images[:split_index]
        val_images = images[split_index:]
        print(f"Number of train data points: {len(train_images)}")
        print(f"Number of validation data points: {len(val_images)}")
        train_dataset = self._get_dataset(train_images, batch_size, is_train=True)
        val_dataset = self._get_dataset(val_images, batch_size, is_train=False)
        return train_dataset, val_dataset
