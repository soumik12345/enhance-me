import tensorflow as tf
from typing import List

from ..commons import read_image
from ..augmentation import AugmentationFactory


class LowLightDataset:
    def __init__(self, image_size: int = 256) -> None:
        self.augmentation_factory = AugmentationFactory(image_size=image_size)

    def load_data(self, low_light_image_path, enhanced_image_path):
        low_light_image = read_image(low_light_image_path)
        enhanced_image = read_image(enhanced_image_path)
        low_light_image, enhanced_image = self.augmentation_factory.random_crop(
            low_light_image, enhanced_image
        )
        return low_light_image, enhanced_image

    def get_dataset(
        self,
        low_light_images: List[str],
        enhanced_images: List[str],
        batch_size: int = 16,
    ):
        dataset = tf.data.Dataset.from_tensor_slices(
            (low_light_images, enhanced_images)
        )
        dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset
