import tensorflow as tf


class AugmentationFactory:
    def __init__(self, image_size) -> None:
        self.image_size = image_size

    def random_crop(self, input_image, enhanced_image):
        input_image_shape = tf.shape(input_image)[:2]
        low_w = tf.random.uniform(
            shape=(), maxval=input_image_shape[1] - self.image_size + 1, dtype=tf.int32
        )
        low_h = tf.random.uniform(
            shape=(), maxval=input_image_shape[0] - self.image_size + 1, dtype=tf.int32
        )
        enhanced_w = low_w
        enhanced_h = low_h
        input_image_cropped = input_image[
            low_h : low_h + self.image_size, low_w : low_w + self.image_size
        ]
        enhanced_image_cropped = enhanced_image[
            enhanced_h : enhanced_h + self.image_size,
            enhanced_w : enhanced_w + self.image_size,
        ]
        return input_image_cropped, enhanced_image_cropped

    def random_horizontal_flip(sefl, input_image, enhanced_image):
        return tf.cond(
            tf.random.uniform(shape=(), maxval=1) < 0.5,
            lambda: (input_image, enhanced_image),
            lambda: (
                tf.image.flip_left_right(input_image),
                tf.image.flip_left_right(enhanced_image),
            ),
        )

    def random_vertical_flip(self, input_image, enhanced_image):
        return tf.cond(
            tf.random.uniform(shape=(), maxval=1) < 0.5,
            lambda: (input_image, enhanced_image),
            lambda: (
                tf.image.flip_up_down(input_image),
                tf.image.flip_up_down(enhanced_image),
            ),
        )

    def random_rotate(self, input_image, enhanced_image):
        condition = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        return tf.image.rot90(input_image, condition), tf.image.rot90(
            enhanced_image, condition
        )


class UnpairedAugmentationFactory:
    def __init__(self, image_size) -> None:
        self.image_size = image_size

    def random_crop(self, image):
        image_shape = tf.shape(image)[:2]
        crop_w = tf.random.uniform(
            shape=(), maxval=image_shape[1] - self.image_size + 1, dtype=tf.int32
        )
        crop_h = tf.random.uniform(
            shape=(), maxval=image_shape[0] - self.image_size + 1, dtype=tf.int32
        )
        return image[
            crop_h : crop_h + self.image_size, crop_w : crop_w + self.image_size
        ]

    def random_horizontal_flip(self, image):
        return tf.cond(
            tf.random.uniform(shape=(), maxval=1) < 0.5,
            lambda: image,
            lambda: tf.image.flip_left_right(image),
        )

    def random_vertical_flip(self, image):
        return tf.cond(
            tf.random.uniform(shape=(), maxval=1) < 0.5,
            lambda: image,
            lambda: tf.image.flip_up_down(image),
        )

    def random_rotate(self, image):
        condition = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        return tf.image.rot90(image, condition)
