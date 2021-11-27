import os
import wandb
from glob import glob
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import utils


def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.cast(image, dtype=tf.float32) / 255.0
    return image


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()


def closest_number(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2


def init_wandb(project_name, experiment_name, wandb_api_key):
    if project_name is not None and experiment_name is not None:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.init(project=project_name, name=experiment_name, sync_tensorboard=True)


def download_lol_dataset():
    utils.get_file(
        "lol_dataset.zip",
        "https://github.com/soumik12345/enhance-me/releases/download/v0.1/lol_dataset.zip",
        cache_dir="./",
        cache_subdir="./datasets",
        extract=True,
    )
    low_images = sorted(glob("./datasets/lol_dataset/our485/low/*"))
    enhanced_images = sorted(glob("./datasets/lol_dataset/our485/high/*"))
    assert len(low_images) == len(enhanced_images)
    test_low_images = sorted(glob("./datasets/lol_dataset/eval15/low/*"))
    test_enhanced_images = sorted(glob("./datasets/lol_dataset/eval15/high/*"))
    assert len(test_low_images) == len(test_enhanced_images)
    return (low_images, enhanced_images), (test_low_images, test_enhanced_images)
