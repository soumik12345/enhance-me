r"""Train an Zero-DCE Model for low-light image-enhancement.
Sample Usage:
python train_zero_dce.py --experiment_configs configs/zero_dce_train.py
"""

import os
from absl import app, flags, logging
from ml_collections.config_flags import config_flags

from enhance_me import ZeroDCE

FLAGS = flags.FLAGS
flags.DEFINE_string("wandb_experiment_name", "zero-dce-exp", "W&B Project Name.")
flags.DEFINE_string("wandb_api_key", None, "W&B Project Name.")
config_flags.DEFINE_config_file("experiment_configs")


def main(_):
    logging.info("Initializing Zero-DCE...")
    zero_dce = ZeroDCE(
        experiment_name=FLAGS.wandb_experiment_name,
        wandb_api_key=None if FLAGS.wandb_api_key == "" else FLAGS.wandb_api_key,
        use_mixed_precision=FLAGS.experiment_configs.use_mixed_precision,
    )
    logging.info(f"Building {FLAGS.experiment_configs.dataset_label} Dataset...")
    zero_dce.build_datasets(
        image_size=FLAGS.experiment_configs.image_size,
        dataset_label=FLAGS.experiment_configs.dataset_label,
        apply_resize=FLAGS.experiment_configs.apply_resize,
        apply_random_horizontal_flip=FLAGS.experiment_configs.apply_random_horizontal_flip,
        apply_random_vertical_flip=FLAGS.experiment_configs.apply_random_vertical_flip,
        apply_random_rotation=FLAGS.experiment_configs.apply_random_rotation,
        val_split=FLAGS.experiment_configs.val_split,
        batch_size=FLAGS.experiment_configs.batch_size,
    )
    logging.info("Starting Zero-DCE Training...")
    zero_dce.compile(learning_rate=FLAGS.experiment_configs.learning_rate)
    history = zero_dce.train(epochs=FLAGS.experiment_configs.epochs)
    logging.info("Training Finished!!!")
    weights_path = os.path.join(FLAGS.experiment_configs.experiment_name, "weights.h5")
    zero_dce.save_weights(weights_path)
    logging.info(f"Saved weights at {weights_path}")


if __name__ == "__main__":
    app.run(main)
