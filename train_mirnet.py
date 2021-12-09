r"""Train an MIRNet Model for low-light image-enhancement.
Sample Usage:
python train_mirnet.py --experiment_configs configs/mirnet_train.py
"""

from absl import app, flags, logging
from ml_collections.config_flags import config_flags

from enhance_me.mirnet import MIRNet

FLAGS = flags.FLAGS
flags.DEFINE_string("wandb_experiment_name", "pointnet_shapenet_core", "W&B Project Name.")
flags.DEFINE_string("wandb_api_key", None, "W&B Project Name.")
config_flags.DEFINE_config_file("experiment_configs")


def main(_):
    logging.info("Initializing MIRNet...")
    mirnet = MIRNet(
        experiment_name=FLAGS.wandb_experiment_name,
        wandb_api_key=None if FLAGS.wandb_api_key == "" else FLAGS.wandb_api_key,
    )
    logging.info(f"Building {FLAGS.experiment_configsdataset_label} Dataset...")
    mirnet.build_datasets(
        image_size=FLAGS.experiment_configsimage_size,
        dataset_label=FLAGS.experiment_configsdataset_label,
        apply_random_horizontal_flip=FLAGS.experiment_configsapply_random_horizontal_flip,
        apply_random_vertical_flip=FLAGS.experiment_configsapply_random_vertical_flip,
        apply_random_rotation=FLAGS.experiment_configsapply_random_rotation,
        val_split=FLAGS.experiment_configsval_split,
        batch_size=FLAGS.experiment_configsbatch_size,
    )
    logging.info("Building MIRNet Model...")
    mirnet.build_model(
        use_mixed_precision=FLAGS.experiment_configsuse_mixed_precision,
        num_recursive_residual_groups=FLAGS.experiment_configsnum_recursive_residual_groups,
        num_multi_scale_residual_blocks=FLAGS.experiment_configsnum_multi_scale_residual_blocks,
        learning_rate=FLAGS.experiment_configslearning_rate,
        epsilon=FLAGS.experiment_configsepsilon,
    )
    logging.info("Starting MIRNet Training...")
    history = mirnet.train(epochs=FLAGS.experiment_configsepochs)
    logging.info("Training Finished!!!")


if __name__ == "__main__":
    app.run(main)
