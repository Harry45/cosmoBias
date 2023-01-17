"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
"""
from absl import flags, app
from ml_collections.config_flags import config_flags

# our scripts
from utils.checkers import make_paths
from utils.logger import get_logger
from trainingpoints import pk_training_set
from src.emulator.training import train_gps

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.")


def main(argv):
    """
    Run the main script.
    """
    logger = get_logger(FLAGS.config)
    logger.info("Running main script")

    make_paths(FLAGS.config)

    # 1) Generate training points
    # cosmologies, powerspec = pk_training_set('lhs_1000', FLAGS.config, redshift=0)

    # 2) Train the Gaussian Processes
    gps = train_gps(FLAGS.config)

    # 3) Infer Parameters


if __name__ == "__main__":
    app.run(main)
