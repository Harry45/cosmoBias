"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
"""
from absl import flags, app
from ml_collections.config_flags import config_flags

# our scripts
from src.cosmo import class_compute
from utils.checkers import make_paths
from utils.logger import get_logger


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Main configuration file.", lock_config=True)


def main(argv):
    """
    Run the main script.
    """
    logger = get_logger(FLAGS.config)
    logger.info("Running main script")

    make_paths(FLAGS.config)

    cosmology = {'omega_cdm': 0.12, 'omega_b': 0.02,
                 'S_8': 0.70, 'n_s': 1.0, 'h': 0.70}
    class_compute(FLAGS.config, cosmology)

    # cosmologies = scale_lhs(FLAGS.config, 'lhs_5d_1000', True, fname='5d_1000')
    # powerspectra = generate_training_pk(FLAGS.config, fname='5d_1000')


if __name__ == "__main__":
    app.run(main)
