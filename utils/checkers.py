"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
"""
import os
import logging
from ml_collections.config_dict import ConfigDict

logger = logging.getLogger(__name__)


def create_init(path: str):
    """Create the init file.

    Args:
        path (str): path where we want to create the init file.
    """
    with open(path + "__init__.py", "w+") as file:
        file.write('import os')


def make_paths(config: ConfigDict) -> None:
    """Make sure all relevant folders where we store outputs exist.
    Args:
        config (ConfigDict): the main configuration file.
    """
    logger.info('Checking if all paths exist.')

    os.makedirs(config.path.data, exist_ok=True)
    os.makedirs(config.path.plots, exist_ok=True)
    os.makedirs(config.path.logs, exist_ok=True)

    create_init(config.path.data)
    create_init(config.path.plots)
    create_init(config.path.logs)
