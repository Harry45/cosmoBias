"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
"""
import os
import logging
from ml_collections.config_dict import ConfigDict

LOGGER = logging.getLogger(__name__)


def create_init(path: str):
    """Create the init file.

    Args:
        path (str): path where we want to create the init file.
    """
    file = os.path.join(path, '__init__.py')
    with open(file, "w+") as file:
        file.write('import os')


def make_paths(config: ConfigDict) -> None:
    """Make sure all relevant folders where we store outputs exist.
    Args:
        config (ConfigDict): the main configuration file.
    """
    LOGGER.info('Checking if all paths exist.')
    for path in list(config.path):
        os.makedirs(path, exist_ok=True)
        create_init(path)
