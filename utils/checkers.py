"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
"""
import os
import logging
import numpy as np
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


def check_wavenumbers(config: ConfigDict, wavenumbers: np.ndarray):
    """Ensure that the wavenumbers are within the range, as specified in the configuration file.

    Args:
        config (ConfigDict): the main configuration file with all the settings.
        wavenumbers (np.ndarray): the new wavenumbers
    """
    kmax_log = np.log(config.grid.kmax + 1E-5)
    kmin_log = np.log(config.grid.kmin - 1E-5)
    cond_min = np.all(wavenumbers >= kmin_log)
    cond_max = np.all(wavenumbers <= kmax_log)
    msg = f'Wavenumbers should be in log (base e) and between {config.grid.kmin} and {config.grid.kmax}'
    assert cond_min and cond_max, msg
