"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
Script: Implementation of different functions for the cosmological part.
"""

import numpy as np


def sigma_eight(cosmo: dict) -> float:
    """Calculates sigma_8 from a dictionary containing the cosmological parameters
    Args:
        cosmo (dict): A dictionary containing omega_cdm, omega_b, S_8, h
    Returns:
        float: the value of sigma_8
    """
    cdm = cosmo['omega_cdm']
    baryon = cosmo['omega_b']
    neutrino = cosmo['m_ncdm'] / 93.14

    # omega_matter (contains factor h**2)
    omega_matter = cdm + baryon + neutrino

    # actual omega_matter
    omega_matter /= cosmo['h']**2

    return cosmo['S_8'] * np.sqrt(0.3 / omega_matter)
