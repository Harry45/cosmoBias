"""
Generates the training points: cosmological parameters (inputs) and linear matter power
spectrum (targets).

Author: Arrykrishna Mootoovaloo
Collaborators: David, Pedro, Jaime
Date: January 2023
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: Emulator for computing the linear matter power spectrum
"""
import os
import logging
from typing import Tuple
import scipy.stats
import pandas as pd
import numpy as np
from ml_collections import ConfigDict

# our scripts and functions
from src.cosmology.cosmo import calculate_pk_fixed_redshift
import utils.helpers as hp

LOGGER = logging.getLogger(__name__)


def generate_cosmo_prior(config: ConfigDict) -> dict:
    """Generates the entity of each parameter by using scipy.stats function.
    Args:
        dictionary (dict): A dictionary with the specifications of the prior.
    Returns:
        dict: the prior distribution of all parameters.
    """
    dictionary = dict()
    for i, key in enumerate(config.cosmo.names):
        specs = (config.cosmo.loc[i], config.cosmo.scale[i])
        dictionary[key] = getattr(scipy.stats, config.cosmo.distribution)(*specs)
    return dictionary


def scale_lhs(config: ConfigDict, fname: str = "lhs_500", save: bool = True) -> list:
    """Scale the Latin Hypercube Samples according to the prior range.

    Args:
        fname (str, optional): The name of the LHS file. Defaults to 'lhs_500'.
        save (bool): Whether to save t he scaled LHS samples. Defaults to True.

    Returns:
        list: A list of dictionaries with the scaled LHS samples.
    """

    # read the LHS samples
    path = os.path.join("data", fname + ".csv")
    lhs = pd.read_csv(path, index_col=[0])
    ncosmo = lhs.shape[0]
    priors = generate_cosmo_prior(config)

    # create an empty list to store the cosmologies
    cosmo_list = list()
    for i in range(ncosmo):
        cosmo = lhs.iloc[i, :]
        cosmo_dict = dict()
        for k in range(config.cosmo.nparams):
            value = priors[config.cosmo.names[k]].ppf(cosmo[k])
            cosmo_dict[config.cosmo.names[k]] = round(value, 6)
        cosmo_list.append(cosmo_dict)
    if save:
        cosmos_df = pd.DataFrame(cosmo_list)
        hp.save_csv(cosmos_df, "data", "cosmologies_" + fname)
        hp.save_pkl(cosmo_list, "data", "cosmologies_" + fname)
    return cosmo_list


def pk_training_set(fname: str, config: ConfigDict, redshift: float = 0.0) -> Tuple[list, np.ndarray]:
    """Generates the training set for building the emulator.

    Args:
        fname (str): name of the LHS file.
        config (ConfigDict): the main configuration file with all settings.
        redshift (float, optional): the redshift at which the power spectrum is calculated. Defaults to 0.0.

    Returns:
        Tuple[list, np.ndarray]: the cosmologies and the power spectra
    """
    LOGGER.info('Generating training points.')
    cosmologies = scale_lhs(config, fname, save=True)
    pk_record = list()
    for i, cosmo in enumerate(cosmologies):
        print(i, cosmo)
        pk_calc = calculate_pk_fixed_redshift(config, cosmo, redshift)
        pk_record.append(pk_calc)

    pk_record = np.asarray(pk_record)
    if config.boolean.linearpk:
        filename = 'pk_lin_' + fname
    else:
        filename = 'pk_non_' + fname
    hp.save_pkl(pk_record, "data", filename)
    return cosmologies, pk_record
