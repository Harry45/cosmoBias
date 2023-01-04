"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
Script: The main configuration file
"""
import os
from ml_collections.config_dict import ConfigDict


def get_config() -> ConfigDict:
    """Generates the main configuration file for Class and the emulator. Note that the wavenumber is in inverse Mpc.
    Returns:
        ConfigDict: the configuration file
    """

    config = ConfigDict()
    config.logname = 'Infer-Bias'

    # boolean settings
    config.boolean = boolean = ConfigDict()
    boolean.linearpk = False

    # paths
    config.path = path = ConfigDict()
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    path.data = 'data/'
    path.plots = 'plots/'
    path.logs = 'logs/'

    # cosmological parameters
    config.cosmo = cosmo = ConfigDict()
    cosmo.names = ['omega_cdm', 'omega_b', 'S_8', 'n_s', 'h']
    cosmo.distribution = 'uniform'
    cosmo.loc = [0.051, 0.019, 0.10, 0.84, 0.64]
    cosmo.scale = [0.204, 0.007, 1.20, 0.26, 0.18]
    cosmo.fiducial = [0.12, 0.020, 0.76, 1.0, 0.70]
    cosmo.nparams = len(cosmo.names)

    # bias parameters
    config.bias = bias = ConfigDict()
    bias.names = ['b0', 'b1', 'b2', 'bs', 'bn']
    bias.distribution = 'normal'
    bias.loc = [1.0, 1.5, 2.0, 2.5, 3.0]
    bias.scale = [0.5, 0.5, 0.5, 0.5, 0.5]
    bias.fiducial = [1.0, 1.5, 2.0, 2.5, 3.0]
    bias.nparams = len(bias.names)

    # CLASS settings
    config.classy = classy = ConfigDict()
    classy.halofit_k_per_decade = 80
    classy.halofit_sigma_precision = 0.05
    classy.output = "mPk"
    classy.mode = 'hmcode'
    classy.cmin = 3.13
    # classy.bbn = '/home/harry/Desktop/class/bbn/sBBN.dat'
    classy.bbn = '/home/mootovaloo/Desktop/class/external/bbn/sBBN.dat'
    classy.k_pivot = 0.05
    classy.Omega_k = 0.0
    classy.k_max_pk = 50
    classy.z_max_pk = 3.0

    # Grid settings
    config.grid = grid = ConfigDict()
    grid.kmin = 1E-4
    grid.kmax = 5.0
    grid.nk = 100
    grid.zmin = 0.0
    grid.zmax = 3.0
    grid.nz = 20

    # neutrino settings
    config.neutrino = neutrino = ConfigDict()
    neutrino.N_ncdm = 1.0
    neutrino.deg_ncdm = 3.0
    neutrino.T_ncdm = 0.71611
    neutrino.N_ur = 0.00641
    neutrino.fixed_nm = 0.06

    return config
