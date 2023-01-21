"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
"""
import os
from typing import Union
import torch
import numpy as np
from ml_collections import ConfigDict

# our scripts and functions
from utils.helpers import load_pkl
from utils.interpolation import spline_interpolate
from utils.checkers import check_wavenumbers


def load_gps(config: ConfigDict) -> list:
    """Load all the trained GPs to make predictions

    Args:
        config (ConfigDict): the main configuration file with all the settings.

    Returns:
        list: a list of the trained GPs
    """
    gps = list()
    path = os.path.join(config.path.gps, str(config.emu.nlhs))
    for i in range(config.grid.nk):
        trained_gp = load_pkl(path, f'pk_non_lhs_1000_wave_{i}')
        gps.append(trained_gp)
    return gps


def interpolate_pk(predictions: np.ndarray, wavenumbers: np.ndarray, config: ConfigDict) -> np.ndarray:
    """Interpolate the power spectrum along the wavenumber axis.

    Args:
        predictions (np.ndarray): the predictions from the GP.
        wavenumbers (np.ndarray): the new set of wavenumbers.
        config (ConfigDict): the configuration file with all the settings.

    Returns:
        np.ndarray: the interpolated power spectrum
    """

    check_wavenumbers(config, wavenumbers)
    grid_k = np.geomspace(config.grid.kmin, config.grid.kmax, config.grid.nk)
    pred_new = spline_interpolate(np.log(grid_k), np.log(predictions), wavenumbers)
    pred_new = np.exp(pred_new)
    return pred_new


def interpolate_gradients(gradients: np.ndarray, wavenumbers: np.ndarray, config: ConfigDict) -> np.ndarray:
    """Interpolate the gradients given a new set of wavenumbers.

    Args:
        gradients (np.ndarray): the gradients calculated at a specific set of wavenumbers.
        wavenumbers (np.ndarray): the new set of wavenumbers.
        config (ConfigDict): the main configuration file with all the settings.

    Returns:
        np.ndarray: the interpolated gradients
    """
    check_wavenumbers(config, wavenumbers)
    grid_k = np.geomspace(config.grid.kmin, config.grid.kmax, config.grid.nk)
    grad_new = np.zeros((wavenumbers.shape[0], config.cosmo.nparams))
    for i in range(config.cosmo.nparams):
        grad_new[:, i] = spline_interpolate(np.log(grid_k), gradients[:, i], wavenumbers)
    return grad_new


def interpolate_hessian(hessian: np.ndarray, wavenumbers: np.ndarray, config: ConfigDict) -> np.ndarray:
    """Interpolate the Hessian given a new set of wavenumbers.

    Args:
        hessian (np.ndarray): the second derivatives of the power spectrum, N x p x p
        wavenumbers (np.ndarray): the new set of wavenumbers
        config (ConfigDict): the main configuration file with all the settings

    Returns:
        np.ndarray: the Hessian interpolated along the wavenumber axis
    """
    check_wavenumbers(config, wavenumbers)
    grid_k = np.geomspace(config.grid.kmin, config.grid.kmax, config.grid.nk)
    hessian_new = np.zeros((wavenumbers.shape[0], config.cosmo.nparams, config.cosmo.nparams))
    for i in range(config.cosmo.nparams):
        for j in range(config.cosmo.nparams):
            hessian_new[:, i, j] = spline_interpolate(np.log(grid_k), hessian[:, i, j], wavenumbers)
    return hessian_new


class GPcalculations:
    """Computes all relevant quantities for the emulated power spectrum.

    Args:
        config (ConfigDict): The main configuration file with all the settings.
    """

    def __init__(self, config: ConfigDict):
        self.cfg = config
        self.gps = load_gps(self.cfg)

    def mean_prediction(self, testpoint: torch.Tensor, wavenumbers: np.ndarray = None) -> np.ndarray:
        """Calculate the mean prediction of the power spectrum

        Args:
            testpoint (torch.Tensor): the point where we want to compute the power spectrum
            wavenumbers (np.ndarray): the new values of the wavenumbers

        Returns:
            np.ndarray: either the original values of power spectrum or the the interpolated ones
        """
        pred = np.zeros(self.cfg.grid.nk)
        for i in range(self.cfg.grid.nk):
            pred[i] = self.gps[i].prediction(testpoint).view(-1).item()

        if wavenumbers is not None:
            pred_new = interpolate_pk(pred, wavenumbers, self.cfg)
            return pred_new
        return pred

    def derivatives(self, testpoint: torch.Tensor, order: int = 1,
                    wavenumbers: np.ndarray = None) -> Union[np.ndarray, np.ndarray]:
        """Calculates the derivatives of the power spectrum with respect to the input cosmological parameters.

        Args:
            testpoint (torch.Tensor): the point where we want to compute the derivatives.
            order (int, optional): either the first or second derivatives. Defaults to 1.
            wavenumbers (np.ndarray, optional): Option to provide a set of wavenumbers. Defaults to None.

        Returns:
            Union[np.ndarray, np.ndarray]: either a matrix of (N x p), for the gradient and/or
            a tensor of size (N x p x p) for the Hessian.

            N is the number of wavenumbers and p is the number of parameters.
        """
        grad = np.zeros((self.cfg.grid.nk, self.cfg.cosmo.nparams))
        hessian = np.zeros((self.cfg.grid.nk, self.cfg.cosmo.nparams, self.cfg.cosmo.nparams))

        if order == 1:

            for i in range(self.cfg.grid.nk):
                grad[i] = self.gps[i].derivatives(testpoint, order).view(-1)

            if wavenumbers is not None:
                grad_new = interpolate_gradients(grad, wavenumbers, self.cfg)
                return grad_new
            return grad

        for i in range(self.cfg.grid.nk):
            grad[i], hessian[i] = self.gps[i].derivatives(testpoint, order)

        if wavenumbers is not None:
            grad_new = interpolate_gradients(grad, wavenumbers, self.cfg)
            hessian_new = interpolate_hessian(hessian, wavenumbers, self.cfg)
            return grad_new, hessian_new
        return grad, hessian
