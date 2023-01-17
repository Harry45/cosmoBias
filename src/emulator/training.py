# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: In this code, we train the GP model using the training data.

import os
import logging
import torch
import matplotlib.pylab as plt
from ml_collections import ConfigDict

# our script and functions
import utils.helpers as hp
from .gaussianprocess import GaussianProcess

plt.rc("text", usetex=True)
plt.rc("font", **{"family": "sans-serif", "serif": ["Palatino"]})
LOGGER = logging.getLogger(__name__)


def plot_loss(config: ConfigDict, optim: dict, fname: str, save: bool = True):
    """Plots the loss of the GP model.

    Args:
        config (ConfigDict): the main configuration file with all the settings.
        optim (dict): A dictionary with the optimizer. There can be more than 1 optimisation.
        fname (str): name of the file to be stored.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """

    nopt = len(optim)
    niter = len(optim[0]["loss"])

    # plot the loss
    plt.figure(figsize=(8, 8))
    for i in range(nopt):
        plt.plot(range(niter), optim[i]["loss"], label=f"Optimization {i + 1}")
    plt.xlabel("Iterations", fontsize=config.plot.fontsize)
    plt.ylabel("Loss", fontsize=config.plot.fontsize)
    plt.tick_params(axis="x", labelsize=config.plot.fontsize)
    plt.tick_params(axis="y", labelsize=config.plot.fontsize)
    plt.legend(loc="best", prop={"family": "sans-serif", "size": 15})

    path = os.path.join(config.path.plots, "loss", str(config.emu.nlhs))
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "/" + fname + ".pdf", bbox_inches="tight")
    plt.savefig(path + "/" + fname + ".png", bbox_inches="tight")
    if save:
        plt.close()
    else:
        plt.show()


def train_gps(config: ConfigDict) -> list:
    """Train the Gaussian Processes and store them.

    Args:
        config (ConfigDict): the main configuration file with all the settings.

    Returns:
        list: A list of Gaussian Processes.
    """
    nlhs = config.emu.nlhs
    inputs = hp.load_csv("data", "cosmologies_lhs_" + str(nlhs))
    if config.boolean.linearpk:
        fname = 'pk_lin_lhs_' + str(nlhs)
    else:
        fname = 'pk_non_lhs_' + str(nlhs)
    outputs = hp.load_pkl("data", fname)
    ins = torch.from_numpy(inputs.values)
    gps = list()

    for i in range(config.grid.nk):

        LOGGER.info('Training GP: %d', i)
        out = torch.from_numpy(outputs[:, i])

        # the GP module
        gp_module = GaussianProcess(config, ins, out)

        # perform the optimisation of the GP model
        opt = gp_module.optimisation(
            torch.randn(config.cosmo.nparams + 1),
            niter=config.emu.niter,
            lrate=config.emu.lr,
            nrestart=config.emu.nrestart,
        )

        gps.append(gp_module)

        # plot and store the loss function of the GP model
        plot_loss(config, opt, f"{fname}_wave_{i}")

        # save the GP model
        path = os.path.join(config.path.gps, str(nlhs))
        os.makedirs(path, exist_ok=True)

        # name of the files to save
        gp_name = f"{fname}_wave_{i}"
        pa_name = "params_" + f"{fname}_wave_{i}"
        al_name = "alpha_" + f"{fname}_wave_{i}"

        hp.save_pkl(gp_module, path, gp_name)
        hp.save_pkl(gp_module.opt_parameters.data.numpy(), path, pa_name)
        hp.save_pkl(gp_module.alpha.data.numpy(), path, al_name)
    return gps
