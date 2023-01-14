"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
"""
import torch


def gaussian_kernel(arr1: torch.Tensor, arr2: torch.Tensor, hyperparam: float) -> torch.Tensor:
    """Calculate the Gaussian kernel given two set of inputs.

    https://nenadmarkus.com/p/all-pairs-euclidean/

    Args:
        input1 (torch.Tensor): the first set of input
        input2 (torch.Tensor): the second set of input
        hyperparam (float): the kernel hyperparameter (lengthscale only)

    Returns:
        torch.Tensor: the Gaussian kernel.
    """
    size_a = arr1.shape[0]
    size_b = arr2.shape[0]

    # divide by the characteristic lengthscale
    arr1 = arr1 / hyperparam
    arr2 = arr2 / hyperparam

    # calculate term squared
    sqr_a = torch.sum(torch.pow(arr1, 2), 1, keepdim=True)
    sqr_b = torch.sum(torch.pow(arr2, 2), 1, keepdim=True)

    sqr_a = sqr_a.expand(size_a, size_b)
    sqr_b = sqr_b.expand(size_a, size_b).t()

    # calculate pairwise distance
    dist = sqr_a - 2 * torch.mm(arr1, arr2.t()) + sqr_b
    return torch.exp(-0.5 * dist)


def compute_alpha(xinput: torch.Tensor, target: torch.tensor, hyperparam: float) -> torch.Tensor:
    """_summary_

    Args:
        xinput (torch.Tensor): _description_
        target (torch.tensor): _description_
        hyperparam (float):

    Returns:
        torch.Tensor: _description_
    """
    kernel = gaussian_kernel(xinput, xinput, hyperparam)
    kernel = kernel + 1E-6 * torch.eye(kernel.shape[0])
    alpha = torch.linalg.solve(kernel, target)
    return alpha


class KernelInterpolation:
    """Performs Kernel Interpolation - see here for an example:

    http://www.kernel-operations.io/keops/_auto_tutorials/interpolation/plot_RBF_interpolation_torch.html

    Args:
        domain (torch.Tensor): the inputs to the function
        target (torch.Tensor): the targets
        hyperparam (float, optional): the hyperparameter in the Gaussian Kernel. Defaults to 1.0.
    """

    def __init__(self, domain: torch.Tensor, target: torch.Tensor, hyperparam: float = 1.0):

        self.domain = domain.view(1, -1)
        self.target = target.view(-1, 1)
        self.hyperparam = hyperparam
        self.alpha = compute_alpha(self.domain, self.target, self.hyperparam)

    def predict(self, xtest: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            xtest (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        k_star = gaussian_kernel(self.domain, xtest, self.hyperparam)
        y_pred = k_star.t() @ self.alpha
        return y_pred
