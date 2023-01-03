## cosmoBias

Implementation of a Gibbs-HMC sampling scheme for inferring EFT bias parameters.

### Reference Papers

-   [HEFTY](https://arxiv.org/abs/2103.09820)
-   [MUSE](https://arxiv.org/abs/2112.09354)
-   [DES Y1](https://arxiv.org/abs/1708.01530)

### Mathematical Notes

The set of bias parameters are:

$$
\boldsymbol{b} = (b_{0}, b_{1}, b_{2}, b_{s}, b_{\nabla})
$$

and the cross-power spectrum between the galaxy and matter overdensities is:

$$
P_{gm}(k)=\sum_{\alpha \in \mathcal{O}} b_{\alpha}P_{1\alpha}(k).
$$

The galaxy-galaxy power spectrum is given by:

$$
P_{gg}(k) = \sum_{\alpha \in \mathcal{O}}\sum_{\beta \in \mathcal{O}}b_{\alpha}b_{\beta}P_{\alpha\beta}(k)
$$

where $\mathcal{O}\equiv\{1,\delta_{L},\delta_{L}^{2},s_{L}^{2},\nabla^{2}\delta_{L}\}$ is the full set of operators. $P_{\alpha \beta}(k)$ is the power spectrum of the fields $\alpha$ and $\beta$ and $b_{\alpha}$ are the corresponding bias coefficients. $P_{11}(k)$ is the non-linear matter power spectrum, and its corresponding bias parameter is $b_{0}=1$.

### To Do

- (MWE - Simulation) Gibbs sampling mechanism.
- (MWE - Simulation) Vary cosmological parameters as well.
- Own MH or HMC sampler.
- Can try EMCEE as well.