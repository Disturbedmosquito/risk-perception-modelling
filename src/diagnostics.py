"""
Model diagnostics and information criteria.

This module provides functions to calculate log-likelihood-based
diagnostics and model selection criteria (AIC, BIC) specifically
for the negative binomial observation model used in the fitting
process.

"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from .observation import nb_loglik

def _as_mask(n: int, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Internal helper to create or validate a boolean mask.

    n : int
        The required length of the mask.
    mask : np.ndarray, optional
        An existing mask to validate. If None, a mask of all `True`
        is created.

    Returns: np.ndarray - A boolean mask of length `n`.

    Raises ValueError if `mask` is provided but its length != `n`.

    """
    if mask is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != n:
        raise ValueError(f"mask length {mask.shape[0]} != n={n}")
    return mask


def info_criteria_nb(
    y: np.ndarray,
    mu: np.ndarray,
    log_theta: float,
    p_params: int,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute log-likelihood-based diagnostics for a Negative Binomial fit.

    Calculates the total log-likelihood (LL) and uses it to compute
    the Negative Log-Likelihood (NLL), Akaike Information Criterion (AIC),
    and Bayesian Information Criterion (BIC).

    Parameters
    ----------
    y : np.ndarray
        The array of observed data (ground truth).
    mu : np.ndarray
        The array of model-predicted means (must be same shape as `y`).
    log_theta : float
        The fitted scalar log-dispersion parameter (log(theta)).
    p_params : int
        The total number of *fitted* parameters in the model
        (e.g., len(free_parameters)).
    mask : np.ndarray, optional
        A boolean mask to apply to `y` and `mu` before calculation
        (e.g., to drop a burn-in period).

    Returns
    -------
    dict
        A dictionary containing the following metrics:
        - 'n': Number of data points used.
        - 'll': Total log-likelihood.
        - 'nll': Total negative log-likelihood (-ll).
        - 'nll_per_obs': NLL divided by n.
        - 'AIC': Akaike Information Criterion.
        - 'BIC': Bayesian Information Criterion.
        - 'k': The number of parameters (`p_params`) used.

    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape")

    m = _as_mask(len(y), mask)
    y_m = y[m]; mu_m = mu[m]
    n = y_m.size

    ll = nb_loglik(y_m, mu_m, log_theta)
    nll = -ll
    aic = -2.0 * ll + 2.0 * p_params
    bic = -2.0 * ll + p_params * np.log(max(1, n))

    return dict(
        n=float(n),
        ll=float(ll),
        nll=float(nll),
        nll_per_obs=float(nll / max(1, n)),
        AIC=float(aic),
        BIC=float(bic),
        k=float(p_params),
    )
