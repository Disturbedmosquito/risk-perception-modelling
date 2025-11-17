"""
Provides helper functions for assessing model performance
and generating forecast evaluation splits.

Includes:
- `time_series_splits`: A generator for rolling-origin
  cross-validation (CV) indices.
- Standard error metrics: `mae`, `rmse`.
- Probabilistic forecast metrics: `nb_mean_log_pred_density`.
- Prediction interval simulation: `simulate_nb_intervals`.

"""
from __future__ import annotations
import numpy as np

def time_series_splits(T, initial_train, test_horizon, step):
    """
    Generates rolling-origin time series cross-validation splits.

    Yields pairs of (train_idx, test_idx) arrays.
    The training set starts at size `initial_train` and expands,
    while the test set is a fixed block of size `test_horizon`
    that "rolls" forward by `step` days in each iteration.

    Parameters
    ----------
    T : int
        The total length of the time series.
    initial_train : int
        The number of data points in the *first* training set.
    test_horizon : int
        The number of data points in each test set.
    step : int
        The number of steps to advance the origin after each split.

    Returns
    ------
    tuple
        A (train_idx, test_idx) tuple, where each element is a
        `np.ndarray` of indices.

    """
    t_end = initial_train - 1
    while t_end + test_horizon < T - 1:
        train_idx = np.arange(0, t_end + 1)
        test_idx  = np.arange(t_end + 1, t_end + 1 + test_horizon)
        yield train_idx, test_idx
        t_end += step

def mae(y, yhat): return float(np.mean(np.abs(y - yhat))) # Mean Absolute Error
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2))) # Root Mean Squared Error

def nb_mean_log_pred_density(y, mu, log_theta):
    """
    Calculates the average log predictive density for our Negative Binomial model.

    Parameters
    ----------
    y : np.ndarray
        The array of true (observed) values in the test set.
    mu : np.ndarray
        The array of predicted means (mu) for the test set.
    log_theta : float
        The fitted scalar log-dispersion parameter.

    Returns
    -------
    float
        The mean log-likelihood per observation.

    """
    from .observation import nb_loglik
    return nb_loglik(y, mu, log_theta) / len(y)

def simulate_nb_intervals(mu, log_theta, n_paths=2000, alpha=(0.1, 0.5)):
    """
    Generates parametric bootstrap prediction intervals for a NB model.

    Simulates `n_paths` random trajectories from the fitted
    Negative Binomial distribution, specified by its mean `mu` and
    dispersion `log_theta`. It then computes quantiles from these
    simulations to form prediction intervals.

    Uses the Gamma-Poisson definition of the NB distribution
    for efficient simulation.

    Parameters
    ----------
    mu : np.ndarray
        The array of predicted means (one for each time step).
    log_theta : float
        The fitted scalar log-dispersion parameter.
    n_paths : int, optional
        The number of simulation paths to generate.
    alpha : tuple, optional
        A tuple defining the quantile levels.
        hardcoded to produce `alpha[0]/2` and `1-alpha[0]/2` (ex: 5% and 95%).

    Returns
    -------
    dict
        A dictionary with "lower", "median", and "upper" quantile
        trajectories, corresponding to the quantiles
        [ alpha[0]/2, 0.5, 1-alpha[0]/2 ].

    """
    theta = np.exp(log_theta)
    # sample y ~ NB(mean=mu, theta)
    # Convert to (r, p) with r=theta, p=theta/(theta+mu)
    p = theta / (theta + mu)
    r = theta
    rng = np.random.default_rng(123)
    # NB as Gamma-Poisson: Poisson(Gamma(r, (1-p)/p))
    lam = rng.gamma(shape=r, scale=(1-p)/p, size=(n_paths, len(mu)))  # mean=r*scale = r*(1-p)/p = mu
    y = rng.poisson(lam=lam)
    qs = np.quantile(y, q=[alpha[0]/2, 0.5, 1-alpha[0]/2], axis=0)
    return {"lower": qs[0], "median": qs[1], "upper": qs[2]}
