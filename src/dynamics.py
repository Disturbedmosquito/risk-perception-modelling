"""
Functions for analyzing the model's dynamic properties.

This module provides tools for stability analysis, equilibrium finding,
and time-series analysis of the model's dynamics.

It provides specialized "reduced" versions of functions for Jacobian 
calculation (`numerical_jacobian`) and equilibrium solving 
(`equilibrium_newton`) that operate on the 5-state system [I, C, R, P, F], 
enforcing the conservation law `S = N - I - C - R`.

"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch, csd, hilbert

def dominant_period_days(series, fs=1.0):
    """
    Finds the dominant period in a time series using Welch's method.

    Parameters
    ----------
    series : np.ndarray
        The 1D time series data.
    fs : float, optional
        The sampling frequency (default is 1.0, e.g., 1 sample per day).

    Returns
    -------
    tuple
        (period, f_star, f, Pxx)
        - period : float - The dominant period (in days if fs=1.0).
        - f_star : float - The dominant frequency.
        - f : np.ndarray - The array of frequencies.
        - Pxx : np.ndarray - The power spectral density.

    """
    f, Pxx = welch(series, fs=fs, nperseg=min(256, len(series)))
    idx = np.argmax(Pxx[1:]) + 1  # skip zero freq
    f_star = f[idx]
    period = np.inf if f_star <= 0 else 1.0 / f_star
    
    return period, f_star, f, Pxx


def cross_spectrum(x, y, fs=1.0):
    """
    Computes the cross-spectral density (CSD) of two time series.

    Parameters
    ----------
    x : np.ndarray
        The first time series.
    y : np.ndarray
        The second time series.
    fs : float, optional
        The sampling frequency.

    Returns
    -------
    tuple
        (f, Pxy)
        - f : np.ndarray - The array of frequencies.
        - Pxy : np.ndarray - The cross-spectral density.

    """
    f, Pxy = csd(x, y, fs=fs, nperseg=min(256, len(x)))
    return f, Pxy


def amplitude_envelope(series):
    """Computes the amplitude envelope of a series using the Hilbert transform.
    """
    return np.abs(hilbert(series))


def sirc_pf_rhs_reduced(t, y_reduced, pars, N, original_rhs_fn):
    """
    RHS wrapper for the 5-state reduced system [I, C, R, P, F].
    It takes the 5-state vector `y_reduced`, calculates the 6th state
    `S = N - I - C - R`, and then calls the `original_rhs_fn` with the
    reconstructed 6-state vector.

    It returns only the 5 derivatives corresponding to `y_reduced`,
    enforcing the conservation law implicitly.

    Parameters
    ----------
    t : float
        Time (typically 0.0 for equilibrium finding).
    y_reduced : np.ndarray
        The 5-state vector [I, C, R, P, F].
    pars : dict
        The parameter dictionary for the ODE function.
    N : float
        The total population size.
    original_rhs_fn : callable
        The *original* 6-state RHS function (e.g., `model.sirc_pf_rhs`).

    Returns
    -------
    np.ndarray
        The 5-state derivative vector [dIdt, dCdt, dRdt, dPdt, dFdt].

    """
    I, C, R, P, F = y_reduced
    S = N - I - C - R
    
    # Check for biologically plausible state
    if S < 0:
        # Return a large derivative to push the solver back
        return np.full(5, 1e10) 
        
    y_full = np.array([S, I, C, R, P, F])
    
    full_ders = original_rhs_fn(t, y_full, pars)
    
    # Return only the 5 derivatives for the reduced states
    return full_ders[1:] # [dSdt, dIdt, dCdt, dRdt, dPdt, dFdt]


def numerical_jacobian_reduced(fun_reduced, y_star_reduced, pars, N, original_rhs_fn, eps=1e-6):
    """
    Calculates the 5x5 numerical Jacobian of `sirc_pf_rhs_REDUCED` with
    respect to the 5-state vector `y_star_reduced`.

    Parameters
    ----------
    fun_reduced : callable
        The reduced 5-state RHS wrapper (`sirc_pf_rhs_REDUCED`).
    y_star_reduced : np.ndarray
        The 5-state equilibrium vector [I, C, R, P, F].
    pars : dict
        The parameter dictionary.
    N : float
        The total population size.
    original_rhs_fn : callable
        The original 6-state RHS function.
    eps : float, optional
        The step size for the finite difference calculation.

    Returns
    -------
    np.ndarray
        The 5x5 numerical Jacobian matrix.

    """
    y_star_reduced = np.asarray(y_star_reduced, float)
    n = len(y_star_reduced) # n=5
    J = np.zeros((n, n))
    
    # Pass all required args to the reduced function
    f0 = fun_reduced(0.0, y_star_reduced, pars, N, original_rhs_fn)
    
    for j in range(n):
        e = np.zeros(n); e[j] = eps
        f1 = fun_reduced(0.0, y_star_reduced + e, pars, N, original_rhs_fn)
        J[:, j] = (f1 - f0) / eps
        
    return J


def equilibrium_newton_reduced(
    fun_reduced, y_guess_reduced, pars, N, original_rhs_fn, 
    max_iter=50, tol=1e-10
):
    """
    Solves for equilibrium (f(y_reduced)=0) using the 5-state system
    using Newton's method on the 5-state reduced system, which avoids the
    singularity of the 6-state Jacobian by implicitly enforcing the
    conservation law `S+I+C+R = N`.

    Parameters
    ----------
    fun_reduced : callable
        The reduced 5-state RHS wrapper (`sirc_pf_rhs_REDUCED`).
    y_guess_reduced : np.ndarray
        A 5-state initial guess [I, C, R, P, F].
    pars : dict
        The parameter dictionary.
    N : float
        The total population size.
    original_rhs_fn : callable
        The original 6-state RHS function (e.g., `model.sirc_pf_rhs`).
    max_iter : int, optional
        Maximum number of Newton iterations.
    tol : float, optional
        Tolerance for convergence.

    Returns
    -------
    np.ndarray
        The 5-state equilibrium vector `y_star_reduced`.

    Raises
    ------
    RuntimeError
        If the solver fails to converge within `max_iter`.

    """
    y = y_guess_reduced.copy()
    for _ in range(max_iter):
        f = fun_reduced(0.0, y, pars, N, original_rhs_fn)
        
        if np.linalg.norm(f, ord=np.inf) < tol:
            return y
            
        J = numerical_jacobian_reduced(fun_reduced, y, pars, N, original_rhs_fn)
        
        try:
            # 5x5 solve
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            # Fallback just in case
            step = -np.linalg.pinv(J) @ f
            
        y = y + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            return y
            
    raise RuntimeError("Equilibrium not found (reduced 5x5 system)")