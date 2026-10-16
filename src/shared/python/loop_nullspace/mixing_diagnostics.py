"""Mixing diagnostics for sampled chains (NSX-70, #12588).

Autocorrelation, split R-hat and the multi-chain effective sample size, after
Gelman et al. (BDA3, 11.4-11.5) and Vehtari et al. (2021) without rank
normalisation.  They diagnose *non-convergence evidence*: a passing value is
never proof of convergence.  Chains are ``(n_chains, n_draws)`` arrays of one
scalar functional.
"""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.loop_nullspace.evidence import Array

__all__ = ["autocorrelation", "effective_sample_size", "split_rhat"]

_MIN_SPLIT_DRAWS = 8


def _chains(chains: Array, *, min_draws: int = 4) -> Array:
    arr = np.asarray(chains, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"chains must be 2-D (n_chains, n_draws), got {arr.ndim}-D")
    if arr.shape[1] < min_draws:
        raise ValueError(f"need at least {min_draws} draws per chain")
    if not np.all(np.isfinite(arr)):
        raise ValueError("chains must be finite")
    return arr


def _autocovariance(x: Array) -> Array:
    """Biased (1/n) autocovariance at every lag, by FFT."""
    n = x.size
    centered = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(centered, size)
    acov = np.fft.irfft(spec * np.conj(spec), size)[:n] / n
    return np.asarray(acov, dtype=np.float64)


def autocorrelation(x: Array, max_lag: int) -> Array:
    """Autocorrelation of one chain at lags ``0..max_lag`` (lag 0 is 1).

    Preconditions: 1-D finite ``x``, ``0 <= max_lag < len(x)``.  Returns NaN
    for a constant chain.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError("x must be a finite 1-D chain")
    if not 0 <= max_lag < arr.size:
        raise ValueError(f"max_lag must be in [0, {arr.size - 1}], got {max_lag}")
    acov = _autocovariance(arr)
    if acov[0] <= 0.0:
        return np.full(max_lag + 1, np.nan)
    return np.asarray(acov[: max_lag + 1] / acov[0], dtype=np.float64)


def split_rhat(chains: Array) -> float:
    """Split potential scale reduction factor (each chain cut into two halves).

    Values well above 1 (a common screening threshold is 1.01) indicate chains
    that disagree or drift.  Returns NaN when every draw is identical.
    Preconditions: finite 2-D input with at least 8 draws per chain.
    """
    arr = _chains(chains, min_draws=_MIN_SPLIT_DRAWS)
    half = arr.shape[1] // 2
    parts = np.concatenate([arr[:, :half], arr[:, half : 2 * half]], axis=0)
    within = float(np.mean(parts.var(axis=1, ddof=1)))
    if within <= 0.0:
        return math.nan
    between = half * float(parts.mean(axis=1).var(ddof=1))
    var_plus = (half - 1) / half * within + between / half
    return math.sqrt(var_plus / within)


def effective_sample_size(chains: Array) -> float:
    """Multi-chain effective sample size (Geyer initial monotone sequence).

    ``ESS = m n / tau`` with ``tau`` the integrated autocorrelation time
    pooled over chains.  Can exceed ``m n`` for anti-correlated draws, bounded
    as in Stan.  Returns NaN for a constant chain.
    Preconditions: finite 2-D input with at least 4 draws per chain.
    """
    arr = _chains(chains)
    m, n = arr.shape
    acov = np.stack([_autocovariance(c) for c in arr])
    within = float(np.mean(acov[:, 0]) * n / (n - 1))
    between = float(arr.mean(axis=1).var(ddof=1)) * n if m > 1 else 0.0
    var_plus = (n - 1) / n * within + between / n
    if var_plus <= 0.0:
        return math.nan
    rho = 1.0 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = rho[: (n // 2) * 2].reshape(-1, 2).sum(axis=1)
    total, prev = 0.0, math.inf
    for pair in pairs:
        if pair < 0.0:
            break
        prev = min(prev, float(pair))
        total += prev
    tau = max(-1.0 + 2.0 * total, 1.0 / math.log10(m * n))
    return m * n / tau
