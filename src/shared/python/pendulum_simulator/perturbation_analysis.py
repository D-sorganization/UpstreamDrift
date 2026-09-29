"""
Monte Carlo perturbation analysis for swing consistency evaluation.

Adds configurable noise to joint torque profiles, runs N simulations,
and computes variability statistics on velocity and position outputs.

Design by Contract
------------------
- n_trials > 0
- noise_amplitude >= 0
- noise_type in {'white', 'pink', 'brown'}
- All returned statistics are finite.

DRY
---
Reuses the polynomial torque builder and integrator from existing modules.
Noise generation is factored into a standalone function for reuse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class SimulateFn(Protocol):
    """Protocol for simulation callable."""

    def __call__(self, coeffs: list[list[float]]) -> object: ...


class ExtractFn(Protocol):
    """Protocol for metric extraction callable."""

    def __call__(self, result: object) -> dict[str, float | np.ndarray]: ...


# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------


def generate_noise(
    noise_type: str,
    n_samples: int,
    amplitude: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a 1-D noise signal.

    Parameters
    ----------
    noise_type : str â€” 'white', 'pink', or 'brown'
    n_samples : int â€” number of samples
    amplitude : float â€” standard deviation of the output signal
    seed : int, optional â€” for reproducibility

    Returns
    -------
    np.ndarray, shape (n_samples,)

    Design by Contract
    ------------------
    Pre:  noise_type in {'white', 'pink', 'brown'}
    Pre:  n_samples > 0, amplitude >= 0
    Post: output shape is (n_samples,)
    """
    assert n_samples > 0, f"n_samples must be positive, got {n_samples}"
    assert amplitude >= 0, f"amplitude must be non-negative, got {amplitude}"

    rng = np.random.default_rng(seed)

    if noise_type == "white":
        noise = rng.normal(0.0, amplitude, size=n_samples)

    elif noise_type == "pink":
        # Pink noise (1/f): filter white noise via cumulative sum + differentiation
        white = rng.normal(0.0, 1.0, size=n_samples)
        # Use Voss-McCartney approximation: sum of octave bands
        pink = np.zeros(n_samples)
        n_octaves = max(1, int(np.log2(n_samples)))
        for k in range(n_octaves):
            step = 2**k
            hold = rng.normal(0.0, 1.0, size=(n_samples + step - 1) // step)
            pink += np.repeat(hold, step)[:n_samples]
        # Normalize and scale
        if np.std(pink) > 0:
            pink[:] = (pink / np.std(pink)) * amplitude
        noise = pink

    elif noise_type == "brown":
        # Brown (Brownian) noise: cumulative sum of white noise
        white = rng.normal(0.0, 1.0, size=n_samples)
        brown = np.cumsum(white)
        # Normalize and scale
        if np.std(brown) > 0:
            brown = brown / np.std(brown) * amplitude
        noise = brown

    else:
        raise ValueError(
            f"Unknown noise type: {noise_type!r}. Must be 'white', 'pink', or 'brown'."
        )

    assert noise.shape == (n_samples,), (
        f"Expected shape ({n_samples},), got {noise.shape}"
    )
    return noise


# ---------------------------------------------------------------------------
# Torque profile perturbation (additive noise on raw time-series)
# ---------------------------------------------------------------------------


def perturb_torque_profile(
    profile: np.ndarray,
    noise_amplitude: float,
    noise_type: str = "additive",
    seed: int | None = None,
) -> np.ndarray:
    """Perturb a raw torque time-series profile with additive noise.

    Parameters
    ----------
    profile : np.ndarray, shape (n_steps,)
        Nominal torque profile over time.
    noise_amplitude : float
        Standard deviation of the additive Gaussian noise. Use 0.0 for no noise.
    noise_type : str
        Currently only 'additive' is supported (zero-mean Gaussian noise).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_steps,)
        Perturbed torque profile.

    Design by Contract
    ------------------
    Pre:  profile.ndim == 1, len(profile) > 0
    Pre:  noise_amplitude >= 0
    Pre:  noise_type == 'additive'
    Post: output.shape == profile.shape
    """
    if profile.ndim != 1 or len(profile) == 0:
        raise ValueError(
            f"profile must be a non-empty 1-D array, got shape {profile.shape}"
        )
    if noise_amplitude < 0:
        raise ValueError(f"noise_amplitude must be non-negative, got {noise_amplitude}")
    if noise_type != "additive":
        raise ValueError(f"noise_type must be 'additive', got {noise_type!r}")
    if noise_amplitude == 0.0:
        return profile.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_amplitude, size=len(profile))
    result = profile + noise
    assert result.shape == profile.shape
    return result


# ---------------------------------------------------------------------------
# Torque coefficient perturbation
# ---------------------------------------------------------------------------


def perturb_torque_coeffs(
    coeffs: list[list[float]],
    noise_amplitude: float,
    noise_type: str = "white",
    seed: int | None = None,
    perturb_mode: str = "additive",
) -> list[list[float]]:
    """Perturb polynomial torque coefficients with noise.

    Each coefficient is independently perturbed by adding noise scaled
    to the given amplitude.

    Parameters
    ----------
    coeffs : list of lists â€” per-joint polynomial coefficients
    noise_amplitude : float â€” amplitude of the perturbation
    noise_type : str â€” noise colour
    seed : int, optional
    perturb_mode : str â€" perturbation mode (currently only 'additive' is used)

    Returns
    -------
    list of lists â€” perturbed coefficients (same shape as input)

    Design by Contract
    ------------------
    Pre:  noise_amplitude >= 0
    Pre:  noise_type in {'white', 'pink', 'brown'}
    Post: output has same shape as input
    """
    assert noise_amplitude >= 0
    assert noise_type in {
        "white",
        "pink",
        "brown",
    }, f"noise_type must be 'white', 'pink', or 'brown'; got {noise_type!r}"
    assert perturb_mode in {
        "additive",
        "multiplicative",
        "both",
    }, (
        f"perturb_mode must be 'additive', 'multiplicative', or 'both'; got {perturb_mode!r}"
    )

    if noise_amplitude == 0.0:
        return [list(c) for c in coeffs]

    # Count total coefficients
    total = sum(len(c) for c in coeffs)
    noise = generate_noise(noise_type, total, noise_amplitude, seed)

    idx = 0
    result = []
    for joint_coeffs in coeffs:
        n = len(joint_coeffs)
        perturbed = [c + noise[idx + i] for i, c in enumerate(joint_coeffs)]
        result.append(perturbed)
        idx += n

    assert len(result) == len(coeffs)
    return result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PerturbationConfig:
    """Configuration for Monte Carlo perturbation analysis.

    Attributes
    ----------
    n_trials : int â€” number of Monte Carlo simulations
    noise_type : str â€” 'white', 'pink', or 'brown'
    noise_amplitude : float â€” perturbation amplitude (relative to peak torque)
    seed : int, optional â€” base seed for reproducibility
    """

    n_trials: int = 100
    noise_type: str = "white"
    noise_amplitude: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        assert self.n_trials > 0, f"n_trials must be positive, got {self.n_trials}"
        assert self.noise_amplitude >= 0, (
            f"noise_amplitude must be non-negative, got {self.noise_amplitude}"
        )
        assert self.noise_type in {
            "white",
            "pink",
            "brown",
        }, f"noise_type must be 'white', 'pink', or 'brown', got {self.noise_type!r}"


# ---------------------------------------------------------------------------
# Variability summary
# ---------------------------------------------------------------------------


def variability_summary(
    results: list[dict],
) -> dict[str, float | np.ndarray]:
    """Compute statistical summary from batch simulation results.

    Parameters
    ----------
    results : list of dicts, each with:
        'tip_speed_final': float
        'tip_position_final': np.ndarray, shape (2,)

    Returns
    -------
    dict with:
        'tip_speed_mean', 'tip_speed_std', 'tip_speed_cv',
        'tip_speed_min', 'tip_speed_max',
        'tip_position_mean', 'tip_position_std'

    Design by Contract
    ------------------
    Pre:  len(results) > 0
    Post: all values are finite
    """
    assert len(results) > 0, "results must be non-empty"

    speeds = np.array([r["tip_speed_final"] for r in results])
    positions = np.array([r["tip_position_final"] for r in results])

    speed_mean = float(np.mean(speeds))
    speed_std = float(np.std(speeds))
    speed_cv = speed_std / speed_mean if speed_mean != 0 else 0.0

    summary: dict[str, float | np.ndarray] = {
        "tip_speed_mean": speed_mean,
        "tip_speed_std": speed_std,
        "tip_speed_cv": speed_cv,
        "tip_speed_min": float(np.min(speeds)),
        "tip_speed_max": float(np.max(speeds)),
        "tip_position_mean": np.mean(positions, axis=0),
        "tip_position_std": np.std(positions, axis=0),
        "n_trials": len(results),
    }

    return summary


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------


def batch_perturb_and_simulate(
    base_coeffs: list[list[float]],
    config: PerturbationConfig,
    simulate_fn: SimulateFn,
    extract_fn: ExtractFn,
) -> list[dict]:
    """Run N perturbed simulations and collect results.

    Parameters
    ----------
    base_coeffs : list of lists â€” nominal polynomial torque coefficients
    config : PerturbationConfig
    simulate_fn : callable(coeffs) -> result
        Function that takes perturbed coefficients and returns a simulation result.
    extract_fn : callable(result) -> dict
        Function that extracts metrics from a simulation result.
        Must return dict with at least 'tip_speed_final' and 'tip_position_final'.

    Returns
    -------
    list of dicts â€” one per trial, each from extract_fn

    Design by Contract
    ------------------
    Pre:  config.n_trials > 0
    Post: len(output) == config.n_trials (or fewer if some trials fail)
    """
    assert base_coeffs is not None, "base_coeffs must be provided"
    results = []
    base_seed = config.seed if config.seed is not None else 0

    for i in range(config.n_trials):
        trial_seed = base_seed + i
        perturbed = perturb_torque_coeffs(
            base_coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            seed=trial_seed,
        )

        try:
            sim_result = simulate_fn(perturbed)
            metrics = extract_fn(sim_result)
            results.append(metrics)
        except (ValueError, RuntimeError, FloatingPointError):
            logger.warning("Trial %d failed, skipping", i, exc_info=True)
            continue

    logger.info(
        "Batch perturbation complete: %d / %d trials succeeded",
        len(results),
        config.n_trials,
    )
    return results
