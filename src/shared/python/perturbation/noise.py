"""Noise generation for perturbation analysis."""

from __future__ import annotations

import numpy as np


def generate_noise(
    noise_type: str,
    n_samples: int,
    amplitude: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a 1-D noise signal.

    Parameters
    ----------
    noise_type : str
        'white', 'pink', or 'brown'.
    n_samples : int
        Number of samples to generate.
    amplitude : float
        Standard deviation of the output signal.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_samples,).

    Design by Contract
    ------------------
    Pre: noise_type in {'white', 'pink', 'brown'}
    Pre: n_samples > 0
    Pre: amplitude >= 0
    Post: output shape is (n_samples,)
    """
    assert n_samples > 0, f"n_samples must be positive, got {n_samples}"
    assert amplitude >= 0, f"amplitude must be non-negative, got {amplitude}"
    assert noise_type in {"white", "pink", "brown"}, f"Unknown noise_type: {noise_type}"

    rng = np.random.default_rng(seed)
    noise: np.ndarray

    if noise_type == "white":
        noise = rng.normal(0.0, amplitude, size=n_samples)

    elif noise_type == "pink":
        # Pink noise (1/f): filter white noise via cumulative sum + differentiation
        # We use Voss-McCartney approximation: sum of octave bands
        pink = np.zeros(n_samples)
        n_octaves = max(1, int(np.log2(n_samples)))
        for k in range(n_octaves):
            step = 2**k
            hold = rng.normal(0.0, 1.0, size=(n_samples + step - 1) // step)
            pink += np.repeat(hold, step)[:n_samples]

        # Normalize to variance=1, then scale
        if np.std(pink) > 0:
            pink = (pink / np.std(pink)) * amplitude  # type: ignore[operator]
        noise = pink  # type: ignore[assignment]

    elif noise_type == "brown":
        # Brown (Brownian) noise: cumulative sum of white noise
        white = rng.normal(0.0, 1.0, size=n_samples)
        brown = np.cumsum(white)
        # Normalize and scale
        if np.std(brown) > 0:
            brown = (brown / np.std(brown)) * amplitude
        noise = brown
    else:
        # Fallback for static type analyzers
        noise = np.zeros(n_samples)

    assert noise.shape == (
        n_samples,
    ), f"Expected shape ({n_samples},), got {noise.shape}"
    return noise
