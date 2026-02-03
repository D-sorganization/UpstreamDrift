"""Noise models for sensor simulation.

This module provides configurable noise models for realistic sensor simulation,
enabling sim-to-real transfer and robustness testing.

Design by Contract:
    All noise models are deterministic given a seed.
    Output dimensions match input dimensions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


class NoiseModel(ABC):
    """Abstract base class for noise models.

    All noise models transform a clean signal into a noisy one,
    with configurable parameters.
    """

    @abstractmethod
    def apply(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply noise to signal.

        Args:
            signal: Clean signal array.

        Returns:
            Noisy signal with same shape as input.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g., bias drift)."""
        ...


@dataclass
class GaussianNoise(NoiseModel):
    """Additive white Gaussian noise.

    Adds i.i.d. Gaussian noise to each element of the signal.

    Attributes:
        std: Standard deviation of noise.
        mean: Mean of noise (bias).
        seed: Random seed for reproducibility.
    """

    std: float = 0.01
    mean: float = 0.0
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random generator."""
        self._rng = np.random.default_rng(self.seed)

    def apply(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Gaussian noise to signal.

        Args:
            signal: Clean signal.

        Returns:
            Signal with additive Gaussian noise.
        """
        noise = self._rng.normal(self.mean, self.std, signal.shape)
        return signal + noise

    def reset(self) -> None:
        """Reset random generator to initial seed."""
        self._rng = np.random.default_rng(self.seed)


@dataclass
class BrownianNoise(NoiseModel):
    """Brownian (random walk) noise for bias drift.

    Models slowly-varying bias that accumulates over time,
    common in IMU sensors.

    Attributes:
        drift_rate: Standard deviation of drift increment per step.
        initial_bias: Starting bias value.
        max_bias: Maximum absolute bias (clipped).
        seed: Random seed for reproducibility.
    """

    drift_rate: float = 0.001
    initial_bias: float = 0.0
    max_bias: float = 1.0
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _current_bias: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize state."""
        self._rng = np.random.default_rng(self.seed)
        self._current_bias = self.initial_bias

    def apply(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply bias drift to signal.

        Args:
            signal: Clean signal.

        Returns:
            Signal with additive drifting bias.
        """
        # Update bias with random walk
        drift = self._rng.normal(0, self.drift_rate)
        self._current_bias += drift

        # Clip to max bias
        self._current_bias = np.clip(
            self._current_bias, -self.max_bias, self.max_bias
        )

        return signal + self._current_bias

    def reset(self) -> None:
        """Reset bias to initial value."""
        self._rng = np.random.default_rng(self.seed)
        self._current_bias = self.initial_bias

    @property
    def current_bias(self) -> float:
        """Get current bias value."""
        return self._current_bias


@dataclass
class QuantizationNoise(NoiseModel):
    """Quantization noise from ADC resolution.

    Models the discrete nature of digital sensors.

    Attributes:
        resolution: Quantization step size (LSB).
        offset: Offset before quantization.
    """

    resolution: float = 0.001
    offset: float = 0.0

    def apply(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply quantization to signal.

        Args:
            signal: Continuous signal.

        Returns:
            Quantized signal.
        """
        shifted = signal - self.offset
        quantized = np.round(shifted / self.resolution) * self.resolution
        return quantized + self.offset

    def reset(self) -> None:
        """No state to reset."""
        pass


@dataclass
class BandwidthLimitedNoise(NoiseModel):
    """Bandwidth-limited noise using low-pass filter.

    Models sensor bandwidth limitations.

    Attributes:
        cutoff_frequency: Filter cutoff frequency [Hz].
        sample_rate: Sampling rate [Hz].
        order: Filter order.
    """

    cutoff_frequency: float = 100.0
    sample_rate: float = 1000.0
    order: int = 2
    _filter_state: NDArray[np.float64] | None = field(
        init=False, repr=False, default=None
    )
    _alpha: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize filter coefficient."""
        # Simple first-order IIR approximation
        dt = 1.0 / self.sample_rate
        tau = 1.0 / (2 * np.pi * self.cutoff_frequency)
        self._alpha = dt / (tau + dt)

    def apply(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply low-pass filter to signal.

        Args:
            signal: Input signal.

        Returns:
            Filtered signal.
        """
        if self._filter_state is None:
            self._filter_state = signal.copy()
            return signal.copy()

        # First-order IIR filter: y = alpha * x + (1-alpha) * y_prev
        self._filter_state = (
            self._alpha * signal + (1 - self._alpha) * self._filter_state
        )
        return self._filter_state.copy()

    def reset(self) -> None:
        """Reset filter state."""
        self._filter_state = None


@dataclass
class CompositeNoise(NoiseModel):
    """Composite noise model combining multiple noise sources.

    Applies multiple noise models in sequence.

    Attributes:
        models: List of noise models to apply in order.
    """

    models: list[NoiseModel] = field(default_factory=list)

    def apply(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply all noise models in sequence.

        Args:
            signal: Clean signal.

        Returns:
            Signal with all noise sources applied.
        """
        result = signal.copy()
        for model in self.models:
            result = model.apply(result)
        return result

    def reset(self) -> None:
        """Reset all noise models."""
        for model in self.models:
            model.reset()

    def add_model(self, model: NoiseModel) -> None:
        """Add a noise model to the composite.

        Args:
            model: Noise model to add.
        """
        self.models.append(model)


def create_realistic_sensor_noise(
    noise_std: float = 0.01,
    bias_drift_rate: float = 0.0001,
    quantization_bits: int = 16,
    signal_range: float = 100.0,
    seed: int | None = None,
) -> CompositeNoise:
    """Create a realistic composite noise model.

    Combines Gaussian noise, bias drift, and quantization.

    Args:
        noise_std: Standard deviation of white noise.
        bias_drift_rate: Bias drift rate per timestep.
        quantization_bits: ADC resolution in bits.
        signal_range: Full-scale signal range.
        seed: Random seed.

    Returns:
        Composite noise model with realistic characteristics.
    """
    resolution = signal_range / (2**quantization_bits)

    return CompositeNoise(
        models=[
            BrownianNoise(drift_rate=bias_drift_rate, seed=seed),
            GaussianNoise(std=noise_std, seed=seed),
            QuantizationNoise(resolution=resolution),
        ]
    )
