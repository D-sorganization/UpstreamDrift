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
        if signal is None:
            raise ValueError("signal must be provided")
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
        if signal is None:
            raise ValueError("signal must be provided")
        drift = self._rng.normal(0, self.drift_rate)
        self._current_bias += drift

        # Clip to max bias
        self._current_bias = np.clip(self._current_bias, -self.max_bias, self.max_bias)

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
        if signal is None:
            raise ValueError("signal must be provided")
        shifted = signal - self.offset
        quantized = np.round(shifted / self.resolution) * self.resolution
        return quantized + self.offset

    def reset(self) -> None:
        """No state to reset."""


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
    _filter_states: list[NDArray[np.float64] | None] = field(
        init=False, repr=False, default_factory=list
    )
    _alpha: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize filter coefficient and per-stage states."""
        if self.order < 1:
            raise ValueError("Filter order must be >= 1")
        # Simple first-order IIR approximation
        dt = 1.0 / self.sample_rate
        tau = 1.0 / (2 * np.pi * self.cutoff_frequency)
        self._alpha = dt / (tau + dt)
        self._filter_states = [None] * self.order

    def apply(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply nth-order low-pass filter by chaining first-order stages.

        The filter is applied ``self.order`` times in cascade to achieve
        a higher-order roll-off.

        Args:
            signal: Input signal.

        Returns:
            Filtered signal.
        """
        if signal is None:
            raise ValueError("signal must be provided")
        result = signal.copy()
        for stage in range(self.order):
            if self._filter_states[stage] is None:
                self._filter_states[stage] = result.copy()
            else:
                # First-order IIR filter: y = alpha * x + (1-alpha) * y_prev
                prev = self._filter_states[stage]
                assert prev is not None  # guarded by if-else above
                self._filter_states[stage] = (
                    self._alpha * result + (1 - self._alpha) * prev
                )
                result = self._filter_states[stage].copy()  # type: ignore[union-attr]
        return result

    def reset(self) -> None:
        """Reset filter state."""
        self._filter_states = [None] * self.order


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
        if signal is None:
            raise ValueError("signal must be provided")
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


@dataclass
class SensorNoiseParameters:
    """Parameters describing realistic sensor noise characteristics.

    Captures the full noise chain used by :class:`NoisySensor`:

        measured = saturate(raw * (1 + scale_factor_error) + bias + white_noise)

    where ``bias`` evolves as a random walk (Brownian) with initial offset.

    Default values target MEMS IMU / mid-grade force-torque sensors.
    Published datasheets informing the defaults:
        - Bosch BMI088 6-axis IMU (accel noise density 175 ug/sqrt(Hz),
          gyro noise density 0.014 deg/s/sqrt(Hz), bias instability
          ~0.1 deg/s, scale factor error ~0.5 %, saturation +/-24 g).
        - ATI Nano17 / Mini45 force-torque (resolution ~1/160 N on Fx,
          noise ~0.025 N RMS, scale factor accuracy ~0.75 % FS).

    Attributes:
        white_std: Additive white Gaussian noise standard deviation.
        bias_initial_std: Std of the initial bias drawn at construction
            (models turn-on bias repeatability).
        bias_walk_std: Std of the per-step bias random-walk increment
            (models in-run bias stability / flicker noise).
        scale_factor_std: Std of the multiplicative scale-factor error,
            sampled once at construction (dimensionless, fractional).
        saturation_limit: Absolute value beyond which the measurement
            saturates. ``None`` disables saturation.
        temperature_coefficient: Bias sensitivity per degree Celsius
            above the reference temperature (same units as white_std).
    """

    white_std: float = 0.01
    bias_initial_std: float = 0.005
    bias_walk_std: float = 1e-5
    scale_factor_std: float = 0.001
    saturation_limit: float | None = None
    temperature_coefficient: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameter ranges (non-negative stds, positive saturation)."""
        for name in (
            "white_std",
            "bias_initial_std",
            "bias_walk_std",
            "scale_factor_std",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.saturation_limit is not None and self.saturation_limit <= 0:
            raise ValueError("saturation_limit must be positive when provided")


@dataclass
class NoisySensor:
    """Composite sensor noise wrapper with bias drift + scale factor.

    Applies a realistic noise chain per call to :meth:`measure`::

        measured = clip(raw * (1 + scale_factor) + bias + temp_drift
                        + white_noise, +/- saturation_limit)

    The bias term is integrated as a random walk, so calling
    ``measure`` repeatedly produces slowly drifting bias on top of
    white noise. ``scale_factor`` is sampled once at construction
    (and on :meth:`reset`); it does not re-randomize per-step.

    Example:
        >>> params = SensorNoiseParameters(white_std=0.01, bias_walk_std=1e-4)
        >>> sensor = NoisySensor(params, seed=0)
        >>> raw = np.array([1.0, 2.0, 3.0])
        >>> noisy = sensor.measure(raw)
    """

    params: SensorNoiseParameters
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _bias: NDArray[np.float64] | float = field(init=False, repr=False)
    _scale_factor: NDArray[np.float64] | float = field(init=False, repr=False)
    _bias_initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Initialize random generator and sample initial bias/scale factor."""
        self._rng = np.random.default_rng(self.seed)
        # Scalars at first; promoted to arrays on first measurement so the
        # shape matches the measured signal.
        self._bias = 0.0
        self._scale_factor = 0.0
        self._bias_initialized = False

    def _ensure_state(self, shape: tuple[int, ...]) -> None:
        """Sample bias/scale factor with the correct shape on first call."""
        if self._bias_initialized:
            return
        self._bias = self._rng.normal(0.0, self.params.bias_initial_std, shape)
        self._scale_factor = self._rng.normal(0.0, self.params.scale_factor_std, shape)
        self._bias_initialized = True

    def measure(
        self,
        raw: NDArray[np.float64],
        temperature_delta: float = 0.0,
    ) -> NDArray[np.float64]:
        """Return a noisy measurement of ``raw``.

        Args:
            raw: True (clean) sensor signal.
            temperature_delta: Temperature deviation from reference [C].

        Returns:
            Measurement with bias drift, scale-factor error, temperature
            drift, additive white noise, and optional saturation applied.
        """
        if raw is None:
            raise ValueError("raw signal must be provided")
        raw = np.asarray(raw, dtype=np.float64)
        self._ensure_state(raw.shape)

        # Bias random walk
        if self.params.bias_walk_std > 0:
            self._bias = self._bias + self._rng.normal(
                0.0, self.params.bias_walk_std, raw.shape
            )

        # White noise
        white = (
            self._rng.normal(0.0, self.params.white_std, raw.shape)
            if self.params.white_std > 0
            else 0.0
        )

        temp_drift = self.params.temperature_coefficient * temperature_delta

        measured = raw * (1.0 + self._scale_factor) + self._bias + temp_drift + white

        if self.params.saturation_limit is not None:
            lim = self.params.saturation_limit
            measured = np.clip(measured, -lim, lim)

        return np.asarray(measured, dtype=np.float64)

    def reset(self) -> None:
        """Reset bias, scale factor, and RNG state to construction-time values."""
        self._rng = np.random.default_rng(self.seed)
        self._bias = 0.0
        self._scale_factor = 0.0
        self._bias_initialized = False

    @property
    def current_bias(self) -> NDArray[np.float64] | float:
        """Expose the current (random-walked) bias for diagnostics/tests."""
        return self._bias

    @property
    def scale_factor(self) -> NDArray[np.float64] | float:
        """Expose the sampled scale-factor error for diagnostics/tests."""
        return self._scale_factor


# Published-datasheet-informed parameter presets. Users can reach for
# these by name without chasing datasheets every time.
IMU_MEMS_DEFAULTS = SensorNoiseParameters(
    white_std=0.02,
    bias_initial_std=0.05,
    bias_walk_std=1e-4,
    scale_factor_std=5e-3,
    saturation_limit=160.0,  # ~16 g
    temperature_coefficient=1e-3,
)
"""Defaults approximating a consumer MEMS IMU (Bosch BMI088-class)."""

FORCE_TORQUE_INDUSTRIAL_DEFAULTS = SensorNoiseParameters(
    white_std=0.05,
    bias_initial_std=0.2,
    bias_walk_std=1e-3,
    scale_factor_std=7.5e-3,
    saturation_limit=1000.0,
    temperature_coefficient=5e-3,
)
"""Defaults approximating an industrial 6-axis force-torque sensor (ATI
Nano17/Mini45-class)."""


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
    if noise_std is None:
        raise ValueError("noise_std must be provided")
    resolution = signal_range / (2**quantization_bits)

    return CompositeNoise(
        models=[
            BrownianNoise(drift_rate=bias_drift_rate, seed=seed),
            GaussianNoise(std=noise_std, seed=seed),
            QuantizationNoise(resolution=resolution),
        ]
    )
