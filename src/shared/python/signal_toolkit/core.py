"""Core signal classes and utilities.

This module provides the fundamental Signal class and SignalGenerator
for creating and manipulating time-series data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Signal:
    """Represents a time-domain signal with associated metadata.

    Attributes:
        time: Time array (1D numpy array).
        values: Signal values (1D or 2D numpy array).
        name: Optional name for the signal.
        units: Optional units string (e.g., 'N*m', 'rad/s').
        metadata: Additional metadata dictionary.
    """

    time: np.ndarray
    values: np.ndarray
    name: str = ""
    units: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and convert inputs to numpy arrays."""
        self.time = np.asarray(self.time, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)

        if self.time.ndim != 1:
            msg = f"Time must be 1D array, got shape {self.time.shape}"
            raise ValueError(msg)

        if self.values.ndim == 1:
            if len(self.time) != len(self.values):
                msg = (
                    f"Time and values must have same length: "
                    f"{len(self.time)} vs {len(self.values)}"
                )
                raise ValueError(msg)
        elif self.values.ndim == 2:
            if self.values.shape[0] != len(self.time):
                msg = (
                    f"First dimension of values must match time length: "
                    f"{self.values.shape[0]} vs {len(self.time)}"
                )
                raise ValueError(msg)
        else:
            msg = f"Values must be 1D or 2D array, got shape {self.values.shape}"
            raise ValueError(msg)

    @property
    def fs(self) -> float:
        """Sampling frequency in Hz."""
        if len(self.time) < 2:
            return 1.0
        return float(1.0 / np.mean(np.diff(self.time)))

    @property
    def dt(self) -> float:
        """Time step in seconds."""
        if len(self.time) < 2:
            return 1.0
        return float(np.mean(np.diff(self.time)))

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if len(self.time) < 2:
            return 0.0
        return self.time[-1] - self.time[0]

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.time)

    def copy(self) -> Signal:
        """Create a deep copy of the signal."""
        return Signal(
            time=self.time.copy(),
            values=self.values.copy(),
            name=self.name,
            units=self.units,
            metadata=dict(self.metadata),
        )

    def slice(self, t_start: float, t_end: float) -> Signal:
        """Extract a time slice of the signal.

        Args:
            t_start: Start time (inclusive).
            t_end: End time (inclusive).

        Returns:
            New Signal with the sliced data.
        """
        mask = (self.time >= t_start) & (self.time <= t_end)
        return Signal(
            time=self.time[mask],
            values=self.values[mask] if self.values.ndim == 1 else self.values[mask, :],
            name=self.name,
            units=self.units,
            metadata=dict(self.metadata),
        )

    def resample(self, new_fs: float) -> Signal:
        """Resample the signal to a new sampling frequency.

        Args:
            new_fs: New sampling frequency in Hz.

        Returns:
            New Signal with resampled data.
        """
        new_dt = 1.0 / new_fs
        new_time = np.arange(self.time[0], self.time[-1], new_dt)
        new_values = np.interp(new_time, self.time, self.values)
        return Signal(
            time=new_time,
            values=new_values,
            name=self.name,
            units=self.units,
            metadata=dict(self.metadata),
        )

    def __add__(self, other: Signal | float | np.ndarray) -> Signal:
        """Add two signals or add a constant."""
        if isinstance(other, Signal):
            if not np.allclose(self.time, other.time):
                msg = "Signals must have the same time array for addition"
                raise ValueError(msg)
            return Signal(
                time=self.time.copy(),
                values=self.values + other.values,
                name=f"({self.name} + {other.name})",
                units=self.units,
                metadata=dict(self.metadata),
            )
        return Signal(
            time=self.time.copy(),
            values=self.values + other,
            name=self.name,
            units=self.units,
            metadata=dict(self.metadata),
        )

    def __mul__(self, other: Signal | float | np.ndarray) -> Signal:
        """Multiply two signals or multiply by a constant."""
        if isinstance(other, Signal):
            if not np.allclose(self.time, other.time):
                msg = "Signals must have the same time array for multiplication"
                raise ValueError(msg)
            return Signal(
                time=self.time.copy(),
                values=self.values * other.values,
                name=f"({self.name} * {other.name})",
                units=self.units,
                metadata=dict(self.metadata),
            )
        return Signal(
            time=self.time.copy(),
            values=self.values * other,
            name=self.name,
            units=self.units,
            metadata=dict(self.metadata),
        )

    def __neg__(self) -> Signal:
        """Negate the signal."""
        return Signal(
            time=self.time.copy(),
            values=-self.values,
            name=f"-{self.name}",
            units=self.units,
            metadata=dict(self.metadata),
        )


class SignalGenerator:
    """Factory class for generating common signal types."""

    @staticmethod
    def constant(
        t: np.ndarray,
        value: float,
        name: str = "constant",
    ) -> Signal:
        """Generate a constant signal.

        Args:
            t: Time array.
            value: Constant value.
            name: Signal name.

        Returns:
            Signal with constant value.
        """
        return Signal(
            time=t,
            values=np.full_like(t, value),
            name=name,
        )

    @staticmethod
    def sinusoid(
        t: np.ndarray,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
        name: str = "sinusoid",
    ) -> Signal:
        """Generate a sinusoidal signal.

        Args:
            t: Time array.
            amplitude: Peak amplitude.
            frequency: Frequency in Hz.
            phase: Phase offset in radians.
            offset: DC offset.
            name: Signal name.

        Returns:
            Signal: y = amplitude * sin(2*pi*frequency*t + phase) + offset
        """
        values = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def cosine(
        t: np.ndarray,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        phase: float = 0.0,
        offset: float = 0.0,
        name: str = "cosine",
    ) -> Signal:
        """Generate a cosine signal.

        Args:
            t: Time array.
            amplitude: Peak amplitude.
            frequency: Frequency in Hz.
            phase: Phase offset in radians.
            offset: DC offset.
            name: Signal name.

        Returns:
            Signal: y = amplitude * cos(2*pi*frequency*t + phase) + offset
        """
        values = amplitude * np.cos(2 * np.pi * frequency * t + phase) + offset
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def exponential(
        t: np.ndarray,
        amplitude: float = 1.0,
        decay_rate: float = 1.0,
        offset: float = 0.0,
        name: str = "exponential",
    ) -> Signal:
        """Generate an exponential signal.

        Args:
            t: Time array.
            amplitude: Initial amplitude (at t=0).
            decay_rate: Decay rate (positive = decay, negative = growth).
            offset: DC offset.
            name: Signal name.

        Returns:
            Signal: y = amplitude * exp(-decay_rate * t) + offset
        """
        t_shifted = t - t[0]  # Start from t=0
        values = amplitude * np.exp(-decay_rate * t_shifted) + offset
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def linear(
        t: np.ndarray,
        slope: float = 1.0,
        intercept: float = 0.0,
        name: str = "linear",
    ) -> Signal:
        """Generate a linear signal (ramp).

        Args:
            t: Time array.
            slope: Slope of the line.
            intercept: Y-intercept at t=0.
            name: Signal name.

        Returns:
            Signal: y = slope * t + intercept
        """
        t_shifted = t - t[0]  # Start from t=0
        values = slope * t_shifted + intercept
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def polynomial(
        t: np.ndarray,
        coefficients: list[float] | np.ndarray,
        name: str = "polynomial",
    ) -> Signal:
        """Generate a polynomial signal.

        Args:
            t: Time array.
            coefficients: Polynomial coefficients [c0, c1, c2, ...]
                where y = c0 + c1*t + c2*t^2 + ...
            name: Signal name.

        Returns:
            Signal with polynomial values.
        """
        coeffs = np.asarray(coefficients)
        t_shifted = t - t[0]  # Start from t=0
        values = np.polyval(coeffs[::-1], t_shifted)
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def step(
        t: np.ndarray,
        step_time: float = 0.0,
        step_value: float = 1.0,
        initial_value: float = 0.0,
        name: str = "step",
    ) -> Signal:
        """Generate a step signal.

        Args:
            t: Time array.
            step_time: Time at which step occurs.
            step_value: Value after step.
            initial_value: Value before step.
            name: Signal name.

        Returns:
            Signal with step at step_time.
        """
        values = np.where(t >= step_time, step_value, initial_value)
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def pulse(
        t: np.ndarray,
        start_time: float = 0.0,
        duration: float = 1.0,
        amplitude: float = 1.0,
        baseline: float = 0.0,
        name: str = "pulse",
    ) -> Signal:
        """Generate a rectangular pulse signal.

        Args:
            t: Time array.
            start_time: Start time of pulse.
            duration: Duration of pulse.
            amplitude: Pulse amplitude.
            baseline: Baseline value outside pulse.
            name: Signal name.

        Returns:
            Signal with rectangular pulse.
        """
        values = np.where(
            (t >= start_time) & (t < start_time + duration),
            amplitude,
            baseline,
        )
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def chirp(
        t: np.ndarray,
        f0: float = 1.0,
        f1: float = 10.0,
        amplitude: float = 1.0,
        method: str = "linear",
        name: str = "chirp",
    ) -> Signal:
        """Generate a frequency-swept chirp signal.

        Args:
            t: Time array.
            f0: Initial frequency in Hz.
            f1: Final frequency in Hz.
            amplitude: Signal amplitude.
            method: Sweep method ('linear' or 'exponential').
            name: Signal name.

        Returns:
            Signal with chirp waveform.
        """
        t_shifted = t - t[0]
        t_end = t_shifted[-1]

        if method == "linear":
            # Linear frequency sweep
            k = (f1 - f0) / t_end
            phase = 2 * np.pi * (f0 * t_shifted + 0.5 * k * t_shifted**2)
        elif method == "exponential":
            # Exponential frequency sweep
            if f0 <= 0 or f1 <= 0:
                msg = "Frequencies must be positive for exponential chirp"
                raise ValueError(msg)
            k = (f1 / f0) ** (1 / t_end)
            phase = 2 * np.pi * f0 * (k**t_shifted - 1) / np.log(k)
        else:
            msg = f"Unknown chirp method: {method}"
            raise ValueError(msg)

        values = amplitude * np.sin(phase)
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def sawtooth(
        t: np.ndarray,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
        name: str = "sawtooth",
    ) -> Signal:
        """Generate a sawtooth wave.

        Args:
            t: Time array.
            frequency: Frequency in Hz.
            amplitude: Peak-to-peak amplitude.
            offset: DC offset.
            name: Signal name.

        Returns:
            Signal with sawtooth waveform.
        """
        period = 1.0 / frequency
        t_shifted = t - t[0]
        phase = (t_shifted % period) / period
        values = amplitude * (2 * phase - 1) + offset
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def triangle(
        t: np.ndarray,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
        name: str = "triangle",
    ) -> Signal:
        """Generate a triangle wave.

        Args:
            t: Time array.
            frequency: Frequency in Hz.
            amplitude: Peak amplitude.
            offset: DC offset.
            name: Signal name.

        Returns:
            Signal with triangle waveform.
        """
        period = 1.0 / frequency
        t_shifted = t - t[0]
        phase = (t_shifted % period) / period
        # Triangle: 4 * |phase - 0.5| - 1
        values = amplitude * (4 * np.abs(phase - 0.5) - 1) + offset
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def square(
        t: np.ndarray,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        duty_cycle: float = 0.5,
        offset: float = 0.0,
        name: str = "square",
    ) -> Signal:
        """Generate a square wave.

        Args:
            t: Time array.
            frequency: Frequency in Hz.
            amplitude: Peak amplitude.
            duty_cycle: Fraction of period that signal is high (0-1).
            offset: DC offset.
            name: Signal name.

        Returns:
            Signal with square waveform.
        """
        period = 1.0 / frequency
        t_shifted = t - t[0]
        phase = (t_shifted % period) / period
        values = np.where(phase < duty_cycle, amplitude, -amplitude) + offset
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def from_function(
        t: np.ndarray,
        func: Callable[[np.ndarray], np.ndarray],
        name: str = "custom",
    ) -> Signal:
        """Generate a signal from a custom function.

        Args:
            t: Time array.
            func: Function that takes time array and returns values.
            name: Signal name.

        Returns:
            Signal with values from the function.
        """
        values = func(t)
        return Signal(time=t, values=values, name=name)

    @staticmethod
    def superposition(
        signals: list[Signal],
        name: str = "superposition",
    ) -> Signal:
        """Create a superposition (sum) of multiple signals.

        Args:
            signals: List of Signal objects with matching time arrays.
            name: Name for the result signal.

        Returns:
            Signal that is the sum of all input signals.
        """
        if not signals:
            msg = "At least one signal required for superposition"
            raise ValueError(msg)

        result = signals[0].copy()
        result.name = name

        for sig in signals[1:]:
            if not np.allclose(result.time, sig.time):
                msg = "All signals must have the same time array"
                raise ValueError(msg)
            result.values = result.values + sig.values

        return result
