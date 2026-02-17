"""Noise generation and disturbance simulation.

This module provides tools for generating various types of noise
and adding disturbances to signals for simulation and testing.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.signal_toolkit.core import Signal


class NoiseType(Enum):
    """Types of noise that can be generated."""

    WHITE = "white"  # Gaussian white noise (uniform spectrum)
    PINK = "pink"  # 1/f noise (more low frequency content)
    BROWN = "brown"  # Brownian/red noise (1/f^2)
    BLUE = "blue"  # Blue noise (f spectrum, more high frequency)
    VIOLET = "violet"  # Violet noise (f^2 spectrum)
    UNIFORM = "uniform"  # Uniform distribution noise
    IMPULSE = "impulse"  # Random impulses/spikes
    QUANTIZATION = "quantization"  # Quantization noise
    PERIODIC = "periodic"  # Periodic disturbance (e.g., line noise)


class NoiseGenerator:
    """Generator for various types of noise signals."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the noise generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        t: np.ndarray,
        noise_type: NoiseType = NoiseType.WHITE,
        amplitude: float = 1.0,
        **kwargs,
    ) -> Signal:
        """Generate a noise signal.

        Design by Contract:
            Preconditions:
                - amplitude >= 0
            Postconditions:
                - output signal has same length as input time array

        Args:
            t: Time array.
            noise_type: Type of noise to generate.
            amplitude: RMS amplitude of the noise.
            **kwargs: Additional parameters for specific noise types.

        Returns:
            Signal containing the noise.
        """
        require(amplitude >= 0, "noise amplitude must be non-negative", amplitude)

        n = len(t)

        if noise_type == NoiseType.WHITE:
            values = self._generate_white_noise(n, amplitude)

        elif noise_type == NoiseType.PINK:
            values = self._generate_pink_noise(n, amplitude)

        elif noise_type == NoiseType.BROWN:
            values = self._generate_brown_noise(n, amplitude)

        elif noise_type == NoiseType.BLUE:
            values = self._generate_blue_noise(n, amplitude)

        elif noise_type == NoiseType.VIOLET:
            values = self._generate_violet_noise(n, amplitude)

        elif noise_type == NoiseType.UNIFORM:
            values = self._generate_uniform_noise(n, amplitude)

        elif noise_type == NoiseType.IMPULSE:
            probability = kwargs.get("probability", 0.01)
            values = self._generate_impulse_noise(n, amplitude, probability)

        elif noise_type == NoiseType.QUANTIZATION:
            levels = kwargs.get("levels", 256)
            values = self._generate_quantization_noise(n, amplitude, levels)

        elif noise_type == NoiseType.PERIODIC:
            frequency = kwargs.get("frequency", 60.0)  # Default 60 Hz (line noise)
            fs = float(1.0 / np.mean(np.diff(t))) if len(t) > 1 else 1000.0
            values = self._generate_periodic_noise(n, amplitude, frequency, fs)

        else:
            values = self._generate_white_noise(n, amplitude)

        result = Signal(
            time=t,
            values=values,
            name=f"{noise_type.value}_noise",
            metadata={"noise_type": noise_type.value, "amplitude": amplitude},
        )

        ensure(
            len(result.values) == len(t),
            "noise signal length must match input time array",
        )

        return result

    def _generate_white_noise(self, n: int, amplitude: float) -> np.ndarray:
        """Generate Gaussian white noise."""
        return self.rng.standard_normal(n) * amplitude

    def _generate_pink_noise(self, n: int, amplitude: float) -> np.ndarray:
        """Generate pink (1/f) noise using the Voss-McCartney algorithm."""
        # Number of random number generators
        num_sources = 16

        # Initialize sources
        values = np.zeros(n)
        max_key = 2**num_sources - 1

        # Running sum
        running_sum = np.zeros(num_sources)

        for i in range(n):
            # Determine which sources to update
            key = i & max_key
            last_key = (i - 1) & max_key if i > 0 else 0

            for j in range(num_sources):
                mask = 1 << j
                if (key & mask) != (last_key & mask):
                    # Update this source
                    running_sum[j] = self.rng.standard_normal()

            values[i] = np.sum(running_sum)

        # Normalize to desired amplitude
        if np.std(values) > 0:
            values = values / np.std(values) * amplitude  # type: ignore[assignment]

        return values

    def _generate_brown_noise(self, n: int, amplitude: float) -> np.ndarray:
        """Generate brown (Brownian) noise - integrated white noise."""
        white = self.rng.standard_normal(n)
        brown = np.cumsum(white)

        # Remove trend and normalize
        brown = brown - np.linspace(brown[0], brown[-1], n)
        if np.std(brown) > 0:
            brown = brown / np.std(brown) * amplitude

        return brown

    def _generate_blue_noise(self, n: int, amplitude: float) -> np.ndarray:
        """Generate blue noise (differentiated white noise)."""
        white = self.rng.standard_normal(n)
        blue = np.diff(white, prepend=white[0])

        if np.std(blue) > 0:
            blue = blue / np.std(blue) * amplitude

        return blue

    def _generate_violet_noise(self, n: int, amplitude: float) -> np.ndarray:
        """Generate violet noise (second derivative of white noise)."""
        white = self.rng.standard_normal(n)
        violet = np.diff(white, n=2, prepend=[white[0], white[0]])

        if np.std(violet) > 0:
            violet = violet / np.std(violet) * amplitude

        return violet

    def _generate_uniform_noise(self, n: int, amplitude: float) -> np.ndarray:
        """Generate uniform distribution noise."""
        # Uniform in [-amplitude*sqrt(3), amplitude*sqrt(3)] to have RMS = amplitude
        half_range = amplitude * np.sqrt(3)
        return self.rng.uniform(-half_range, half_range, n)

    def _generate_impulse_noise(
        self,
        n: int,
        amplitude: float,
        probability: float,
    ) -> np.ndarray:
        """Generate impulse (spike) noise."""
        values = np.zeros(n)
        impulse_mask = self.rng.random(n) < probability
        impulse_signs = self.rng.choice([-1, 1], size=n)
        values[impulse_mask] = impulse_signs[impulse_mask] * amplitude

        return values

    def _generate_quantization_noise(
        self,
        n: int,
        amplitude: float,
        levels: int,
    ) -> np.ndarray:
        """Generate quantization noise (uniform within quantization step)."""
        step = 2 * amplitude / levels
        return self.rng.uniform(-step / 2, step / 2, n)

    def _generate_periodic_noise(
        self,
        n: int,
        amplitude: float,
        frequency: float,
        fs: float,
    ) -> np.ndarray:
        """Generate periodic disturbance (like power line noise)."""
        t = np.arange(n) / fs
        # Add some harmonics for realism
        values = amplitude * np.sin(2 * np.pi * frequency * t)
        values += (
            0.3 * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
        )  # 2nd harmonic
        values += (
            0.1 * amplitude * np.sin(2 * np.pi * 3 * frequency * t)
        )  # 3rd harmonic

        return values


def add_noise_to_signal(
    signal: Signal,
    noise_type: NoiseType = NoiseType.WHITE,
    snr_db: float | None = None,
    amplitude: float | None = None,
    seed: int | None = None,
    **kwargs,
) -> Signal:
    """Add noise to an existing signal.

    Args:
        signal: Input signal.
        noise_type: Type of noise to add.
        snr_db: Signal-to-noise ratio in dB (if specified, overrides amplitude).
        amplitude: Noise amplitude (if snr_db not specified).
        seed: Random seed for reproducibility.
        **kwargs: Additional parameters for specific noise types.

    Returns:
        Signal with noise added.
    """
    generator = NoiseGenerator(seed)

    if snr_db is not None:
        # Calculate amplitude from SNR
        signal_power = np.mean(signal.values**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        amplitude = float(np.sqrt(noise_power))
    elif amplitude is None:
        amplitude = float(0.1 * np.std(signal.values))

    noise = generator.generate(
        signal.time,
        noise_type=noise_type,
        amplitude=amplitude,
        **kwargs,
    )

    noisy_signal = signal.copy()
    noisy_signal.values = signal.values + noise.values
    noisy_signal.name = f"{signal.name}_noisy"
    noisy_signal.metadata["noise_type"] = noise_type.value
    noisy_signal.metadata["noise_amplitude"] = amplitude
    if snr_db is not None:
        noisy_signal.metadata["snr_db"] = snr_db

    return noisy_signal


def generate_disturbance_profile(
    t: np.ndarray,
    disturbance_type: str = "step",
    **kwargs,
) -> Signal:
    """Generate a disturbance signal for simulation.

    Args:
        t: Time array.
        disturbance_type: Type of disturbance:
            - 'step': Step disturbance
            - 'pulse': Rectangular pulse
            - 'ramp': Ramp disturbance
            - 'sine': Sinusoidal disturbance
            - 'random_steps': Random step changes
            - 'chirp': Frequency sweep
        **kwargs: Parameters for the disturbance type.

    Returns:
        Signal containing the disturbance.
    """
    n = len(t)
    values = np.zeros(n)

    if disturbance_type == "step":
        step_time = kwargs.get("step_time", t[n // 2])
        magnitude = kwargs.get("magnitude", 1.0)
        values = np.where(t >= step_time, magnitude, 0.0)  # type: ignore[assignment]

    elif disturbance_type == "pulse":
        start_time = kwargs.get("start_time", t[n // 4])
        duration = kwargs.get("duration", (t[-1] - t[0]) / 10)
        magnitude = kwargs.get("magnitude", 1.0)
        values = np.where(  # type: ignore[assignment]
            (t >= start_time) & (t < start_time + duration),
            magnitude,
            0.0,
        )

    elif disturbance_type == "ramp":
        start_time = kwargs.get("start_time", t[0])
        end_time = kwargs.get("end_time", t[-1])
        start_value = kwargs.get("start_value", 0.0)
        end_value = kwargs.get("end_value", 1.0)

        mask = (t >= start_time) & (t <= end_time)
        t_norm = (t[mask] - start_time) / (end_time - start_time + 1e-10)
        values[mask] = start_value + (end_value - start_value) * t_norm
        values[t > end_time] = end_value

    elif disturbance_type == "sine":
        frequency = kwargs.get("frequency", 1.0)
        amplitude = kwargs.get("amplitude", 1.0)
        phase = kwargs.get("phase", 0.0)
        values = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    elif disturbance_type == "random_steps":
        num_steps = kwargs.get("num_steps", 5)
        max_magnitude = kwargs.get("max_magnitude", 1.0)
        rng = np.random.default_rng(kwargs.get("seed"))

        step_times = np.sort(rng.uniform(t[0], t[-1], num_steps))
        step_values = rng.uniform(-max_magnitude, max_magnitude, num_steps)

        current_value = 0.0
        step_idx = 0
        for i, ti in enumerate(t):
            while step_idx < num_steps and ti >= step_times[step_idx]:
                current_value = step_values[step_idx]
                step_idx += 1
            values[i] = current_value

    elif disturbance_type == "chirp":
        f0 = kwargs.get("f0", 0.1)
        f1 = kwargs.get("f1", 10.0)
        amplitude = kwargs.get("amplitude", 1.0)

        t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-10)
        phase = 2 * np.pi * (f0 * t_norm + 0.5 * (f1 - f0) * t_norm**2)
        values = amplitude * np.sin(phase * (t[-1] - t[0]))

    return Signal(
        time=t,
        values=values,
        name=f"{disturbance_type}_disturbance",
        metadata={"disturbance_type": disturbance_type, **kwargs},
    )


class DisturbanceSimulator:
    """Simulates complex disturbance scenarios.

    Combines multiple disturbance types for realistic simulation.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the simulator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.noise_generator = NoiseGenerator(seed)
        self.disturbances: list[tuple[str, dict]] = []

    def add_noise(
        self,
        noise_type: NoiseType = NoiseType.WHITE,
        amplitude: float = 0.1,
        **kwargs,
    ) -> DisturbanceSimulator:
        """Add a noise component.

        Args:
            noise_type: Type of noise.
            amplitude: Noise amplitude.
            **kwargs: Additional noise parameters.

        Returns:
            Self for method chaining.
        """
        self.disturbances.append(
            ("noise", {"noise_type": noise_type, "amplitude": amplitude, **kwargs})
        )
        return self

    def add_step(
        self,
        step_time: float,
        magnitude: float = 1.0,
    ) -> DisturbanceSimulator:
        """Add a step disturbance.

        Args:
            step_time: Time of step.
            magnitude: Step magnitude.

        Returns:
            Self for method chaining.
        """
        self.disturbances.append(
            (
                "disturbance",
                {"type": "step", "step_time": step_time, "magnitude": magnitude},
            )
        )
        return self

    def add_pulse(
        self,
        start_time: float,
        duration: float,
        magnitude: float = 1.0,
    ) -> DisturbanceSimulator:
        """Add a pulse disturbance.

        Args:
            start_time: Start time of pulse.
            duration: Pulse duration.
            magnitude: Pulse magnitude.

        Returns:
            Self for method chaining.
        """
        self.disturbances.append(
            (
                "disturbance",
                {
                    "type": "pulse",
                    "start_time": start_time,
                    "duration": duration,
                    "magnitude": magnitude,
                },
            )
        )
        return self

    def add_periodic(
        self,
        frequency: float,
        amplitude: float,
    ) -> DisturbanceSimulator:
        """Add a periodic disturbance.

        Args:
            frequency: Frequency in Hz.
            amplitude: Amplitude.

        Returns:
            Self for method chaining.
        """
        self.disturbances.append(
            (
                "disturbance",
                {"type": "sine", "frequency": frequency, "amplitude": amplitude},
            )
        )
        return self

    def generate(self, t: np.ndarray) -> Signal:
        """Generate the combined disturbance signal.

        Args:
            t: Time array.

        Returns:
            Signal with all disturbances combined.
        """
        combined = np.zeros(len(t))

        for dist_type, params in self.disturbances:
            if dist_type == "noise":
                noise = self.noise_generator.generate(t, **params)
                combined += noise.values
            elif dist_type == "disturbance":
                dist_params = {k: v for k, v in params.items() if k != "type"}
                disturb = generate_disturbance_profile(
                    t, disturbance_type=params["type"], **dist_params
                )
                combined += disturb.values

        return Signal(
            time=t,
            values=combined,
            name="combined_disturbance",
            metadata={"components": len(self.disturbances)},
        )

    def apply_to_signal(self, signal: Signal) -> Signal:
        """Apply all disturbances to an existing signal.

        Args:
            signal: Input signal.

        Returns:
            Signal with disturbances applied.
        """
        disturbance = self.generate(signal.time)
        result = signal.copy()
        result.values = signal.values + disturbance.values
        result.name = f"{signal.name}_disturbed"
        return result
