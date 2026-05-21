"""Data models for dataset generation.

Extracted from dataset_generator.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ParameterRange:
    """Defines a range for parameter variation.

    Attributes:
        name: Parameter identifier.
        min_val: Minimum value.
        max_val: Maximum value.
        distribution: Sampling distribution ('uniform', 'normal', 'linspace').
        num_points: Number of discrete points for linspace distribution.
    """

    name: str
    min_val: float
    max_val: float
    distribution: str = "uniform"
    num_points: int = 10

    def __post_init__(self) -> None:
        """Validate parameter range.

        Raises:
            ValueError: If min_val > max_val or distribution is unknown.
        """
        if self.min_val > self.max_val:
            raise ValueError(
                f"Invalid range for '{self.name}': "
                f"min_val ({self.min_val}) > max_val ({self.max_val})"
            )
        valid_distributions = {"uniform", "normal", "linspace"}
        if self.distribution not in valid_distributions:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. "
                f"Valid: {sorted(valid_distributions)}"
            )

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from this range.

        Args:
            rng: NumPy random generator.

        Returns:
            Sampled value within the defined range.
        """
        if rng is None:
            raise ValueError("rng must be provided")
        if self.distribution == "uniform":
            return float(rng.uniform(self.min_val, self.max_val))
        if self.distribution == "normal":
            mean = (self.min_val + self.max_val) / 2.0
            std = (self.max_val - self.min_val) / 6.0  # 99.7% within range
            val = float(rng.normal(mean, std))
            return float(np.clip(val, self.min_val, self.max_val))
        # linspace
        points = np.linspace(self.min_val, self.max_val, self.num_points)
        return float(rng.choice(points))

    def linspace(self) -> np.ndarray:
        """Generate evenly spaced values across the range.

        Returns:
            Array of evenly spaced values.
        """
        return np.linspace(self.min_val, self.max_val, self.num_points)


@dataclass
class ControlProfile:
    """Defines a control input profile for dataset generation.

    Attributes:
        name: Profile identifier.
        profile_type: Type of control profile.
        parameters: Profile-specific parameters.
    """

    name: str
    profile_type: str = "zero"  # zero, constant, sinusoidal, random, step
    parameters: dict[str, Any] = field(default_factory=dict)

    def generate(
        self, n_actuators: int, n_steps: int, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate control input sequence.

        Args:
            n_actuators: Number of actuators/DOFs.
            n_steps: Number of timesteps.
            dt: Timestep size.
            rng: Random generator.

        Returns:
            Control array of shape (n_steps, n_actuators).
        """
        if n_actuators is None:
            raise ValueError("n_actuators must be provided")
        if self.profile_type == "zero":
            return np.zeros((n_steps, n_actuators))
        if self.profile_type == "constant":
            magnitude = self.parameters.get("magnitude", 1.0)
            return np.full((n_steps, n_actuators), magnitude)
        if self.profile_type == "sinusoidal":
            freq = self.parameters.get("frequency", 1.0)
            amplitude = self.parameters.get("amplitude", 1.0)
            t = np.arange(n_steps) * dt
            base = amplitude * np.sin(2.0 * np.pi * freq * t)
            return np.column_stack([base] * n_actuators)
        if self.profile_type == "random":
            scale = self.parameters.get("scale", 1.0)
            return rng.normal(0, scale, (n_steps, n_actuators))
        if self.profile_type == "step":
            magnitude = self.parameters.get("magnitude", 1.0)
            step_time = self.parameters.get("step_time", 0.5)
            step_idx = int(step_time / dt)
            profile = np.zeros((n_steps, n_actuators))
            if step_idx < n_steps:
                profile[step_idx:] = magnitude
            return profile
        return np.zeros((n_steps, n_actuators))


@dataclass
class GeneratorConfig:
    """Configuration for dataset generation.

    Attributes:
        num_samples: Number of simulation runs to generate.
        duration: Duration of each simulation in seconds.
        timestep: Simulation timestep in seconds.
        seed: Random seed for reproducibility.
        vary_initial_positions: Whether to randomize initial joint positions.
        vary_initial_velocities: Whether to randomize initial joint velocities.
        position_ranges: Ranges for initial position variation.
        velocity_ranges: Ranges for initial velocity variation.
        control_profiles: Control profiles to sample from.
        record_mass_matrix: Whether to record inertia matrices.
        record_bias_forces: Whether to record bias forces.
        record_gravity: Whether to record gravity forces.
        record_jacobians: Whether to record Jacobians.
        record_contact_forces: Whether to record contact forces.
        record_drift_control: Whether to record drift/control decomposition.
        record_counterfactuals: Whether to record ZTCF/ZVCF.
        output_fields: Explicit list of fields to record (None = all).
    """

    num_samples: int = 100
    duration: float = 2.0
    timestep: float = 0.002
    seed: int = 42
    vary_initial_positions: bool = True
    vary_initial_velocities: bool = False
    position_ranges: list[ParameterRange] = field(default_factory=list)
    velocity_ranges: list[ParameterRange] = field(default_factory=list)
    control_profiles: list[ControlProfile] = field(
        default_factory=lambda: [
            ControlProfile(name="zero"),
        ]
    )
    record_mass_matrix: bool = True
    record_bias_forces: bool = True
    record_gravity: bool = True
    record_jacobians: bool = False
    record_contact_forces: bool = True
    record_drift_control: bool = True
    record_counterfactuals: bool = False
    output_fields: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.timestep <= 0:
            raise ValueError(f"timestep must be positive, got {self.timestep}")
        if self.duration < self.timestep:
            raise ValueError(
                f"duration ({self.duration}) must be >= timestep ({self.timestep}); "
                "otherwise no steps would be recorded"
            )


@dataclass
class SimulationSample:
    """A single simulation run's recorded data.

    Attributes:
        sample_id: Unique sample identifier.
        metadata: Configuration and provenance metadata.
        times: Time array (n_steps,).
        positions: Joint positions (n_steps, n_q).
        velocities: Joint velocities (n_steps, n_v).
        accelerations: Joint accelerations (n_steps, n_v).
        torques: Applied joint torques (n_steps, n_v).
        mass_matrices: Mass matrices per step (n_steps, n_v, n_v) or None.
        bias_forces: Bias forces per step (n_steps, n_v) or None.
        gravity_forces: Gravity forces per step (n_steps, n_v) or None.
        contact_forces: Contact forces per step (n_steps, 3) or None.
        drift_accelerations: Drift accelerations (n_steps, n_v) or None.
        control_accelerations: Control accelerations (n_steps, n_v) or None.
        energies: Energy data dict.
    """

    sample_id: int
    metadata: dict[str, Any]
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    torques: np.ndarray
    mass_matrices: np.ndarray | None = None
    bias_forces: np.ndarray | None = None
    gravity_forces: np.ndarray | None = None
    contact_forces: np.ndarray | None = None
    drift_accelerations: np.ndarray | None = None
    control_accelerations: np.ndarray | None = None
    energies: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class TrainingDataset:
    """Collection of simulation samples forming a training dataset.

    Attributes:
        samples: List of simulation samples.
        config: Generator configuration used.
        model_name: Name of the model used.
        engine_name: Name of the physics engine used.
        joint_names: Names of joints in the model.
        creation_time: Unix timestamp of dataset creation.
    """

    samples: list[SimulationSample]
    config: GeneratorConfig
    model_name: str
    engine_name: str
    joint_names: list[str]
    creation_time: float = field(default_factory=time.time)

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.samples)

    @property
    def total_frames(self) -> int:
        """Total number of frames across all samples."""
        return sum(len(s.times) for s in self.samples)


__all__ = [
    "ControlProfile",
    "GeneratorConfig",
    "ParameterRange",
    "SimulationSample",
    "TrainingDataset",
]
