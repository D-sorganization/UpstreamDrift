import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import GeneratorConfig


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
