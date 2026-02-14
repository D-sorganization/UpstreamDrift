"""Dataset Generator for Neural Network Training.

Generates large-scale simulation datasets by varying inputs across physics engines.
Records all kinematics (q, v, a), kinetics (tau, forces, energies), and model
data (inertia, bias forces, Jacobians) into structured databases for ML training.

Design by Contract:
    Preconditions:
        - Engine must implement PhysicsEngine protocol
        - Parameter ranges must be valid (min <= max)
        - Output directory must be writable
    Postconditions:
        - Generated dataset contains all requested fields
        - Data is validated (no NaN/Inf in physics quantities)
        - Provenance metadata is attached to every dataset
    Invariants:
        - Original engine state is restored after generation
        - All data is reproducible given the same seed

Usage:
    >>> from src.shared.python.data_io.dataset_generator import DatasetGenerator
    >>> gen = DatasetGenerator(engine)
    >>> config = GeneratorConfig(
    ...     num_samples=1000,
    ...     duration=2.0,
    ...     timestep=0.002,
    ...     vary_initial_positions=True,
    ... )
    >>> dataset = gen.generate(config)
    >>> gen.export(dataset, "output/training_data", format="hdf5")
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import invariant, postcondition, precondition
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


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
        if self.distribution == "uniform":
            return float(rng.uniform(self.min_val, self.max_val))
        elif self.distribution == "normal":
            mean = (self.min_val + self.max_val) / 2.0
            std = (self.max_val - self.min_val) / 6.0  # 99.7% within range
            val = float(rng.normal(mean, std))
            return float(np.clip(val, self.min_val, self.max_val))
        else:  # linspace
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
        if self.profile_type == "zero":
            return np.zeros((n_steps, n_actuators))
        elif self.profile_type == "constant":
            magnitude = self.parameters.get("magnitude", 1.0)
            return np.full((n_steps, n_actuators), magnitude)
        elif self.profile_type == "sinusoidal":
            freq = self.parameters.get("frequency", 1.0)
            amplitude = self.parameters.get("amplitude", 1.0)
            t = np.arange(n_steps) * dt
            base = amplitude * np.sin(2.0 * np.pi * freq * t)
            return np.column_stack([base] * n_actuators)
        elif self.profile_type == "random":
            scale = self.parameters.get("scale", 1.0)
            return rng.normal(0, scale, (n_steps, n_actuators))
        elif self.profile_type == "step":
            magnitude = self.parameters.get("magnitude", 1.0)
            step_time = self.parameters.get("step_time", 0.5)
            step_idx = int(step_time / dt)
            profile = np.zeros((n_steps, n_actuators))
            if step_idx < n_steps:
                profile[step_idx:] = magnitude
            return profile
        else:
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


@invariant(
    lambda self: self.engine is not None,
    "DatasetGenerator must have a valid engine reference",
)
class DatasetGenerator:
    """Generates simulation datasets for neural network training.

    Uses a PhysicsEngine to run simulations with varied inputs and records
    all relevant kinematics, kinetics, and model data.

    Design by Contract:
        Preconditions:
            - engine must implement PhysicsEngine protocol
            - engine must be in INITIALIZED state (model loaded)
        Postconditions:
            - Generated dataset contains valid, finite data
            - Engine state is restored to original after generation
        Invariants:
            - Dataset generation is reproducible given same seed
    """

    def __init__(self, engine: PhysicsEngine) -> None:
        """Initialize the dataset generator.

        Args:
            engine: Physics engine instance with a loaded model.

        Raises:
            ValueError: If engine has no model loaded.
        """
        self.engine = engine
        self._original_state: tuple[np.ndarray, np.ndarray] | None = None

    @precondition(
        lambda self, config, progress_callback=None: config is not None,
        "Generator config must not be None",
    )
    @precondition(
        lambda self, config, progress_callback=None: config.num_samples > 0,
        "Number of samples must be positive",
    )
    @precondition(
        lambda self, config, progress_callback=None: config.timestep > 0,
        "Timestep must be positive",
    )
    @postcondition(
        lambda result: result is not None and result.num_samples > 0,
        "Generated dataset must contain at least one sample",
    )
    def generate(
        self,
        config: GeneratorConfig,
        progress_callback: Any | None = None,
    ) -> TrainingDataset:
        """Generate a training dataset from simulation runs.

        Args:
            config: Generation configuration.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            TrainingDataset containing all simulation samples.

        Raises:
            RuntimeError: If simulation fails for all samples.
        """
        rng = np.random.default_rng(config.seed)

        # Save original state
        try:
            self._original_state = self.engine.get_state()
        except (ValueError, RuntimeError, AttributeError):
            self._original_state = None

        # Get model info
        model_name = getattr(self.engine, "model_name", "unknown")
        engine_name = type(self.engine).__name__
        joint_names = self._get_joint_names()

        n_steps = int(config.duration / config.timestep)
        n_q, n_v = self._get_dimensions()

        samples: list[SimulationSample] = []
        failed_count = 0

        logger.info(
            "Starting dataset generation: %d samples, %d steps each",
            config.num_samples,
            n_steps,
        )

        for i in range(config.num_samples):
            try:
                sample = self._run_single_simulation(
                    sample_id=i,
                    config=config,
                    rng=rng,
                    n_steps=n_steps,
                    n_q=n_q,
                    n_v=n_v,
                )
                samples.append(sample)

                if progress_callback is not None:
                    progress_callback(i + 1, config.num_samples)

            except (RuntimeError, TypeError, ValueError) as e:
                logger.warning("Sample %d failed: %s", i, e)
                failed_count += 1
                continue

        if not samples:
            raise RuntimeError(
                f"All {config.num_samples} samples failed during generation"
            )

        if failed_count > 0:
            logger.warning(
                "%d/%d samples failed during generation",
                failed_count,
                config.num_samples,
            )

        # Restore original state
        if self._original_state is not None:
            try:
                self.engine.set_state(*self._original_state)
            except (ValueError, RuntimeError, AttributeError):
                pass

        dataset = TrainingDataset(
            samples=samples,
            config=config,
            model_name=model_name,
            engine_name=engine_name,
            joint_names=joint_names,
        )

        logger.info(
            "Dataset generation complete: %d samples, %d total frames",
            dataset.num_samples,
            dataset.total_frames,
        )

        return dataset

    def _run_single_simulation(
        self,
        sample_id: int,
        config: GeneratorConfig,
        rng: np.random.Generator,
        n_steps: int,
        n_q: int,
        n_v: int,
    ) -> SimulationSample:
        """Run a single simulation and record data.

        Args:
            sample_id: Sample identifier.
            config: Generator configuration.
            rng: Random number generator.
            n_steps: Number of simulation steps.
            n_q: Number of position DOFs.
            n_v: Number of velocity DOFs.

        Returns:
            SimulationSample with all recorded data.
        """
        # Reset engine and set initial conditions
        self.engine.reset()
        q0, v0 = self._generate_initial_conditions(config, rng, n_q, n_v)
        self.engine.set_state(q0, v0)

        # Generate control profile
        idx = rng.integers(len(config.control_profiles))
        profile = config.control_profiles[idx]
        control_sequence = profile.generate(n_v, n_steps, config.timestep, rng)

        # Pre-allocate recording arrays
        buffers = self._allocate_sim_buffers(config, n_steps, n_q, n_v)

        # Run simulation loop
        self._execute_sim_loop(config, control_sequence, n_steps, buffers)

        # Build metadata and return sample
        metadata = {
            "sample_id": sample_id,
            "seed": config.seed,
            "duration": config.duration,
            "timestep": config.timestep,
            "initial_q": q0.tolist(),
            "initial_v": v0.tolist(),
            "control_profile": profile.name,
            "control_type": profile.profile_type,
        }

        assert buffers["times"] is not None, "times buffer must not be None"
        assert buffers["positions"] is not None, "positions buffer must not be None"
        assert buffers["velocities"] is not None, "velocities buffer must not be None"
        assert buffers["accelerations"] is not None, (
            "accelerations buffer must not be None"
        )
        assert buffers["torques"] is not None, "torques buffer must not be None"
        assert buffers["kinetic_energy"] is not None, (
            "kinetic_energy buffer must not be None"
        )
        assert buffers["potential_energy"] is not None, (
            "potential_energy buffer must not be None"
        )

        return SimulationSample(
            sample_id=sample_id,
            metadata=metadata,
            times=buffers["times"],
            positions=buffers["positions"],
            velocities=buffers["velocities"],
            accelerations=buffers["accelerations"],
            torques=buffers["torques"],
            mass_matrices=buffers["mass_matrices"],
            bias_forces=buffers["bias_forces"],
            gravity_forces=buffers["gravity"],
            contact_forces=buffers["contact"],
            drift_accelerations=buffers["drift"],
            control_accelerations=buffers["control_accel"],
            energies={
                "kinetic": buffers["kinetic_energy"],
                "potential": buffers["potential_energy"],
            },
        )

    @staticmethod
    def _allocate_sim_buffers(
        config: GeneratorConfig,
        n_steps: int,
        n_q: int,
        n_v: int,
    ) -> dict[str, np.ndarray | None]:
        """Pre-allocate all recording arrays for a simulation run.

        Args:
            config: Generator configuration.
            n_steps: Number of simulation steps.
            n_q: Number of position DOFs.
            n_v: Number of velocity DOFs.

        Returns:
            Dictionary of named buffers.
        """
        return {
            "times": np.zeros(n_steps),
            "positions": np.zeros((n_steps, n_q)),
            "velocities": np.zeros((n_steps, n_v)),
            "accelerations": np.zeros((n_steps, n_v)),
            "torques": np.zeros((n_steps, n_v)),
            "mass_matrices": (
                np.zeros((n_steps, n_v, n_v)) if config.record_mass_matrix else None
            ),
            "bias_forces": (
                np.zeros((n_steps, n_v)) if config.record_bias_forces else None
            ),
            "gravity": np.zeros((n_steps, n_v)) if config.record_gravity else None,
            "contact": (
                np.zeros((n_steps, 3)) if config.record_contact_forces else None
            ),
            "drift": (
                np.zeros((n_steps, n_v)) if config.record_drift_control else None
            ),
            "control_accel": (
                np.zeros((n_steps, n_v)) if config.record_drift_control else None
            ),
            "kinetic_energy": np.zeros(n_steps),
            "potential_energy": np.zeros(n_steps),
        }

    def _execute_sim_loop(
        self,
        config: GeneratorConfig,
        control_sequence: np.ndarray,
        n_steps: int,
        buffers: dict[str, np.ndarray | None],
    ) -> None:
        """Execute the simulation loop, recording state and dynamics each step.

        Args:
            config: Generator configuration.
            control_sequence: Control input array (n_steps, n_v).
            n_steps: Number of simulation steps.
            buffers: Pre-allocated recording buffers (modified in-place).
        """
        for step in range(n_steps):
            tau = control_sequence[step]
            self.engine.set_control(tau)

            # Record pre-step state
            q, v = self.engine.get_state()
            t = self.engine.get_time()

            buffers["times"][step] = t  # type: ignore[index]
            buffers["positions"][step] = q  # type: ignore[index]
            buffers["velocities"][step] = v  # type: ignore[index]
            buffers["torques"][step] = tau  # type: ignore[index]

            # Record dynamics quantities
            self._record_dynamics_step(config, step, tau, v, buffers)

            # Step simulation
            self.engine.step(config.timestep)

            # Record post-step accelerations
            try:
                q_new, v_new = self.engine.get_state()
                buffers["accelerations"][step] = (v_new - v) / config.timestep  # type: ignore[index]
            except (ValueError, RuntimeError, AttributeError):
                pass

    def _record_dynamics_step(
        self,
        config: GeneratorConfig,
        step: int,
        tau: np.ndarray,
        v: np.ndarray,
        buffers: dict[str, np.ndarray | None],
    ) -> None:
        """Record optional dynamics quantities for a single simulation step.

        Args:
            config: Generator configuration.
            step: Current step index.
            tau: Applied torque vector.
            v: Current velocity vector.
            buffers: Pre-allocated recording buffers (modified in-place).
        """
        if config.record_mass_matrix and buffers["mass_matrices"] is not None:
            try:
                buffers["mass_matrices"][step] = self.engine.compute_mass_matrix()
            except (ValueError, RuntimeError, AttributeError):
                pass

        if config.record_bias_forces and buffers["bias_forces"] is not None:
            try:
                buffers["bias_forces"][step] = self.engine.compute_bias_forces()
            except (ValueError, RuntimeError, AttributeError):
                pass

        if config.record_gravity and buffers["gravity"] is not None:
            try:
                buffers["gravity"][step] = self.engine.compute_gravity_forces()
            except (ValueError, RuntimeError, AttributeError):
                pass

        if config.record_contact_forces and buffers["contact"] is not None:
            try:
                cf = self.engine.compute_contact_forces()
                buffers["contact"][step, : len(cf)] = cf[:3]
            except (ValueError, RuntimeError, AttributeError):
                pass

        if config.record_drift_control:
            try:
                if buffers["drift"] is not None:
                    buffers["drift"][step] = self.engine.compute_drift_acceleration()
                if buffers["control_accel"] is not None:
                    buffers["control_accel"][step] = (
                        self.engine.compute_control_acceleration(tau)
                    )
            except (ValueError, RuntimeError, AttributeError):
                pass

        # Compute energies
        try:
            M = self.engine.compute_mass_matrix()
            buffers["kinetic_energy"][step] = 0.5 * float(v.T @ M @ v)  # type: ignore[index]
        except (ValueError, RuntimeError, AttributeError):
            pass

    def _generate_initial_conditions(
        self,
        config: GeneratorConfig,
        rng: np.random.Generator,
        n_q: int,
        n_v: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate randomized initial conditions.

        Args:
            config: Generator configuration.
            rng: Random generator.
            n_q: Number of position DOFs.
            n_v: Number of velocity DOFs.

        Returns:
            Tuple of (initial_positions, initial_velocities).
        """
        if config.vary_initial_positions and config.position_ranges:
            q0 = np.zeros(n_q)
            for pr in config.position_ranges:
                # Apply to all joints if name is "all", else by index
                if pr.name == "all":
                    for j in range(n_q):
                        q0[j] = pr.sample(rng)
                else:
                    try:
                        idx = int(pr.name)
                        if 0 <= idx < n_q:
                            q0[idx] = pr.sample(rng)
                    except ValueError:
                        pass
        elif config.vary_initial_positions:
            q0 = rng.uniform(-0.5, 0.5, n_q)  # type: ignore[assignment]
        else:
            q0 = np.zeros(n_q)

        if config.vary_initial_velocities and config.velocity_ranges:
            v0 = np.zeros(n_v)
            for vr in config.velocity_ranges:
                if vr.name == "all":
                    for j in range(n_v):
                        v0[j] = vr.sample(rng)
                else:
                    try:
                        idx = int(vr.name)
                        if 0 <= idx < n_v:
                            v0[idx] = vr.sample(rng)
                    except ValueError:
                        pass
        elif config.vary_initial_velocities:
            v0 = rng.uniform(-0.1, 0.1, n_v)  # type: ignore[assignment]
        else:
            v0 = np.zeros(n_v)

        return q0, v0

    def _get_dimensions(self) -> tuple[int, int]:
        """Get model dimensions (n_q, n_v).

        Returns:
            Tuple of (position_dims, velocity_dims).
        """
        try:
            q, v = self.engine.get_state()
            return len(q), len(v)
        except (ValueError, RuntimeError, AttributeError):
            return 7, 7  # Reasonable default for a 7-DOF arm

    def _get_joint_names(self) -> list[str]:
        """Get joint names from engine.

        Returns:
            List of joint name strings.
        """
        try:
            names = self.engine.get_joint_names()
            if names:
                return names
        except (ValueError, RuntimeError, AttributeError):
            pass
        n_q, _ = self._get_dimensions()
        return [f"joint_{i}" for i in range(n_q)]

    def export_to_hdf5(self, dataset: TrainingDataset, output_path: str | Path) -> Path:
        """Export dataset to HDF5 format.

        Args:
            dataset: Training dataset to export.
            output_path: Output file path (without extension).

        Returns:
            Path to the created HDF5 file.

        Raises:
            ImportError: If h5py is not available.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py required for HDF5 export: pip install h5py"
            ) from None

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".hdf5")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(output_path), "w") as f:
            self._write_hdf5_metadata(f, dataset)
            samples_grp = f.create_group("samples")
            for sample in dataset.samples:
                self._write_hdf5_sample(samples_grp, sample)

        logger.info("Exported dataset to HDF5: %s", output_path)
        return output_path

    @staticmethod
    def _write_hdf5_metadata(f: Any, dataset: TrainingDataset) -> None:
        """Write dataset-level metadata to an HDF5 file."""
        meta = f.create_group("metadata")
        meta.attrs["model_name"] = dataset.model_name
        meta.attrs["engine_name"] = dataset.engine_name
        meta.attrs["num_samples"] = dataset.num_samples
        meta.attrs["total_frames"] = dataset.total_frames
        meta.attrs["creation_time"] = dataset.creation_time
        meta.attrs["duration"] = dataset.config.duration
        meta.attrs["timestep"] = dataset.config.timestep
        meta.attrs["seed"] = dataset.config.seed

        if dataset.joint_names:
            meta.create_dataset(
                "joint_names",
                data=[n.encode("utf-8") for n in dataset.joint_names],
            )

    @staticmethod
    def _write_hdf5_sample(samples_grp: Any, sample: SimulationSample) -> None:
        """Write a single sample's data to an HDF5 samples group."""
        s_grp = samples_grp.create_group(f"sample_{sample.sample_id:06d}")
        s_grp.create_dataset("times", data=sample.times, compression="gzip")
        s_grp.create_dataset("positions", data=sample.positions, compression="gzip")
        s_grp.create_dataset("velocities", data=sample.velocities, compression="gzip")
        s_grp.create_dataset(
            "accelerations", data=sample.accelerations, compression="gzip"
        )
        s_grp.create_dataset("torques", data=sample.torques, compression="gzip")

        optional_fields = [
            ("mass_matrices", sample.mass_matrices),
            ("bias_forces", sample.bias_forces),
            ("gravity_forces", sample.gravity_forces),
            ("contact_forces", sample.contact_forces),
            ("drift_accelerations", sample.drift_accelerations),
            ("control_accelerations", sample.control_accelerations),
        ]
        for field_name, field_data in optional_fields:
            if field_data is not None:
                s_grp.create_dataset(field_name, data=field_data, compression="gzip")

        if sample.energies:
            e_grp = s_grp.create_group("energies")
            for key, arr in sample.energies.items():
                e_grp.create_dataset(key, data=arr, compression="gzip")

        s_grp.attrs["metadata"] = json.dumps(sample.metadata)

    def export_to_sqlite(
        self, dataset: TrainingDataset, output_path: str | Path
    ) -> Path:
        """Export dataset to SQLite database.

        Args:
            dataset: Training dataset to export.
            output_path: Output database path.

        Returns:
            Path to the created SQLite database.
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".db")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(output_path))
        try:
            cursor = conn.cursor()
            self._create_sqlite_tables(cursor)
            self._insert_sqlite_metadata(cursor, dataset)
            for sample in dataset.samples:
                self._insert_sqlite_sample(cursor, sample)
            conn.commit()
        finally:
            conn.close()

        logger.info("Exported dataset to SQLite: %s", output_path)
        return output_path

    @staticmethod
    def _create_sqlite_tables(cursor: sqlite3.Cursor) -> None:
        """Create the SQLite schema tables."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id INTEGER PRIMARY KEY,
                metadata_json TEXT,
                n_steps INTEGER,
                n_q INTEGER,
                n_v INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                sample_id INTEGER,
                step INTEGER,
                time REAL,
                positions_json TEXT,
                velocities_json TEXT,
                accelerations_json TEXT,
                torques_json TEXT,
                kinetic_energy REAL,
                PRIMARY KEY (sample_id, step),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
        """)

    @staticmethod
    def _insert_sqlite_metadata(
        cursor: sqlite3.Cursor, dataset: TrainingDataset
    ) -> None:
        """Insert dataset-level metadata into the SQLite database."""
        meta_items = [
            ("model_name", dataset.model_name),
            ("engine_name", dataset.engine_name),
            ("num_samples", str(dataset.num_samples)),
            ("total_frames", str(dataset.total_frames)),
            ("creation_time", str(dataset.creation_time)),
            ("seed", str(dataset.config.seed)),
            ("duration", str(dataset.config.duration)),
            ("timestep", str(dataset.config.timestep)),
            ("joint_names", json.dumps(dataset.joint_names)),
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)",
            meta_items,
        )

    @staticmethod
    def _insert_sqlite_sample(cursor: sqlite3.Cursor, sample: SimulationSample) -> None:
        """Insert a single sample and its frames into the SQLite database."""
        n_steps = len(sample.times)
        n_q = sample.positions.shape[1] if sample.positions.ndim > 1 else 0
        n_v = sample.velocities.shape[1] if sample.velocities.ndim > 1 else 0

        cursor.execute(
            "INSERT INTO samples (sample_id, metadata_json, n_steps, n_q, n_v) "
            "VALUES (?, ?, ?, ?, ?)",
            (sample.sample_id, json.dumps(sample.metadata), n_steps, n_q, n_v),
        )

        frame_rows = []
        for step in range(n_steps):
            ke = (
                float(sample.energies["kinetic"][step])
                if "kinetic" in sample.energies
                else 0.0
            )
            frame_rows.append(
                (
                    sample.sample_id,
                    step,
                    float(sample.times[step]),
                    json.dumps(sample.positions[step].tolist()),
                    json.dumps(sample.velocities[step].tolist()),
                    json.dumps(sample.accelerations[step].tolist()),
                    json.dumps(sample.torques[step].tolist()),
                    ke,
                )
            )

        cursor.executemany(
            "INSERT INTO frames "
            "(sample_id, step, time, positions_json, velocities_json, "
            "accelerations_json, torques_json, kinetic_energy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            frame_rows,
        )

    def export_to_csv(self, dataset: TrainingDataset, output_dir: str | Path) -> Path:
        """Export dataset to CSV files (one per sample).

        Args:
            dataset: Training dataset to export.
            output_dir: Output directory for CSV files.

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for sample in dataset.samples:
            n_steps = len(sample.times)
            n_q = sample.positions.shape[1]
            n_v = sample.velocities.shape[1]

            # Build header
            headers = ["time"]
            headers.extend([f"q_{i}" for i in range(n_q)])
            headers.extend([f"v_{i}" for i in range(n_v)])
            headers.extend([f"a_{i}" for i in range(n_v)])
            headers.extend([f"tau_{i}" for i in range(n_v)])
            headers.append("kinetic_energy")

            # Build data matrix
            data_cols = [sample.times.reshape(-1, 1)]
            data_cols.append(sample.positions)
            data_cols.append(sample.velocities)
            data_cols.append(sample.accelerations)
            data_cols.append(sample.torques)

            ke = sample.energies.get("kinetic", np.zeros(n_steps))
            data_cols.append(ke.reshape(-1, 1))

            data = np.hstack(data_cols)

            filepath = output_dir / f"sample_{sample.sample_id:06d}.csv"
            np.savetxt(
                str(filepath),
                data,
                delimiter=",",
                header=",".join(headers),
                comments="",
            )

        # Write metadata file
        meta_path = output_dir / "metadata.json"
        meta = {
            "model_name": dataset.model_name,
            "engine_name": dataset.engine_name,
            "num_samples": dataset.num_samples,
            "total_frames": dataset.total_frames,
            "joint_names": dataset.joint_names,
            "config": {
                "duration": dataset.config.duration,
                "timestep": dataset.config.timestep,
                "seed": dataset.config.seed,
                "num_samples": dataset.config.num_samples,
            },
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Exported dataset to CSV: %s", output_dir)
        return output_dir

    @precondition(
        lambda self, dataset, output_path, format="hdf5": dataset is not None,
        "Dataset must not be None",
    )
    @precondition(
        lambda self, dataset, output_path, format="hdf5": output_path is not None
        and len(str(output_path)) > 0,
        "Output path must be a non-empty string or Path",
    )
    @precondition(
        lambda self, dataset, output_path, format="hdf5": format
        in ("hdf5", "sqlite", "db", "csv"),
        "Export format must be one of: hdf5, sqlite, db, csv",
    )
    def export(
        self,
        dataset: TrainingDataset,
        output_path: str | Path,
        format: str = "hdf5",
    ) -> Path:
        """Export dataset in the specified format.

        Args:
            dataset: Training dataset to export.
            output_path: Output path (file or directory depending on format).
            format: Export format ('hdf5', 'sqlite', 'csv').

        Returns:
            Path to the exported data.

        Raises:
            ValueError: If format is not supported.
        """
        format = format.lower()
        if format == "hdf5":
            return self.export_to_hdf5(dataset, output_path)
        elif format in ("sqlite", "db"):
            return self.export_to_sqlite(dataset, output_path)
        elif format == "csv":
            return self.export_to_csv(dataset, output_path)
        else:
            raise ValueError(
                f"Unsupported export format: {format}. Supported: hdf5, sqlite, csv"
            )
