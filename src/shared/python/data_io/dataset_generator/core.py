"""Dataset Generator for Neural Network Training.

Generates large-scale simulation datasets by varying inputs across physics engines.
Records all kinematics (q, v, a), kinetics (tau, forces, energies), and model
data (inertia, bias forces, Jacobians) into structured databases for ML training.

Data models are in _dataset_models.py.
Export methods are in _dataset_export_mixin.py.

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

import contextlib
from typing import Any

import numpy as np

from src.shared.python.core.contracts import invariant, postcondition, precondition
from src.shared.python.core.error_utils import SimulationError
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

from .._dataset_export_mixin import _DatasetExportMixin
from .config import (
    ControlProfile,
    GeneratorConfig,
    ParameterRange,
)
from .models import (
    SimulationSample,
    TrainingDataset,
)

logger = get_logger(__name__)


@invariant(
    lambda self: self.engine is not None,
    "DatasetGenerator must have a valid engine reference",
)
class DatasetGenerator(_DatasetExportMixin):
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

        if engine is None:
            raise ValueError("engine must be provided")
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

        if config is None:
            raise ValueError("config must be provided")
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

        try:
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
                raise SimulationError(
                    f"All {config.num_samples} samples failed during generation"
                )

            if failed_count > 0:
                logger.warning(
                    "%d/%d samples failed during generation",
                    failed_count,
                    config.num_samples,
                )

        finally:
            # Restore original state regardless of success or failure
            if self._original_state is not None:
                with contextlib.suppress(ValueError, RuntimeError, AttributeError):
                    self.engine.set_state(*self._original_state)

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

        if not (buffers["times"] is not None):
            raise ValueError("times buffer must not be None")
        if not (buffers["positions"] is not None):
            raise ValueError("positions buffer must not be None")
        if not (buffers["velocities"] is not None):
            raise ValueError("velocities buffer must not be None")
        if not (buffers["accelerations"] is not None):
            raise ValueError("accelerations buffer must not be None")
        if not (buffers["torques"] is not None):
            raise ValueError("torques buffer must not be None")
        if not (buffers["kinetic_energy"] is not None):
            raise ValueError("kinetic_energy buffer must not be None")
        if not (buffers["potential_energy"] is not None):
            raise ValueError("potential_energy buffer must not be None")

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

    def _record_dynamics_step(  # noqa: C901
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

        if config is None:
            raise ValueError("config must be provided")
        if config.record_mass_matrix and buffers["mass_matrices"] is not None:
            with contextlib.suppress(ValueError, RuntimeError, AttributeError):
                buffers["mass_matrices"][step] = self.engine.compute_mass_matrix()

        if config.record_bias_forces and buffers["bias_forces"] is not None:
            with contextlib.suppress(ValueError, RuntimeError, AttributeError):
                buffers["bias_forces"][step] = self.engine.compute_bias_forces()

        if config.record_gravity and buffers["gravity"] is not None:
            with contextlib.suppress(ValueError, RuntimeError, AttributeError):
                buffers["gravity"][step] = self.engine.compute_gravity_forces()

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
        with contextlib.suppress(ValueError, RuntimeError, AttributeError):
            buffers["potential_energy"][step] = float(  # type: ignore[index]
                self.engine.compute_potential_energy()  # type: ignore[attr-defined]
            )

    def _generate_initial_conditions(  # noqa: C901
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

        if config is None:
            raise ValueError("config must be provided")
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
                        logger.debug(
                            "Skipping position range %r: not a valid joint index",
                            pr.name,
                        )
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
                        logger.debug(
                            "Skipping velocity range %r: not a valid joint index",
                            vr.name,
                        )
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


__all__ = [
    "ControlProfile",
    "DatasetGenerator",
    "GeneratorConfig",
    "ParameterRange",
    "SimulationSample",
    "TrainingDataset",
]
