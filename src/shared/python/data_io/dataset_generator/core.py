"""Dataset Generator for Neural Network Training.

Generates large-scale simulation datasets by varying inputs across physics engines.
Records all kinematics (q, v, a), kinetics (tau, forces, energies), and model
data (inertia, bias forces, Jacobians) into structured databases for ML training.

NM-02 (#10617): optional channels carry availability evidence; suppressed
dynamics failures cannot leave a zero buffer presented as measurement.
Instantaneous native acceleration is recorded separately from interval
finite differences. Applied controls are distinguished from requested
commands when saturation intervenes.

Data models are in models.py / labels.py.
Sim-step recording is in sim_recording.py.
Channel finalize helpers are in channel_finalize.py.
Export methods are in _dataset_export_mixin.py.

Design by Contract:
    Preconditions:
        - Engine must implement PhysicsEngine protocol
        - Parameter ranges must be valid (min <= max)
        - Output directory must be writable
    Postconditions:
        - Generated dataset contains all requested fields with evidence
        - Data is validated (no NaN/Inf in available physics quantities)
        - Provenance metadata is attached to every dataset
    Invariants:
        - Original engine state/time is restored after generation or
          StateError is raised when restore fails
        - All data is reproducible given the same seed
"""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Any

import numpy as np

from src.shared.python.core.contracts import invariant, postcondition, precondition
from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.core.error_utils import SimulationError
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

from .._dataset_export_mixin import _DatasetExportMixin
from .channel_finalize import finalize_channels
from .config import (
    ControlProfile,
    GeneratorConfig,
    ParameterRange,
)
from .labels import (
    LABEL_SCHEMA,
    AccelerationKind,
    ActuationKind,
    ModelDoFLayout,
    SampleProvenance,
)
from .models import (
    SimulationSample,
    TrainingDataset,
)
from .sim_recording import _OPTIONAL_CATCH, _SimRecordingMixin

logger = get_logger(__name__)


@invariant(
    lambda self: self.engine is not None,
    "DatasetGenerator must have a valid engine reference",
)
class DatasetGenerator(_SimRecordingMixin, _DatasetExportMixin):
    """Generates simulation datasets for neural network training.

    Uses a PhysicsEngine to run simulations with varied inputs and records
    all relevant kinematics, kinetics, and model data with channel evidence.
    """

    def __init__(self, engine: PhysicsEngine) -> None:
        """Initialize the dataset generator.

        Args:
            engine: Physics engine instance with a loaded model.

        Raises:
            ValueError: If engine is None.
        """
        if engine is None:
            raise ValueError("engine must be provided")
        self.engine = engine
        self._original_state: tuple[np.ndarray, np.ndarray] | None = None
        self._original_time: float | None = None

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
            SimulationError: If simulation fails for all samples.
            StateError: If original engine state/time cannot be restored.
        """
        if config is None:
            raise ValueError("config must be provided")
        rng = np.random.default_rng(config.seed)

        self._capture_original_state()

        model_name = getattr(self.engine, "model_name", "unknown")
        engine_name = type(self.engine).__name__
        joint_names = self._get_joint_names()
        layout = self._get_layout()

        n_steps = int(config.duration / config.timestep)
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
                        layout=layout,
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
            self._restore_original_state()

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

    def _capture_original_state(self) -> None:
        try:
            self._original_state = self.engine.get_state()
        except _OPTIONAL_CATCH:
            self._original_state = None
        try:
            self._original_time = float(self.engine.get_time())
        except _OPTIONAL_CATCH:
            self._original_time = None

    def _restore_original_state(self) -> None:
        """Restore engine state and time, or raise StateError."""
        if self._original_state is None and self._original_time is None:
            return
        errors: list[str] = []
        if self._original_state is not None:
            try:
                self.engine.set_state(*self._original_state)
            except _OPTIONAL_CATCH as exc:
                errors.append(f"state restore failed: {exc}")
        if self._original_time is not None:
            set_time = getattr(self.engine, "set_time", None)
            if callable(set_time):
                try:
                    set_time(self._original_time)
                except _OPTIONAL_CATCH as exc:
                    errors.append(f"time restore failed: {exc}")
            else:
                try:
                    current = float(self.engine.get_time())
                    if abs(current - self._original_time) > 1e-9:
                        errors.append(
                            "time restore unsupported: engine has no set_time "
                            f"(current={current}, expected={self._original_time})"
                        )
                except _OPTIONAL_CATCH as exc:
                    errors.append(f"time restore unsupported: {exc}")
        if errors:
            raise StateError(
                "Failed to restore engine state/time after dataset generation: "
                + "; ".join(errors),
                operation="restore",
            )

    def _run_single_simulation(
        self,
        sample_id: int,
        config: GeneratorConfig,
        rng: np.random.Generator,
        n_steps: int,
        layout: ModelDoFLayout,
    ) -> SimulationSample:
        """Run a single simulation and record labeled data."""
        self.engine.reset()
        q0, v0 = self._generate_initial_conditions(config, rng, layout.n_q, layout.n_v)
        self.engine.set_state(q0, v0)

        idx = rng.integers(len(config.control_profiles))
        profile = config.control_profiles[idx]
        control_sequence = profile.generate(layout.n_u, n_steps, config.timestep, rng)

        buffers, trackers = self._allocate_sim_buffers(config, n_steps, layout)
        self._execute_sim_loop(config, control_sequence, n_steps, buffers, trackers)

        channel_evidence, finalized = finalize_channels(
            config, buffers, trackers, layout, n_steps
        )
        actuation_kind = self._infer_actuation_kind(
            finalized["requested_controls"], finalized["applied_controls"]
        )
        provenance = self._build_provenance(config, layout)

        metadata = {
            "sample_id": sample_id,
            "seed": config.seed,
            "duration": config.duration,
            "timestep": config.timestep,
            "initial_q": q0.tolist(),
            "initial_v": v0.tolist(),
            "control_profile": profile.name,
            "control_type": profile.profile_type,
            "label_schema": LABEL_SCHEMA,
            "dimensions": layout.as_dict(),
        }

        return SimulationSample(
            sample_id=sample_id,
            metadata=metadata,
            times=finalized["times"],
            positions=finalized["positions"],
            velocities=finalized["velocities"],
            accelerations=finalized["interval_accelerations"],
            torques=finalized["applied_controls"],
            mass_matrices=finalized["mass_matrices"],
            bias_forces=finalized["bias_forces"],
            gravity_forces=finalized["gravity"],
            contact_forces=finalized["contact"],
            drift_accelerations=finalized["drift"],
            control_accelerations=finalized["control_accel"],
            energies={
                key: value
                for key, value in (
                    ("kinetic", finalized["kinetic_energy"]),
                    ("potential", finalized["potential_energy"]),
                )
                if value is not None
            },
            dimensions=layout,
            channel_evidence=channel_evidence,
            acceleration_kind=AccelerationKind.INTERVAL_FINITE_DIFFERENCE,
            actuation_kind=actuation_kind,
            requested_controls=finalized["requested_controls"],
            applied_controls=finalized["applied_controls"],
            native_accelerations=finalized["native_accelerations"],
            interval_accelerations=finalized["interval_accelerations"],
            contact_labels=(
                ["fx", "fy", "fz"] if finalized["contact"] is not None else None
            ),
            provenance=provenance,
        )

    @staticmethod
    def _infer_actuation_kind(
        requested: np.ndarray | None, applied: np.ndarray | None
    ) -> ActuationKind:
        if requested is None or applied is None:
            return ActuationKind.APPLIED
        if not np.allclose(requested, applied):
            return ActuationKind.APPLIED_SATURATED
        return ActuationKind.APPLIED

    def _build_provenance(
        self, config: GeneratorConfig, layout: ModelDoFLayout
    ) -> SampleProvenance:
        model_name = str(getattr(self.engine, "model_name", "unknown"))
        engine_name = type(self.engine).__name__
        settings = {
            "seed": config.seed,
            "duration": config.duration,
            "timestep": config.timestep,
            "dimensions": layout.as_dict(),
            "record_mass_matrix": config.record_mass_matrix,
            "record_bias_forces": config.record_bias_forces,
            "record_gravity": config.record_gravity,
            "record_contact_forces": config.record_contact_forces,
        }
        settings_digest = hashlib.sha256(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        model_hash = hashlib.sha256(
            f"{engine_name}:{model_name}:{layout.as_dict()}".encode()
        ).hexdigest()
        return SampleProvenance(
            model_name=model_name,
            engine_name=engine_name,
            model_hash=model_hash,
            native_runtime=platform.python_version(),
            settings_digest=settings_digest,
            numerical_refinement="none",
        )

    def _generate_initial_conditions(
        self,
        config: GeneratorConfig,
        rng: np.random.Generator,
        n_q: int,
        n_v: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate randomized initial conditions."""
        if config is None:
            raise ValueError("config must be provided")
        if config.vary_initial_positions and config.position_ranges:
            q0 = np.zeros(n_q)
            for pr in config.position_ranges:
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

    def _get_layout(self) -> ModelDoFLayout:
        """Resolve n_q / n_v / n_u from the engine."""
        try:
            q, v = self.engine.get_state()
            n_q, n_v = len(q), len(v)
        except _OPTIONAL_CATCH:
            n_q = n_v = 7
        n_u = n_v
        getter = getattr(self.engine, "get_control_dim", None)
        num_u_attr = getattr(self.engine, "num_u", None)
        if callable(getter):
            try:
                n_u = int(getter())
            except _OPTIONAL_CATCH:
                n_u = n_v
        elif num_u_attr is not None:
            n_u = int(num_u_attr)
        return ModelDoFLayout(n_q=n_q, n_v=n_v, n_u=n_u, n_force=0)

    def _get_joint_names(self) -> list[str]:
        """Get joint names from engine."""
        try:
            names = self.engine.get_joint_names()
            if names:
                return names
        except _OPTIONAL_CATCH:
            pass
        layout = self._get_layout()
        return [f"joint_{i}" for i in range(layout.n_q)]


__all__ = [
    "ControlProfile",
    "DatasetGenerator",
    "GeneratorConfig",
    "ParameterRange",
    "SimulationSample",
    "TrainingDataset",
]
