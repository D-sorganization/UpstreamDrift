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
from .config import (
    ControlProfile,
    GeneratorConfig,
    ParameterRange,
)
from .labels import (
    LABEL_SCHEMA,
    AccelerationKind,
    ActuationKind,
    ChannelAvailability,
    ChannelEvidence,
    ModelDoFLayout,
    SampleProvenance,
)
from .models import (
    SimulationSample,
    TrainingDataset,
)

logger = get_logger(__name__)

_OPTIONAL_CATCH = (
    ValueError,
    RuntimeError,
    AttributeError,
    TypeError,
    NotImplementedError,
)


@invariant(
    lambda self: self.engine is not None,
    "DatasetGenerator must have a valid engine reference",
)
class DatasetGenerator(_DatasetExportMixin):
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

        channel_evidence, finalized = self._finalize_channels(
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
    def _allocate_sim_buffers(
        config: GeneratorConfig,
        n_steps: int,
        layout: ModelDoFLayout,
    ) -> tuple[dict[str, np.ndarray | None], dict[str, bool]]:
        """Pre-allocate recording arrays and per-channel failure trackers."""
        n_q, n_v, n_u = layout.n_q, layout.n_v, layout.n_u
        buffers: dict[str, np.ndarray | None] = {
            "times": np.zeros(n_steps),
            "positions": np.zeros((n_steps, n_q)),
            "velocities": np.zeros((n_steps, n_v)),
            "interval_accelerations": np.zeros((n_steps, n_v)),
            "native_accelerations": np.zeros((n_steps, n_v)),
            "requested_controls": np.zeros((n_steps, n_u)),
            "applied_controls": np.zeros((n_steps, n_u)),
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
        trackers = {
            "mass_matrices": False,
            "bias_forces": False,
            "gravity": False,
            "contact": False,
            "drift": False,
            "control_accel": False,
            "native_accelerations": False,
            "kinetic_energy": False,
            "potential_energy": False,
            "interval_accelerations": False,
            "applied_controls": False,
        }
        return buffers, trackers

    def _execute_sim_loop(
        self,
        config: GeneratorConfig,
        control_sequence: np.ndarray,
        n_steps: int,
        buffers: dict[str, np.ndarray | None],
        trackers: dict[str, bool],
    ) -> None:
        """Execute the simulation loop, recording state and dynamics each step."""
        for step in range(n_steps):
            requested = np.asarray(control_sequence[step], dtype=float)
            buffers["requested_controls"][step] = requested  # type: ignore[index]
            self.engine.set_control(requested)

            applied = self._read_applied_control(requested, trackers)
            buffers["applied_controls"][step] = applied  # type: ignore[index]

            # Refresh instantaneous dynamics so native labels match current u.
            self._refresh_instantaneous_dynamics()

            q, v = self.engine.get_state()
            t = self.engine.get_time()
            buffers["times"][step] = t  # type: ignore[index]
            buffers["positions"][step] = q  # type: ignore[index]
            buffers["velocities"][step] = v  # type: ignore[index]

            self._record_native_acceleration(step, buffers, trackers)
            self._record_dynamics_step(config, step, applied, v, buffers, trackers)

            self.engine.step(config.timestep)

            try:
                _, v_new = self.engine.get_state()
                buffers["interval_accelerations"][step] = (  # type: ignore[index]
                    (v_new - v) / config.timestep
                )
            except _OPTIONAL_CATCH:
                trackers["interval_accelerations"] = True

    def _refresh_instantaneous_dynamics(self) -> None:
        """Call ``forward()`` when available so native a matches current u."""
        forward = getattr(self.engine, "forward", None)
        if not callable(forward):
            return
        try:
            forward()
        except _OPTIONAL_CATCH:
            logger.debug(
                "engine.forward() unavailable; native acceleration may be stale",
                exc_info=True,
            )

    def _read_applied_control(
        self, requested: np.ndarray, trackers: dict[str, bool]
    ) -> np.ndarray:
        getter = getattr(self.engine, "get_applied_control", None)
        if callable(getter):
            try:
                return np.asarray(getter(), dtype=float).reshape(-1)
            except _OPTIONAL_CATCH:
                trackers["applied_controls"] = True
                return requested.copy()
        return requested.copy()

    def _record_native_acceleration(
        self,
        step: int,
        buffers: dict[str, np.ndarray | None],
        trackers: dict[str, bool],
    ) -> None:
        """Record native a after ``_refresh_instantaneous_dynamics``."""
        getter = getattr(self.engine, "get_joint_accelerations", None)
        if not callable(getter):
            trackers["native_accelerations"] = True
            return
        try:
            accel = np.asarray(getter(), dtype=float).reshape(-1).copy()
            target = buffers["native_accelerations"]
            if target is None:
                trackers["native_accelerations"] = True
                return
            n_v = target.shape[1]
            if accel.size < n_v:
                trackers["native_accelerations"] = True
                return
            target[step] = accel[:n_v]
        except _OPTIONAL_CATCH:
            trackers["native_accelerations"] = True

    def _record_dynamics_step(
        self,
        config: GeneratorConfig,
        step: int,
        tau: np.ndarray,
        v: np.ndarray,
        buffers: dict[str, np.ndarray | None],
        trackers: dict[str, bool],
    ) -> None:
        """Record optional dynamics quantities; failures mark trackers."""
        if config is None:
            raise ValueError("config must be provided")

        if config.record_mass_matrix and buffers["mass_matrices"] is not None:
            try:
                buffers["mass_matrices"][step] = self.engine.compute_mass_matrix()
            except _OPTIONAL_CATCH:
                trackers["mass_matrices"] = True

        if config.record_bias_forces and buffers["bias_forces"] is not None:
            try:
                buffers["bias_forces"][step] = self.engine.compute_bias_forces()
            except _OPTIONAL_CATCH:
                trackers["bias_forces"] = True

        if config.record_gravity and buffers["gravity"] is not None:
            try:
                buffers["gravity"][step] = self.engine.compute_gravity_forces()
            except _OPTIONAL_CATCH:
                trackers["gravity"] = True

        if config.record_contact_forces and buffers["contact"] is not None:
            try:
                cf = np.asarray(self.engine.compute_contact_forces(), dtype=float)
                buffers["contact"][step, : min(3, cf.size)] = cf.reshape(-1)[:3]
            except _OPTIONAL_CATCH:
                trackers["contact"] = True

        if config.record_drift_control:
            try:
                if buffers["drift"] is not None:
                    buffers["drift"][step] = self.engine.compute_drift_acceleration()
            except _OPTIONAL_CATCH:
                trackers["drift"] = True
            try:
                if buffers["control_accel"] is not None:
                    buffers["control_accel"][step] = (
                        self.engine.compute_control_acceleration(tau)
                    )
            except _OPTIONAL_CATCH:
                trackers["control_accel"] = True

        try:
            M = self.engine.compute_mass_matrix()
            buffers["kinetic_energy"][step] = 0.5 * float(v.T @ M @ v)  # type: ignore[index]
        except _OPTIONAL_CATCH:
            trackers["kinetic_energy"] = True
        try:
            pe = getattr(self.engine, "compute_potential_energy", None)
            if callable(pe):
                buffers["potential_energy"][step] = float(pe())  # type: ignore[index]
            else:
                trackers["potential_energy"] = True
        except _OPTIONAL_CATCH:
            trackers["potential_energy"] = True

    def _finalize_channels(
        self,
        config: GeneratorConfig,
        buffers: dict[str, np.ndarray | None],
        trackers: dict[str, bool],
        layout: ModelDoFLayout,
        n_steps: int,
    ) -> tuple[dict[str, ChannelEvidence], dict[str, Any]]:
        """Drop failed optional buffers and build channel evidence."""
        finalized: dict[str, Any] = {
            "times": buffers["times"],
            "positions": buffers["positions"],
            "velocities": buffers["velocities"],
            "requested_controls": buffers["requested_controls"],
            "applied_controls": buffers["applied_controls"],
            "interval_accelerations": (
                None
                if trackers["interval_accelerations"]
                else buffers["interval_accelerations"]
            ),
            "native_accelerations": (
                None
                if trackers["native_accelerations"]
                else buffers["native_accelerations"]
            ),
            "mass_matrices": self._drop_if_failed(
                config.record_mass_matrix,
                trackers["mass_matrices"],
                buffers["mass_matrices"],
            ),
            "bias_forces": self._drop_if_failed(
                config.record_bias_forces,
                trackers["bias_forces"],
                buffers["bias_forces"],
            ),
            "gravity": self._drop_if_failed(
                config.record_gravity, trackers["gravity"], buffers["gravity"]
            ),
            "contact": self._drop_if_failed(
                config.record_contact_forces,
                trackers["contact"],
                buffers["contact"],
            ),
            "drift": self._drop_if_failed(
                config.record_drift_control, trackers["drift"], buffers["drift"]
            ),
            "control_accel": self._drop_if_failed(
                config.record_drift_control,
                trackers["control_accel"],
                buffers["control_accel"],
            ),
            "kinetic_energy": (
                None if trackers["kinetic_energy"] else buffers["kinetic_energy"]
            ),
            "potential_energy": (
                None if trackers["potential_energy"] else buffers["potential_energy"]
            ),
        }

        evidence: dict[str, ChannelEvidence] = {
            "positions": ChannelEvidence(
                name="positions",
                availability=ChannelAvailability.AVAILABLE,
                semantic="configuration coordinates q",
                units="rad_or_m",
                shape=(n_steps, layout.n_q),
            ),
            "velocities": ChannelEvidence(
                name="velocities",
                availability=ChannelAvailability.AVAILABLE,
                semantic="tangent velocities v",
                units="rad_s_or_m_s",
                shape=(n_steps, layout.n_v),
            ),
            "requested_controls": ChannelEvidence(
                name="requested_controls",
                availability=ChannelAvailability.AVAILABLE,
                semantic="commanded actuator inputs before saturation",
                units="N_m_or_N",
                shape=(n_steps, layout.n_u),
            ),
            "applied_controls": ChannelEvidence(
                name="applied_controls",
                availability=ChannelAvailability.AVAILABLE,
                semantic="applied actuator inputs after saturation",
                units="N_m_or_N",
                shape=(n_steps, layout.n_u),
            ),
            "interval_accelerations": self._evidence_for(
                name="interval_accelerations",
                requested=True,
                failed=trackers["interval_accelerations"],
                values=finalized["interval_accelerations"],
                semantic="interval finite-difference acceleration from post-step v",
                units="rad_s2_or_m_s2",
                shape=(n_steps, layout.n_v),
            ),
            "native_accelerations": self._evidence_for(
                name="native_accelerations",
                requested=True,
                failed=trackers["native_accelerations"],
                values=finalized["native_accelerations"],
                semantic="instantaneous native acceleration from engine dynamics",
                units="rad_s2_or_m_s2",
                shape=(n_steps, layout.n_v),
            ),
            "mass_matrices": self._evidence_for(
                name="mass_matrices",
                requested=config.record_mass_matrix,
                failed=trackers["mass_matrices"],
                values=finalized["mass_matrices"],
                semantic="mass matrix M(q)",
                units="kg_m2",
                shape=(n_steps, layout.n_v, layout.n_v),
                fail_note="engine raised while computing mass matrix",
            ),
            "bias_forces": self._evidence_for(
                name="bias_forces",
                requested=config.record_bias_forces,
                failed=trackers["bias_forces"],
                values=finalized["bias_forces"],
                semantic="bias forces h(q,v)",
                units="N_m_or_N",
                shape=(n_steps, layout.n_v),
                fail_note="engine raised while computing bias forces",
            ),
            "gravity_forces": self._evidence_for(
                name="gravity_forces",
                requested=config.record_gravity,
                failed=trackers["gravity"],
                values=finalized["gravity"],
                semantic="gravity generalized forces",
                units="N_m_or_N",
                shape=(n_steps, layout.n_v),
                fail_note="engine raised while computing gravity forces",
            ),
            "contact_forces": self._evidence_for(
                name="contact_forces",
                requested=config.record_contact_forces,
                failed=trackers["contact"],
                values=finalized["contact"],
                semantic="contact force components",
                units="N",
                shape=(n_steps, 3),
                fail_note="engine raised or lacks contact channels",
            ),
        }
        return evidence, finalized

    @staticmethod
    def _drop_if_failed(
        requested: bool, failed: bool, buffer: np.ndarray | None
    ) -> np.ndarray | None:
        if not requested:
            return None
        if failed:
            return None
        return buffer

    @staticmethod
    def _evidence_for(
        *,
        name: str,
        requested: bool,
        failed: bool,
        values: np.ndarray | None,
        semantic: str,
        units: str,
        shape: tuple[int, ...],
        fail_note: str = "channel unavailable",
    ) -> ChannelEvidence:
        if not requested:
            return ChannelEvidence(
                name=name,
                availability=ChannelAvailability.NOT_REQUESTED,
                semantic=semantic,
                units=units,
                shape=None,
                notes="not requested by GeneratorConfig",
            )
        if failed or values is None:
            return ChannelEvidence(
                name=name,
                availability=ChannelAvailability.UNAVAILABLE,
                semantic=semantic,
                units=units,
                shape=None,
                notes=fail_note,
            )
        return ChannelEvidence(
            name=name,
            availability=ChannelAvailability.AVAILABLE,
            semantic=semantic,
            units=units,
            shape=shape,
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
