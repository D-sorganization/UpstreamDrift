"""First-wave native generator adapters for NM-02 (#10617).

Qualify DatasetGenerator evidence against MockPhysicsEngine (software
contracts) and the analytical ODE double-pendulum backend (native residual
on the source clock). Model ids reuse the NM-01 roster — no parallel store.
Remaining engines stay under NM-09 ownership.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from src.shared.python.data_io.dataset_generator.config import (
    ControlProfile,
    GeneratorConfig,
)
from src.shared.python.data_io.dataset_generator.core import DatasetGenerator
from src.shared.python.data_io.dataset_generator.models import SimulationSample
from src.shared.python.data_io.dataset_generator.labels import (
    LABEL_SCHEMA,
    ChannelAvailability,
    dynamics_residual,
)
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine
from src.shared.python.neural_motion.roster import (
    RosterStage,
    build_neural_model_roster,
    resolve_roster_entry,
)

__all__ = [
    "FIRST_WAVE_MODEL_IDS",
    "NativeLabelReceipt",
    "qualify_first_wave_adapters",
    "qualify_mock_adapter",
    "qualify_ode_double_pendulum_adapter",
]


def _pilot_model_ids() -> tuple[str, ...]:
    roster = build_neural_model_roster()
    return tuple(
        entry.model_id
        for entry in roster.entries
        if entry.pilot_stage is RosterStage.PILOT_ELIGIBLE
    )


FIRST_WAVE_MODEL_IDS: tuple[str, ...] = _pilot_model_ids()


@dataclass(frozen=True)
class NativeLabelReceipt:
    """Content-addressed qualification receipt for one adapter."""

    schema: str
    model_id: str
    adapter: str
    qualified: bool
    residual_norm: float | None
    channel_summary: dict[str, str]
    limitations: tuple[str, ...]
    content_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "adapter": self.adapter,
            "qualified": self.qualified,
            "residual_norm": self.residual_norm,
            "channel_summary": dict(self.channel_summary),
            "limitations": list(self.limitations),
            "content_digest": self.content_digest,
        }


def _digest(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _channel_summary(sample: Any) -> dict[str, str]:
    return {
        name: evidence.availability.value
        for name, evidence in sample.channel_evidence.items()
    }


def _first_step_native_residual(sample: SimulationSample) -> np.ndarray | None:
    """M·a + h − u on step 0 when all native dynamics channels are present."""
    if (
        sample.mass_matrices is None
        or sample.bias_forces is None
        or sample.native_accelerations is None
        or sample.applied_controls is None
    ):
        return None
    return dynamics_residual(
        mass=sample.mass_matrices[0],
        acceleration=sample.native_accelerations[0],
        bias=sample.bias_forces[0],
        applied=sample.applied_controls[0],
    )


def qualify_mock_adapter() -> NativeLabelReceipt:
    """Software-contract qualification via MockPhysicsEngine."""
    engine = MockPhysicsEngine(num_joints=2)
    engine.load_from_string("<mock/>")
    gen = DatasetGenerator(cast(PhysicsEngine, engine))
    config = GeneratorConfig(
        num_samples=1,
        duration=0.04,
        timestep=0.01,
        seed=11,
        vary_initial_positions=False,
        vary_initial_velocities=False,
        control_profiles=[
            ControlProfile(
                name="const",
                profile_type="constant",
                parameters={"magnitude": 0.5},
            )
        ],
        record_mass_matrix=True,
        record_bias_forces=True,
        record_gravity=False,
        record_contact_forces=False,
        record_drift_control=False,
    )
    sample = gen.generate(config).samples[0]
    residual_norm: float | None = None
    qualified = False
    limitations: list[str] = [
        "MockPhysicsEngine is a software-contract fixture, not native evidence",
    ]
    residual = _first_step_native_residual(sample)
    if residual is not None:
        residual_norm = float(np.linalg.norm(residual))
        qualified = residual_norm < 1e-8 and all(
            sample.channel_evidence[name].availability is ChannelAvailability.AVAILABLE
            for name in (
                "positions",
                "velocities",
                "native_accelerations",
                "applied_controls",
                "mass_matrices",
                "bias_forces",
            )
        )
    payload = {
        "schema": LABEL_SCHEMA,
        "model_id": "mock_software_contract",
        "adapter": "MockPhysicsEngine",
        "qualified": qualified,
        "residual_norm": residual_norm,
        "channel_summary": _channel_summary(sample),
        "limitations": limitations,
        "native_runtime": platform.python_version(),
    }
    return NativeLabelReceipt(
        schema=LABEL_SCHEMA,
        model_id="mock_software_contract",
        adapter="MockPhysicsEngine",
        qualified=qualified,
        residual_norm=residual_norm,
        channel_summary=_channel_summary(sample),
        limitations=tuple(limitations),
        content_digest=_digest(payload),
    )


class _ODEPhysicsEngineAdapter:
    """Thin PhysicsEngine-shaped wrapper around ODEBackend for generation."""

    def __init__(self) -> None:
        from src.shared.python.simulation_backends.model_params import GolfModelParams
        from src.shared.python.simulation_backends.ode_backend import ODEBackend

        self._backend = ODEBackend(GolfModelParams.default(), dt=0.01)
        self.model_name = "driven_double_pendulum"
        self._time = 0.0

    def reset(self) -> None:
        self._backend.reset()
        self._time = float(self._backend.get_time())

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        state = self._backend.get_state()
        return state.q.copy(), state.v.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        from src.shared.python.simulation_backends.protocol import SimState

        self._backend.reset(
            SimState(
                q=np.asarray(q, dtype=float),
                v=np.asarray(v, dtype=float),
                time=self._time,
            )
        )

    def set_control(self, u: np.ndarray) -> None:
        self._backend.set_control(np.asarray(u, dtype=float))

    def get_applied_control(self) -> np.ndarray:
        applied = getattr(self._backend, "get_control", None)
        if callable(applied):
            return np.asarray(applied(), dtype=float).copy()
        return np.asarray(self._backend._u, dtype=float).copy()

    def get_time(self) -> float:
        return float(self._backend.get_time())

    def set_time(self, time: float) -> None:
        if time is None or not np.isfinite(time):
            raise ValueError(f"time must be finite; got {time!r}")
        self._time = float(time)
        q, v = self.get_state()
        self.set_state(q, v)

    def step(self, dt: float) -> None:
        self._backend.step(dt)
        self._time = float(self._backend.get_time())

    def forward(self) -> None:
        # Instantaneous dynamics are evaluated on demand in getters.
        return

    def get_joint_names(self) -> list[str]:
        return ["shoulder", "wrist"]

    def get_control_dim(self) -> int:
        return 2

    def get_joint_accelerations(self) -> np.ndarray:
        q, v = self.get_state()
        return self._backend.forward_dynamics(q, v, self.get_applied_control())

    def compute_mass_matrix(self) -> np.ndarray:
        q, _ = self.get_state()
        return self._backend.mass_matrix(q)

    def compute_bias_forces(self) -> np.ndarray:
        q, v = self.get_state()
        return self._backend.bias_forces(q, v)

    def compute_gravity_forces(self) -> np.ndarray:
        raise RuntimeError("ODE adapter does not expose gravity separately")

    def compute_contact_forces(self) -> np.ndarray:
        raise RuntimeError("ODE double pendulum has no contact channels")


def qualify_ode_double_pendulum_adapter() -> NativeLabelReceipt:
    """Native residual qualification for driven_double_pendulum via ODEBackend."""
    resolve_roster_entry(build_neural_model_roster(), "driven_double_pendulum")
    engine = _ODEPhysicsEngineAdapter()
    gen = DatasetGenerator(cast(PhysicsEngine, engine))
    config = GeneratorConfig(
        num_samples=1,
        duration=0.05,
        timestep=0.01,
        seed=3,
        vary_initial_positions=False,
        vary_initial_velocities=False,
        control_profiles=[
            ControlProfile(
                name="const",
                profile_type="constant",
                parameters={"magnitude": 0.25},
            )
        ],
        record_mass_matrix=True,
        record_bias_forces=True,
        record_gravity=False,
        record_contact_forces=False,
        record_drift_control=False,
    )
    sample = gen.generate(config).samples[0]
    limitations = [
        "Contact/gravity channels intentionally unavailable on planar ODE model",
        "Triple-pendulum and upper-body adapters deferred to NM-09 ownership",
    ]
    residual_norm: float | None = None
    qualified = False
    residual = _first_step_native_residual(sample)
    if residual is not None:
        residual_norm = float(np.linalg.norm(residual))
        qualified = (
            residual_norm < 1e-6
            and sample.channel_evidence["native_accelerations"].availability
            is ChannelAvailability.AVAILABLE
            and sample.channel_evidence["contact_forces"].availability
            is ChannelAvailability.NOT_REQUESTED
        )
    payload = {
        "schema": LABEL_SCHEMA,
        "model_id": "driven_double_pendulum",
        "adapter": "ODEBackend",
        "qualified": qualified,
        "residual_norm": residual_norm,
        "channel_summary": _channel_summary(sample),
        "limitations": limitations,
        "native_runtime": platform.python_version(),
        "model_hash": sample.provenance.model_hash if sample.provenance else "",
    }
    return NativeLabelReceipt(
        schema=LABEL_SCHEMA,
        model_id="driven_double_pendulum",
        adapter="ODEBackend",
        qualified=qualified,
        residual_norm=residual_norm,
        channel_summary=_channel_summary(sample),
        limitations=tuple(limitations),
        content_digest=_digest(payload),
    )


def qualify_first_wave_adapters() -> tuple[NativeLabelReceipt, ...]:
    """Return receipts for first-wave adapters exercised in this child."""
    return (
        qualify_mock_adapter(),
        qualify_ode_double_pendulum_adapter(),
    )
