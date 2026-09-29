"""Canonical-core adapter helpers for MyoSuite MJCF models.

MyoSuite is activation-driven and backed by MuJoCo MJCF models. Its native
state layout matches MuJoCo free-joint ``qpos`` / ``qvel`` conventions:
``qpos=[xyz, quat_wxyz, joints...]`` and ``qvel=[v_xyz, w_body, djoints...]``.
The canonical-core contract therefore needs validation and provenance, not a
coordinate permutation, at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.biomechanics import rust_muscle
from src.shared.python.simulation_backends.protocol import Trace

CANONICAL_CONVENTION = "canonical-v2"
CANONICAL_FRAME = "world_Zup"
CANONICAL_UNITS = "SI"

SUPPORTED_CAPABILITIES = ("MUSCLES", "FORWARD_DYN", "CONTACT")
UNSUPPORTED_CAPABILITIES = ("JOINT_TORQUE_INVERSE_DYN",)


def _as_vector(name: str, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    if value is None:
        raise ValueError(f"{name} must be provided")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _as_history(
    name: str, value: npt.ArrayLike, expected_rows: int | None = None
) -> npt.NDArray[np.float64]:
    if value is None:
        raise ValueError(f"{name} must be provided")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D array, got {array.shape}")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError(f"{name} has {array.shape[0]} rows, expected {expected_rows}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _normalise_free_joint_quaternion(q: npt.NDArray[np.float64]) -> np.ndarray:
    out = q.copy()
    if out.size < 7:
        return out
    norm = float(np.linalg.norm(out[3:7]))
    if norm <= 0.0:
        raise ValueError("free-joint quaternion must have non-zero norm")
    out[3:7] = out[3:7] / norm
    return out


@dataclass(frozen=True)
class NativeMyoSuiteState:
    """MyoSuite/MuJoCo-native state at one instant."""

    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    qacc: npt.NDArray[np.float64] | None = None
    ctrl: npt.NDArray[np.float64] | None = None
    time: float = 0.0

    def __post_init__(self) -> None:
        qpos = _as_vector("qpos", self.qpos)
        qvel = _as_vector("qvel", self.qvel)
        if qpos.size not in {qvel.size, qvel.size + 1}:
            raise ValueError(
                "MyoSuite qpos/qvel sizes must match fixed-base layout or "
                f"free-joint layout; got qpos={qpos.size}, qvel={qvel.size}"
            )
        qacc = None if self.qacc is None else _as_vector("qacc", self.qacc)
        if qacc is not None and qacc.shape != qvel.shape:
            raise ValueError(
                f"qacc shape must match qvel; got {qacc.shape} vs {qvel.shape}"
            )
        ctrl = None if self.ctrl is None else _as_vector("ctrl", self.ctrl)
        if not np.isfinite(float(self.time)):
            raise ValueError("time must be finite")
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "qvel", qvel)
        object.__setattr__(self, "qacc", qacc)
        object.__setattr__(self, "ctrl", ctrl)
        object.__setattr__(self, "time", float(self.time))


@dataclass(frozen=True)
class MyoSuiteCanonicalState:
    """Canonical-v2 state plus provenance tags for a MyoSuite frame."""

    q: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]
    t: float
    convention: str = CANONICAL_CONVENTION
    frame: str = CANONICAL_FRAME
    units: str = CANONICAL_UNITS
    angular_velocity_frame: str = "body"

    def __post_init__(self) -> None:
        q = _as_vector("q", self.q)
        v = _as_vector("v", self.v)
        a = _as_vector("a", self.a)
        if q.size not in {v.size, v.size + 1}:
            raise ValueError(
                "canonical q/v sizes must match fixed-base layout or "
                f"free-joint layout; got q={q.size}, v={v.size}"
            )
        if a.shape != v.shape:
            raise ValueError(f"a shape must match v; got {a.shape} vs {v.shape}")
        if not np.isfinite(float(self.t)):
            raise ValueError("t must be finite")
        object.__setattr__(self, "q", _normalise_free_joint_quaternion(q))
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "t", float(self.t))

    def provenance_meta(self) -> dict[str, str]:
        """Return schema-safe provenance fields for Trace metadata."""
        return {
            "convention": self.convention,
            "frame": self.frame,
            "units": self.units,
            "angular_velocity_frame": self.angular_velocity_frame,
        }


@dataclass(frozen=True)
class MyoSuiteMuscleOutputs:
    """Muscle-output history to persist through the unified Trace schema."""

    muscle_names: tuple[str, ...]
    activations: npt.NDArray[np.float64]
    forces: npt.NDArray[np.float64]
    lengths: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.muscle_names)
        if not names:
            raise ValueError("muscle_names must be non-empty")
        activations = _as_history("activations", self.activations)
        forces = _as_history("forces", self.forces, activations.shape[0])
        lengths = _as_history("lengths", self.lengths, activations.shape[0])
        velocities = _as_history("velocities", self.velocities, activations.shape[0])
        expected_shape = activations.shape
        for name, array in (
            ("forces", forces),
            ("lengths", lengths),
            ("velocities", velocities),
        ):
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} shape must match activations; "
                    f"got {array.shape} vs {expected_shape}"
                )
        if expected_shape[1] != len(names):
            raise ValueError(
                f"muscle_names has {len(names)} entries but histories have "
                f"{expected_shape[1]} columns"
            )
        object.__setattr__(self, "muscle_names", names)
        object.__setattr__(self, "activations", np.clip(activations, 0.0, 1.0))
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "velocities", velocities)

    @classmethod
    def from_analyzer(cls, analyzer: Any) -> MyoSuiteMuscleOutputs:
        """Build a one-frame output record from ``MyoSuiteMuscleAnalyzer``."""
        return cls(
            muscle_names=tuple(analyzer.muscle_names),
            activations=analyzer.get_muscle_activations(),
            forces=analyzer.get_muscle_forces(),
            lengths=analyzer.get_muscle_lengths(),
            velocities=analyzer.get_muscle_velocities(),
        )


class MyoSuiteCanonicalAdapter:
    """Adapter boundary for MyoSuite canonical-core state and outputs."""

    engine_name = "myosuite"

    def supported_capabilities(self) -> tuple[str, ...]:
        """Return CC-7 capability IDs this adapter intentionally supports."""
        return SUPPORTED_CAPABILITIES

    def unsupported_capabilities(self) -> tuple[str, ...]:
        """Return capability IDs this adapter intentionally does not claim."""
        return UNSUPPORTED_CAPABILITIES

    def to_canonical_state(
        self, native_state: NativeMyoSuiteState
    ) -> MyoSuiteCanonicalState:
        """Map MyoSuite native state to canonical-v2 state."""
        qpos = _normalise_free_joint_quaternion(native_state.qpos)
        qacc = (
            np.zeros_like(native_state.qvel)
            if native_state.qacc is None
            else native_state.qacc
        )
        return MyoSuiteCanonicalState(
            q=qpos,
            v=native_state.qvel,
            a=qacc,
            t=native_state.time,
        )

    def from_canonical_state(
        self,
        state: MyoSuiteCanonicalState,
        *,
        ctrl: npt.ArrayLike | None = None,
    ) -> NativeMyoSuiteState:
        """Map canonical-v2 state back to MyoSuite native qpos/qvel/qacc."""
        ctrl_array = None if ctrl is None else _as_vector("ctrl", ctrl)
        return NativeMyoSuiteState(
            qpos=state.q,
            qvel=state.v,
            qacc=state.a,
            ctrl=ctrl_array,
            time=state.t,
        )

    def advance_activations(
        self,
        excitations: npt.ArrayLike,
        activations: npt.ArrayLike,
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Advance muscle activations through the upstream-muscle kernel."""
        return rust_muscle.activation_step_batch(
            _as_vector("excitations", excitations),
            _as_vector("activations", activations),
            float(dt),
        )

    def estimate_muscle_forces(
        self,
        activations: npt.ArrayLike,
        lengths: npt.ArrayLike,
        velocities: npt.ArrayLike,
        params: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Estimate muscle forces through the upstream-muscle Hill kernel."""
        return rust_muscle.muscle_force_batch(
            _as_vector("activations", activations),
            _as_vector("lengths", lengths),
            _as_vector("velocities", velocities),
            np.asarray(params, dtype=np.float64),
        )

    def build_trace(
        self,
        *,
        t: npt.ArrayLike,
        q: npt.ArrayLike,
        v: npt.ArrayLike,
        muscle_outputs: MyoSuiteMuscleOutputs,
        u: npt.ArrayLike | None = None,
        dt: float = 0.0,
        meta: dict[str, object] | None = None,
    ) -> Trace:
        """Create a unified Trace carrying MyoSuite muscle outputs."""
        t_arr = _as_vector("t", t)
        q_arr = _as_history("q", q, t_arr.size)
        v_arr = _as_history("v", v, t_arr.size)
        if muscle_outputs.activations.shape[0] != t_arr.size:
            raise ValueError(
                "muscle output history rows must match t; "
                f"got {muscle_outputs.activations.shape[0]} vs {t_arr.size}"
            )
        trace_meta: dict[str, object] = {
            "convention": CANONICAL_CONVENTION,
            "frame": CANONICAL_FRAME,
            "units": CANONICAL_UNITS,
            "activation_source": "upstream-muscle",
        }
        if meta:
            trace_meta.update(meta)
        return Trace(
            t=t_arr,
            q=q_arr,
            v=v_arr,
            u=None if u is None else _as_history("u", u, t_arr.size),
            dt=float(dt),
            backend=self.engine_name,
            meta=trace_meta,
            muscle_names=muscle_outputs.muscle_names,
            muscle_activations=muscle_outputs.activations,
            muscle_forces=muscle_outputs.forces,
            muscle_lengths=muscle_outputs.lengths,
            muscle_velocities=muscle_outputs.velocities,
        )


__all__ = [
    "CANONICAL_CONVENTION",
    "CANONICAL_FRAME",
    "CANONICAL_UNITS",
    "MyoSuiteCanonicalAdapter",
    "MyoSuiteCanonicalState",
    "MyoSuiteMuscleOutputs",
    "NativeMyoSuiteState",
    "SUPPORTED_CAPABILITIES",
    "UNSUPPORTED_CAPABILITIES",
]
