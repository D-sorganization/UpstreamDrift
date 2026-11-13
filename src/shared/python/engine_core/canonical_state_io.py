"""ADR-0026 canonical-v2 dynamic state I/O for physics engines (#14936).

Converts between engine-native ``(q, v, a)`` and :class:`~pose_interchange.canonical_state.CanonicalState`
via the existing CC-7/CC-9 adapter boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.pose_interchange.adapters.mujoco import (
    CanonicalV2State,
    MujocoAdapter,
    MujocoNativeState,
)
from src.shared.python.pose_interchange.adapters.pinocchio_reference import (
    CanonicalV2State as PinocchioCanonicalV2State,
    PinocchioNativeState,
    PinocchioReferenceAdapter,
)
from src.shared.python.pose_interchange.canonical_state import (
    CONVENTION_TAG_V2,
    CanonicalState,
)

if TYPE_CHECKING:
    from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine

_CANONICAL_STATE_ENGINES = frozenset({"mujoco", "pinocchio"})


class _NullPinocchioBackend:
    """Backend placeholder; ``to_canonical_v2`` / ``from_canonical_v2`` do not call it."""

    def fk(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        raise RuntimeError(
            f"NullPinocchioBackend does not implement fk ({frame_name!r})"
        )  # tracked: #14936

    def jacobian(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        raise RuntimeError(
            f"NullPinocchioBackend does not implement jacobian ({frame_name!r})"
        )  # tracked: #14936

    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "NullPinocchioBackend does not implement rnea"
        )  # tracked: #14936

    def aba(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "NullPinocchioBackend does not implement aba"
        )  # tracked: #14936


def engine_supports_canonical_state(engine_type: str) -> bool:
    """Return whether ``engine_type`` has a canonical-v2 adapter wired."""
    return engine_type in _CANONICAL_STATE_ENGINES


def _v2_to_canonical(state: CanonicalV2State) -> CanonicalState:
    return CanonicalState(
        q=state.q,
        v=state.v,
        a=state.a,
        t=state.t,
        convention=state.convention,
        frame=state.frame,
        units=state.units,
    )


def _canonical_to_v2(state: CanonicalState) -> CanonicalV2State:
    if state.convention != CONVENTION_TAG_V2:
        raise ValueError(
            f"CanonicalState convention must be {CONVENTION_TAG_V2!r}, "
            f"got {state.convention!r}"
        )
    return CanonicalV2State(
        q=np.asarray(state.q, dtype=np.float64),
        v=np.asarray(state.v, dtype=np.float64),
        a=np.asarray(state.a, dtype=np.float64),
        t=float(state.t),
    )


def _native_acceleration(engine: BasePhysicsEngine, v: np.ndarray) -> np.ndarray:
    engine_type = engine.engine_type
    if engine_type == "mujoco":
        data = getattr(engine, "data", None)
        if data is not None and hasattr(data, "qacc"):
            # Do not call forward() here: mj_forward overwrites qacc with fresh
            # dynamics and drops an imported canonical acceleration (#14936).
            return np.asarray(data.qacc, dtype=np.float64).copy()
    if engine_type == "pinocchio":
        a_attr = getattr(engine, "a", None)
        if a_attr is not None:
            return np.asarray(a_attr, dtype=np.float64).copy()
    return np.zeros(v.shape[0], dtype=np.float64)


def read_canonical_state(engine: BasePhysicsEngine) -> CanonicalState:
    """Build canonical-v2 state from the engine's native dynamic arrays."""
    engine_type = engine.engine_type
    if not engine_supports_canonical_state(engine_type):
        raise NotImplementedError(  # tracked: #14936
            f"Engine {engine_type!r} does not implement canonical-v2 state I/O yet."
        )

    raw_state = engine.get_state()
    if raw_state is None or not isinstance(raw_state, tuple):
        raise TypeError(
            f"{engine_type!r} get_state() must return (q, v); got {type(raw_state)!r}"
        )
    q, v = raw_state
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    a_arr = _native_acceleration(engine, v_arr)
    t = float(engine.get_time())

    if engine_type == "mujoco":
        native = MujocoNativeState(qpos=q_arr, qvel=v_arr, qacc=a_arr, time=t)
        return _v2_to_canonical(MujocoAdapter().to_canonical_v2(native))
    if engine_type == "pinocchio":
        pin_native = PinocchioNativeState(q=q_arr, v=v_arr, a=a_arr)
        pin_v2 = PinocchioReferenceAdapter(_NullPinocchioBackend()).to_canonical_v2(
            pin_native
        )
        return CanonicalState(
            q=pin_v2.q,
            v=pin_v2.v,
            a=pin_v2.a,
            t=t,
        )
    raise NotImplementedError(engine_type)  # tracked: #14936  # pragma: no cover


def apply_canonical_state(
    engine: BasePhysicsEngine,
    state: CanonicalState,
) -> None:
    """Apply a canonical-v2 state through the adapter boundary."""
    engine_type = engine.engine_type
    if not engine_supports_canonical_state(engine_type):
        raise NotImplementedError(  # tracked: #14936
            f"Engine {engine_type!r} does not implement canonical-v2 state I/O yet."
        )

    if engine_type == "mujoco":
        mj_v2 = _canonical_to_v2(state)
        mj_native = MujocoAdapter().from_canonical_v2(mj_v2)
        engine.set_state(mj_native.qpos, mj_native.qvel)
        data = getattr(engine, "data", None)
        if data is not None:
            if hasattr(data, "qacc"):
                data.qacc[:] = mj_native.qacc
            if hasattr(data, "time"):
                data.time = float(state.t)
    elif engine_type == "pinocchio":
        if state.convention != CONVENTION_TAG_V2:
            raise ValueError(
                f"CanonicalState convention must be {CONVENTION_TAG_V2!r}, "
                f"got {state.convention!r}"
            )
        pin_v2 = PinocchioCanonicalV2State(
            q=np.asarray(state.q, dtype=np.float64),
            v=np.asarray(state.v, dtype=np.float64),
            a=np.asarray(state.a, dtype=np.float64),
        )
        pin_native = PinocchioReferenceAdapter(
            _NullPinocchioBackend()
        ).from_canonical_v2(pin_v2)
        engine.set_state(pin_native.q, pin_native.v)
        if hasattr(engine, "a"):
            engine.a = np.asarray(pin_native.a, dtype=np.float64).copy()
    else:  # pragma: no cover
        raise NotImplementedError(engine_type)  # tracked: #14936

    if hasattr(engine, "time"):
        engine.time = float(state.t)
