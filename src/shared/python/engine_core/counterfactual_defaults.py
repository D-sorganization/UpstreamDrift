"""Default counterfactual accelerations for engines without native hooks (#14943)."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Protocol

import numpy as np

if TYPE_CHECKING:

    class _MassGravityEngine(Protocol):
        def compute_mass_matrix(self) -> np.ndarray: ...
        def compute_gravity_forces(self) -> np.ndarray: ...
        def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray: ...


@contextmanager
def _with_state_at_q(engine: Any, q: np.ndarray) -> Iterator[None]:
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if hasattr(engine, "get_state") and hasattr(engine, "set_state"):
        orig = engine.get_state()
        try:
            is_pair = isinstance(orig, tuple) and len(orig) == 2
            engine.set_state(q_arr, np.zeros_like(orig[1]) if is_pair else q_arr)
            yield
        finally:
            if isinstance(orig, tuple):
                engine.set_state(*orig)
            else:
                engine.set_state(orig)
    elif hasattr(engine, "q"):
        orig_q = engine.q
        try:
            engine.q = q_arr
            yield
        finally:
            engine.q = orig_q
    else:
        yield


def default_gravity_only_acceleration(
    engine: _MassGravityEngine, q: np.ndarray
) -> np.ndarray:
    with _with_state_at_q(engine, q):
        return np.linalg.solve(
            engine.compute_mass_matrix(), -engine.compute_gravity_forces()
        )


def default_control_acceleration_at_state(
    engine: _MassGravityEngine, q: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    tau_v = np.asarray(tau, dtype=float).reshape(-1)
    with _with_state_at_q(engine, q):
        return engine.compute_control_acceleration(tau_v)


def deprecated_compute_zvcf_shim(engine: Any, q: np.ndarray) -> np.ndarray:
    msg = "compute_zvcf(q) is legacy; use compute_control_acceleration_at_state or compute_gravity_only_acceleration"
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    if hasattr(engine, "compute_gravity_only_acceleration"):
        return engine.compute_gravity_only_acceleration(q)
    return default_gravity_only_acceleration(engine, q)
