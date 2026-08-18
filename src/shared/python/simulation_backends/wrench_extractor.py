"""Engine-agnostic GRF and contact-wrench helpers for canonical traces.

This module promotes the existing :class:`bunkershot3d.postproc.WrenchTrace`
primitive into the shared simulation layer without forking its integration and
resampling behavior. Shared backends exchange wrenches through the canonical
``Trace.wrench`` array with layout ``[fx, fy, fz, tx, ty, tz]`` in world-frame
newtons and newton-metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.bunkershot3d.postproc.wrench_trace import WrenchTrace

from .protocol import Trace

WrenchArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WrenchImpulses:
    """Linear and angular impulses integrated from a wrench trace."""

    linear_impulse: WrenchArray
    angular_impulse: WrenchArray


def _float_array(value: object, *, name: str) -> WrenchArray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _time_vector(time: object) -> WrenchArray:
    arr = _float_array(time, name="time").reshape(-1)
    if arr.size == 0:
        raise ValueError("time must be non-empty")
    if arr.size > 1 and np.any(np.diff(arr) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return arr


def _component_array(value: object, *, name: str) -> WrenchArray:
    arr = _float_array(value, name=name)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (T, 3), got {arr.shape}")
    return arr


def wrench_trace_from_force_torque(
    time: object,
    force_world: object,
    torque_world: object,
) -> WrenchTrace:
    """Build the shared WrenchTrace primitive from force and torque arrays."""

    time_arr = _time_vector(time)
    force_arr = _component_array(force_world, name="force_world")
    torque_arr = _component_array(torque_world, name="torque_world")
    if force_arr.shape[0] != time_arr.size or torque_arr.shape[0] != time_arr.size:
        raise ValueError(
            "time, force_world, and torque_world must share the same timestep count"
        )
    return WrenchTrace(time_arr, force_arr, torque_arr)


def wrench_array_from_force_torque(
    force_world: object, torque_world: object
) -> WrenchArray:
    """Pack force and torque arrays into canonical ``Trace.wrench`` layout."""

    force_arr = _component_array(force_world, name="force_world")
    torque_arr = _component_array(torque_world, name="torque_world")
    if force_arr.shape[0] != torque_arr.shape[0]:
        raise ValueError(
            "force_world and torque_world must have the same timestep count"
        )
    return np.concatenate((force_arr, torque_arr), axis=1)


def force_torque_from_wrench_array(wrench: object) -> tuple[WrenchArray, WrenchArray]:
    """Unpack canonical ``Trace.wrench`` data into force and torque arrays."""

    wrench_arr = _float_array(wrench, name="wrench")
    if wrench_arr.ndim != 2 or wrench_arr.shape[1] != 6:
        raise ValueError(f"wrench must have shape (T, 6), got {wrench_arr.shape}")
    return wrench_arr[:, :3], wrench_arr[:, 3:]


def wrench_trace_from_array(time: object, wrench: object) -> WrenchTrace:
    """Build a WrenchTrace from canonical ``Trace.wrench`` data."""

    force_world, torque_world = force_torque_from_wrench_array(wrench)
    return wrench_trace_from_force_torque(time, force_world, torque_world)


def wrench_array_from_trace(wrench_trace: WrenchTrace) -> WrenchArray:
    """Convert the existing WrenchTrace primitive to canonical array layout."""

    return wrench_array_from_force_torque(
        wrench_trace.force_world,
        wrench_trace.torque_world,
    )


def compute_wrench_impulses(time: object, wrench: object) -> WrenchImpulses:
    """Integrate canonical wrench data using WrenchTrace.get_impulses()."""

    wrench_trace = wrench_trace_from_array(time, wrench)
    linear_impulse, angular_impulse = wrench_trace.get_impulses()
    return WrenchImpulses(
        linear_impulse=np.asarray(linear_impulse, dtype=float),
        angular_impulse=np.asarray(angular_impulse, dtype=float),
    )


def trace_wrench_impulses(trace: Trace) -> WrenchImpulses | None:
    """Return integrated impulses for a trace's wrench field, if present."""

    if trace.wrench is None:
        return None
    return compute_wrench_impulses(trace.t, trace.wrench)


def trace_with_wrench_trace(trace: Trace, wrench_trace: WrenchTrace) -> Trace:
    """Return a copy of ``trace`` with its canonical wrench field populated."""

    if np.asarray(wrench_trace.time).shape != trace.t.shape or not np.allclose(
        wrench_trace.time, trace.t
    ):
        raise ValueError("wrench_trace.time must match trace.t")
    return Trace(
        t=trace.t,
        q=trace.q,
        v=trace.v,
        u=trace.u,
        dt=trace.dt,
        backend=trace.backend,
        meta=trace.meta,
        schema_version=trace.schema_version,
        torques=trace.torques,
        wrench=wrench_array_from_trace(wrench_trace),
        markers=trace.markers,
        contacts=trace.contacts,
    )


def static_support_wrench_trace(
    time: object,
    *,
    body_mass_kg: float,
    gravity_m_s2: float = 9.80665,
) -> WrenchTrace:
    """Known-case GRF trace: vertical support force equals body weight."""

    if body_mass_kg <= 0.0:
        raise ValueError("body_mass_kg must be positive")
    if gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be positive")
    time_arr = _time_vector(time)
    force = np.zeros((time_arr.size, 3), dtype=float)
    force[:, 2] = body_mass_kg * gravity_m_s2
    torque = np.zeros_like(force)
    return WrenchTrace(time_arr, force, torque)
