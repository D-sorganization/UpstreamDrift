"""Engine-agnostic GRF/wrench extraction utilities (CC-25, #6798).

Bridges the canonical ``Trace.wrench`` field (CC-4) with analysis primitives
for ground-reaction force and contact-wrench data.

The ``Trace.wrench`` field follows the CC-4 layout::

    [fx, fy, fz, tx, ty, tz]   in N and N·m (world frame)

Functions in this module let callers:

* **Pack** separate force/torque arrays into the ``(T, 6)`` wrench field
  understood by :class:`~simulation_backends.protocol.Trace`.
* **Unpack** a wrench array back into force and torque components.
* **Integrate** impulses from a wrench trace or directly from a
  :class:`~simulation_backends.protocol.Trace`.

No domain-specific (bunkershot3d) imports are used; the module operates
purely on NumPy arrays so any engine can feed it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import Trace

__all__ = [
    "WrenchImpulses",
    "compute_wrench_impulses",
    "force_torque_from_wrench_array",
    "trace_wrench_impulses",
    "wrench_array_from_force_torque",
]


@dataclass(frozen=True)
class WrenchImpulses:
    """Linear and angular impulses integrated over a wrench trace.

    Attributes:
        linear_impulse: Time-integrated force, shape ``(3,)`` [N·s].
        angular_impulse: Time-integrated torque, shape ``(3,)`` [N·m·s].
    """

    linear_impulse: np.ndarray
    angular_impulse: np.ndarray


def wrench_array_from_force_torque(
    force: np.ndarray,
    torque: np.ndarray,
) -> np.ndarray:
    """Pack separate force and torque arrays into a ``(T, 6)`` wrench array.

    The layout matches the CC-4 ``Trace.wrench`` convention:
    ``[fx, fy, fz, tx, ty, tz]``.

    Args:
        force: World-frame forces, shape ``(T, 3)`` [N].
        torque: World-frame torques, shape ``(T, 3)`` [N·m].

    Returns:
        Combined wrench, shape ``(T, 6)``.

    Raises:
        ValueError: If *force* or *torque* do not have shape ``(T, 3)``,
            or if their time dimensions disagree.
    """
    force = np.asarray(force, dtype=float)
    torque = np.asarray(torque, dtype=float)
    if force.ndim != 2 or force.shape[1] != 3:
        raise ValueError(f"force must have shape (T, 3), got {force.shape}")
    if torque.ndim != 2 or torque.shape[1] != 3:
        raise ValueError(f"torque must have shape (T, 3), got {torque.shape}")
    if force.shape[0] != torque.shape[0]:
        raise ValueError(
            "force and torque must have the same number of timesteps; "
            f"got {force.shape[0]} vs {torque.shape[0]}"
        )
    return np.concatenate([force, torque], axis=1)


def force_torque_from_wrench_array(
    wrench: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a ``(T, 6)`` wrench array into separate force and torque arrays.

    Args:
        wrench: Combined wrench, shape ``(T, 6)``
            ``[fx, fy, fz, tx, ty, tz]``.

    Returns:
        Tuple ``(force, torque)`` each of shape ``(T, 3)``.

    Raises:
        ValueError: If *wrench* does not have shape ``(T, 6)``.
    """
    wrench = np.asarray(wrench, dtype=float)
    if wrench.ndim != 2 or wrench.shape[1] != 6:
        raise ValueError(f"wrench must have shape (T, 6), got {wrench.shape}")
    return wrench[:, :3], wrench[:, 3:]


def compute_wrench_impulses(
    time: np.ndarray,
    wrench: np.ndarray,
) -> WrenchImpulses:
    """Integrate a wrench trace to compute linear and angular impulses.

    Uses the trapezoidal rule for integration, matching the convention in
    ``WrenchTrace.get_impulses`` (bunkershot3d).

    Args:
        time: Sample times, shape ``(T,)`` [s].  Should be strictly
            increasing.
        wrench: Wrench array, shape ``(T, 6)``
            ``[fx, fy, fz, tx, ty, tz]``.

    Returns:
        :class:`WrenchImpulses` with ``linear_impulse`` (3,) [N·s]
        and ``angular_impulse`` (3,) [N·m·s].

    Raises:
        ValueError: If *time* and *wrench* disagree in length, or if
            *wrench* does not have shape ``(T, 6)``.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    wrench = np.asarray(wrench, dtype=float)
    if wrench.ndim != 2 or wrench.shape[1] != 6:
        raise ValueError(f"wrench must have shape (T, 6), got {wrench.shape}")
    if time.shape[0] != wrench.shape[0]:
        raise ValueError(
            f"time length {time.shape[0]} does not match wrench rows {wrench.shape[0]}"
        )
    force, torque = force_torque_from_wrench_array(wrench)
    linear_impulse = np.trapezoid(force, x=time, axis=0)
    angular_impulse = np.trapezoid(torque, x=time, axis=0)
    return WrenchImpulses(
        linear_impulse=linear_impulse,
        angular_impulse=angular_impulse,
    )


def trace_wrench_impulses(trace: Trace) -> WrenchImpulses | None:
    """Return wrench impulses from a :class:`~.protocol.Trace`, or ``None``.

    Convenience wrapper that extracts the wrench field and delegates to
    :func:`compute_wrench_impulses`.

    Args:
        trace: Simulation trace with optional wrench data.

    Returns:
        :class:`WrenchImpulses` if ``trace.wrench`` is not ``None``,
        otherwise ``None``.
    """
    if trace.wrench is None:
        return None
    return compute_wrench_impulses(trace.t, trace.wrench)
