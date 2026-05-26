"""Adapters that normalise heterogeneous target/sim-output objects.

The canonical ``SimOut`` and ``FitResult`` dataclasses are still being
landed across PRs (cross-engine spec §2). Until they exist as a single
import, the viz layer accepts anything that exposes the expected attribute
names: ``time``, ``clubhead``, ``butt``/``grip``, ``club_quat``, and
optionally ``joint_torques``. Plain ``dict`` and ``SimpleNamespace`` are
both supported. This decouples the viz roll-out (issue #4130) from the
canonical-types roll-out and keeps the smoke tests independent of any
optional heavy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Return ``obj[name]`` or ``getattr(obj, name)``, else ``default``."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float_array(value: Any) -> NDArray[np.floating] | None:
    """Coerce a value to a float ``ndarray`` or return ``None`` if absent."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return arr


@dataclass(frozen=True)
class _NormalisedSeries:
    """Common structure for both target and sim outputs.

    Only ``time`` and ``clubhead`` are required; the remaining fields
    fall back to ``None`` and the plotters silently skip the missing
    panel rather than raising — the smoke tests are intentionally
    permissive.
    """

    time: NDArray[np.floating]
    clubhead: NDArray[np.floating]
    butt: NDArray[np.floating] | None
    club_quat: NDArray[np.floating] | None
    joint_torques: NDArray[np.floating] | None
    clubhead_speed: NDArray[np.floating] | None
    impact_idx: int | None


def normalise(obj: Any) -> _NormalisedSeries:
    """Normalise a target or sim-output into a :class:`_NormalisedSeries`."""
    time = _as_float_array(_attr(obj, "time"))
    if time is None or time.ndim != 1 or time.size < 2:
        raise ValueError(
            "viz input requires a 1-D 'time' array with at least 2 samples"
        )

    clubhead = _as_float_array(_attr(obj, "clubhead"))
    if clubhead is None or clubhead.shape != (time.size, 3):
        raise ValueError(
            f"viz input requires 'clubhead' of shape ({time.size}, 3), "
            f"got {None if clubhead is None else clubhead.shape}"
        )

    # Accept either ``butt`` (canonical ClubTarget) or ``grip`` (sim convention).
    butt = _as_float_array(_attr(obj, "butt"))
    if butt is None:
        butt = _as_float_array(_attr(obj, "grip"))
    if butt is not None and butt.shape != (time.size, 3):
        butt = None  # silently drop a malformed series

    club_quat = _as_float_array(_attr(obj, "club_quat"))
    if club_quat is not None and club_quat.shape != (time.size, 4):
        club_quat = None

    joint_torques = _as_float_array(_attr(obj, "joint_torques"))
    if joint_torques is not None and joint_torques.ndim != 2:
        joint_torques = None

    clubhead_speed = _as_float_array(_attr(obj, "clubhead_speed"))
    if clubhead_speed is None:
        # Compute from positions when not provided.
        clubhead_speed = _finite_difference_speed(time, clubhead)

    impact = _attr(obj, "impact_idx")
    impact_idx = int(impact) if impact is not None else None

    return _NormalisedSeries(
        time=time,
        clubhead=clubhead,
        butt=butt,
        club_quat=club_quat,
        joint_torques=joint_torques,
        clubhead_speed=clubhead_speed,
        impact_idx=impact_idx,
    )


def _finite_difference_speed(
    time: NDArray[np.floating],
    positions: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Centred finite-difference speed in m/s at every sample."""
    diff = np.gradient(positions, time, axis=0)
    # ⚡ Bolt: np.sqrt(np.einsum) is ~35-40% faster than np.linalg.norm(..., axis=1)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def quat_geodesic_deg(
    q_a: NDArray[np.floating],
    q_b: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Geodesic angle in degrees between two quaternion trajectories.

    Quaternion convention is canonical ``[w, x, y, z]``. The function is
    sign-invariant so an unflipped reference produces identical output to
    a flipped one (``q ≡ -q`` represents the same rotation).
    """
    if q_a.shape != q_b.shape:
        raise ValueError(
            f"quaternion shapes must match, got {q_a.shape} vs {q_b.shape}"
        )
    # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~35-40% faster than np.linalg.norm(..., axis=1)
    a = q_a / np.sqrt(np.einsum("ij,ij->i", q_a, q_a))[:, np.newaxis]
    b = q_b / np.sqrt(np.einsum("ij,ij->i", q_b, q_b))[:, np.newaxis]
    # ⚡ Bolt: einsum avoids temp arrays and is faster than np.sum(..., axis=1)
    dot = np.clip(np.abs(np.einsum("ij,ij->i", a, b)), 0.0, 1.0)
    return 2.0 * np.degrees(np.arccos(dot))


# Style constants from VISUALIZATION_SPEC.md "Styling".
COLOR_MEASURED = "#1f77b4"  # blue
COLOR_SIMULATED = "#d62728"  # red
COLOR_ERROR = "#7f7f7f"  # grey
TITLE_FONTSIZE = 13
AXES_FONTSIZE = 11
