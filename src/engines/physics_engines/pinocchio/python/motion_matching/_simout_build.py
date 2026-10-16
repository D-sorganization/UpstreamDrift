"""Argument normalisation for the Pinocchio ``SimOut`` dataclass (issue #12843).

``SimOut`` historically accepted a 14-field canonical signature plus five
keyword-only legacy aliases (``t``, ``grip_position``, ``grip_rotation``,
``clubhead_position``, ``clubhead_rotation``) and filled every omitted array
with zeros. Holding that in a 19-parameter ``__init__`` blew the architecture
budget, so the signature is now data (:data:`SIMOUT_SIGNATURE`) and the
normalisation lives here as small single-purpose helpers.

This module deliberately imports nothing from ``simulate`` so ``simulate``
can import it without a cycle.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.math_utils.quaternion import rotmat_to_quat

__all__ = ["SIMOUT_SIGNATURE", "normalize_simout_args"]

_P = inspect.Parameter

#: Canonical positional-or-keyword parameters, in field order, with defaults.
_CANONICAL: tuple[tuple[str, Any], ...] = (
    ("time", None),
    ("q", None),
    ("qd", None),
    ("qdd", None),
    ("tau", None),
    ("grip", None),
    ("grip_quat", None),
    ("clubhead", None),
    ("club_quat", None),
    ("solver_status", "success"),
    ("duration_s", 0.0),
    ("kinetic_energy", None),
    ("potential_energy", None),
    ("meta", None),
)

#: Keyword-only legacy aliases accepted for backward compatibility.
_ALIASES: tuple[str, ...] = (
    "t",
    "grip_position",
    "grip_rotation",
    "clubhead_position",
    "clubhead_rotation",
)

SIMOUT_SIGNATURE: inspect.Signature = inspect.Signature(
    [_P(name, _P.POSITIONAL_OR_KEYWORD, default=d) for name, d in _CANONICAL]
    + [_P(name, _P.KEYWORD_ONLY, default=None) for name in _ALIASES]
)


def _f64(value: Any) -> npt.NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _time_vector(args: Mapping[str, Any]) -> npt.NDArray[np.float64]:
    """Return the time axis from ``time`` (preferred) or the ``t`` alias."""
    t_arr = args["time"] if args["time"] is not None else args["t"]
    if t_arr is None:
        raise ValueError("Either 'time' or 't' must be provided")
    return _f64(t_arr)


def _joint_space_arrays(
    args: Mapping[str, Any], n: int
) -> tuple[npt.NDArray[np.float64], ...]:
    """Return ``(q, qd, qdd, tau)``; ``qdd``/``tau`` default to ``nv`` zeros."""

    def _given_or_zeros(name: str, cols: int) -> npt.NDArray[np.float64]:
        if args[name] is not None:
            return _f64(args[name])
        return np.zeros((n, cols), dtype=np.float64)

    q_mat = _given_or_zeros("q", 0)
    qd_mat = _given_or_zeros("qd", 0)
    nv = qd_mat.shape[1] if qd_mat.ndim == 2 else 0
    return q_mat, qd_mat, _given_or_zeros("qdd", nv), _given_or_zeros("tau", nv)


def _pose_array(
    args: Mapping[str, Any],
    primary: str,
    position_alias: str | None,
    rotation_alias: str | None,
    shape: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Resolve a position/quaternion field from canonical, alias or default."""
    if args[primary] is not None:
        return _f64(args[primary])
    if position_alias is not None and args[position_alias] is not None:
        return _f64(args[position_alias])
    if rotation_alias is not None and args[rotation_alias] is not None:
        return rotmat_to_quat(args[rotation_alias])
    return np.zeros(shape, dtype=np.float64)


def _pose_arrays(
    args: Mapping[str, Any], n: int
) -> tuple[npt.NDArray[np.float64], ...]:
    """Return ``(grip, grip_quat, clubhead, club_quat)``."""
    return (
        _pose_array(args, "grip", "grip_position", None, (n, 3)),
        _pose_array(args, "grip_quat", None, "grip_rotation", (n, 4)),
        _pose_array(args, "clubhead", "clubhead_position", None, (n, 3)),
        _pose_array(args, "club_quat", None, "clubhead_rotation", (n, 4)),
    )


def _energy_vector(value: Any, n: int) -> npt.NDArray[np.float64]:
    return _f64(value) if value is not None else np.zeros(n, dtype=np.float64)


def normalize_simout_args(
    args: Sequence[Any], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind ``SimOut(*args, **kwargs)`` and return the canonical field values.

    Raises:
        TypeError: On unknown keywords or too many positional arguments
            (same contract as the former explicit ``__init__``).
        ValueError: If neither ``time`` nor ``t`` is supplied.
    """
    bound = SIMOUT_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    given: Mapping[str, Any] = bound.arguments
    t_vec = _time_vector(given)
    n = t_vec.shape[0]
    q_mat, qd_mat, qdd_mat, tau_mat = _joint_space_arrays(given, n)
    grip, grip_quat, clubhead, club_quat = _pose_arrays(given, n)
    return {
        "time": t_vec,
        "q": q_mat,
        "qd": qd_mat,
        "qdd": qdd_mat,
        "tau": tau_mat,
        "grip": grip,
        "grip_quat": grip_quat,
        "clubhead": clubhead,
        "club_quat": club_quat,
        "solver_status": given["solver_status"],
        "duration_s": float(given["duration_s"]),
        "kinetic_energy": _energy_vector(given["kinetic_energy"], n),
        "potential_energy": _energy_vector(given["potential_energy"], n),
        "meta": given["meta"] if given["meta"] is not None else {},
    }
