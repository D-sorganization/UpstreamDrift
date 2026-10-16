"""Argument normalisation and validation for the Drake ``SimOut`` (issue #12843).

``SimOut`` accepts the 14 canonical fields plus six keyword-only legacy
aliases (``metadata``, ``t``, ``grip_position``, ``grip_rotation``,
``clubhead_position``, ``clubhead_rotation``). Omitted pose / energy arrays are
NaN-filled (never zero) so a missing measurement cannot masquerade as a real
one (issue #10960). The signature is data (:data:`SIMOUT_SIGNATURE`) and the
checks live here as small single-purpose helpers so ``SimOut`` itself can be a
plain frozen dataclass like the MuJoCo one.

This module imports nothing from ``simulate`` (no import cycle).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.math_utils.quaternion import rotmat_to_quat

__all__ = ["SIMOUT_SIGNATURE", "VALID_SOLVER_STATUSES", "normalize_simout_args"]

_P = inspect.Parameter

VALID_SOLVER_STATUSES: frozenset[str] = frozenset(
    {"success", "warning", "failed", "partial"}
)

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

_KEYWORD_ONLY: tuple[str, ...] = (
    "metadata",
    "t",
    "grip_position",
    "grip_rotation",
    "clubhead_position",
    "clubhead_rotation",
)

SIMOUT_SIGNATURE: inspect.Signature = inspect.Signature(
    [_P(name, _P.POSITIONAL_OR_KEYWORD, default=d) for name, d in _CANONICAL]
    + [_P(name, _P.KEYWORD_ONLY, default=None) for name in _KEYWORD_ONLY]
)


def _f64(value: Any) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _time_vector(args: Mapping[str, Any]) -> NDArray[np.float64]:
    """Return the 1-D time axis from ``time`` (preferred) or ``t``."""
    t_arr = args["time"] if args["time"] is not None else args["t"]
    if t_arr is None:
        raise ValueError("Either 'time' or 't' must be provided")
    t_vec = _f64(t_arr)
    if t_vec.ndim != 1:
        msg = f"SimOut.time must be 1-D; got shape {t_vec.shape}"
        raise ValueError(msg)
    return t_vec


def _joint_space_arrays(
    args: Mapping[str, Any], n: int
) -> dict[str, NDArray[np.float64]]:
    """Return ``q/qd/qdd/tau``; ``qdd``/``tau`` default to ``nv`` zeros."""

    def _given_or_zeros(name: str, cols: int) -> NDArray[np.float64]:
        if args[name] is not None:
            return _f64(args[name])
        return np.zeros((n, cols), dtype=np.float64)

    q_mat = _given_or_zeros("q", 0)
    qd_mat = _given_or_zeros("qd", 0)
    nv = qd_mat.shape[1] if qd_mat.ndim == 2 else 0
    return {
        "q": q_mat,
        "qd": qd_mat,
        "qdd": _given_or_zeros("qdd", nv),
        "tau": _given_or_zeros("tau", nv),
    }


def _pose_array(
    args: Mapping[str, Any],
    primary: str,
    position_alias: str | None,
    rotation_alias: str | None,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Resolve a pose field from canonical, alias or NaN default."""
    if args[primary] is not None:
        return _f64(args[primary])
    if position_alias is not None and args[position_alias] is not None:
        return _f64(args[position_alias])
    if rotation_alias is not None and args[rotation_alias] is not None:
        return rotmat_to_quat(_f64(args[rotation_alias]))
    return np.full(shape, np.nan, dtype=np.float64)


def _pose_arrays(args: Mapping[str, Any], n: int) -> dict[str, NDArray[np.float64]]:
    """Return ``grip / grip_quat / clubhead / club_quat``."""
    spec = (
        ("grip", "grip_position", None, (n, 3)),
        ("grip_quat", None, "grip_rotation", (n, 4)),
        ("clubhead", "clubhead_position", None, (n, 3)),
        ("club_quat", None, "clubhead_rotation", (n, 4)),
    )
    return {row[0]: _pose_array(args, *row) for row in spec}


def _energy_vector(value: Any, n: int) -> NDArray[np.float64]:
    return _f64(value) if value is not None else np.full(n, np.nan, dtype=np.float64)


def _resolve_meta(meta: Any, metadata: Any) -> Any:
    """``meta`` wins over the ``metadata`` alias; default is a fresh dict."""
    if meta is not None:
        return meta
    return metadata if metadata is not None else {}


def _check_array_shapes(fields: Mapping[str, Any], n: int) -> None:
    """Every trajectory array must be ``(N, ...)`` with canonical widths."""
    widths = {"grip": 3, "grip_quat": 4, "clubhead": 3, "club_quat": 4}
    names = ("q", "qd", "qdd", "tau", "grip", "grip_quat", "clubhead", "club_quat")
    for name in names:
        arr = fields[name]
        if arr.ndim != 2 or arr.shape[0] != n:
            msg = f"SimOut.{name} must have shape (N={n}, ...); got {arr.shape}"
            raise ValueError(msg)
        cols = widths.get(name)
        if cols is not None and arr.shape[1] != cols:
            msg = f"SimOut.{name} must have shape (N, {cols}); got {arr.shape}"
            raise ValueError(msg)


def _check_status_and_energy(fields: Mapping[str, Any], n: int) -> None:
    status = fields["solver_status"]
    if status not in VALID_SOLVER_STATUSES:
        msg = (
            "SimOut.solver_status must be 'success' / 'warning' / 'failed' / 'partial'; "
            f"got {status!r}"
        )
        raise ValueError(msg)
    for name in ("kinetic_energy", "potential_energy"):
        vec = fields[name]
        if vec.ndim != 1 or vec.shape[0] != n:
            raise ValueError(f"SimOut.{name} must have shape ({n},); got {vec.shape}")


def normalize_simout_args(
    args: Sequence[Any], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind ``SimOut(*args, **kwargs)``; return validated canonical fields.

    Raises:
        TypeError: On unknown keywords / too many positionals.
        ValueError: On a missing or non-1-D time axis, a mis-shaped array, an
            unknown ``solver_status`` or mis-shaped energy vectors.
    """
    bound = SIMOUT_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    given: Mapping[str, Any] = bound.arguments
    t_vec = _time_vector(given)
    n = t_vec.shape[0]
    fields: dict[str, Any] = {"time": t_vec}
    fields.update(_joint_space_arrays(given, n))
    fields.update(_pose_arrays(given, n))
    _check_array_shapes(fields, n)
    fields["solver_status"] = given["solver_status"]
    fields["duration_s"] = float(given["duration_s"])
    fields["kinetic_energy"] = _energy_vector(given["kinetic_energy"], n)
    fields["potential_energy"] = _energy_vector(given["potential_energy"], n)
    _check_status_and_energy(fields, n)
    fields["meta"] = _resolve_meta(given["meta"], given["metadata"])
    return fields
