"""Simscape logged force/torque channels to a ForceTorqueSeries (#11303, FTO-18).

Pure Python: reads the exported dataset CSV columns
(``<Group>Logs_<Signal>_<1|2|3>``, ``_I11.._I33`` for 3x3 matrices) and needs
no MATLAB. The channel table :data:`SIMSCAPE_FORCE_CHANNELS` is the single
declarative definition of what is drawable.

Rotation convention (matches ``calculateForceMoments.m``)::

    R = [[I11, I12, I13], [I21, I22, I23], [I31, I32, I33]]
    v_world = R @ v_local

Unavailable data is ``None``, never zero. A joint-local channel whose logged
rotation is missing is unavailable, not drawn unrotated.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np

from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
)

logger = logging.getLogger(__name__)

_Column: TypeAlias = Sequence[Any] | np.ndarray

__all__ = [
    "DEFAULT_ROTATION_TOL",
    "SIMSCAPE_FORCE_CHANNELS",
    "SIMSCAPE_JOINTS",
    "ChannelSpec",
    "force_series_from_columns",
    "load_simscape_force_series",
]

#: Joint chain, defined once. Joints with local force channels and a logged R.
SIMSCAPE_JOINTS: tuple[str, ...] = (
    "LScap",
    "RScap",
    "LS",
    "RS",
    "LF",
    "RF",
    "Spine",
    "Torso",
)

#: Orthonormality tolerance for logged rotations (det +1, R^T R = I).
DEFAULT_ROTATION_TOL = 1e-6

_SOURCE = "simscape_csv"
_ENGINE = "simscape"
_VEC = ("1", "2", "3")
_XYZ = ("X", "Y", "Z")


@dataclass(frozen=True)
class ChannelSpec:
    """One declarative force/torque channel of the Simscape dataset."""

    label: str
    kind: WrenchKind
    body: str
    force_cols: tuple[str, str, str] | None
    # A None entry is an axis the joint has no actuator about (exact zero).
    torque_cols: tuple[str | None, str | None, str | None] | None
    point_cols: tuple[str, str, str]
    frame: Literal["world", "joint_local"]
    rotation_prefix: str | None = None

    def __post_init__(self) -> None:
        if self.force_cols is None and self.torque_cols is None:
            raise ValueError(f"{self.label}: needs force or torque columns")
        if self.frame == "joint_local" and not self.rotation_prefix:
            raise ValueError(f"{self.label}: joint_local needs rotation_prefix")


def _cols(prefix: str, suffixes: tuple[str, ...] = _VEC) -> tuple[str, str, str]:
    a, b, c = (f"{prefix}{s}" for s in suffixes)
    return (a, b, c)


#: Actuated local axes per joint, from calculateJointPowerWork.m::
#: getActuatorTorques. Torso is a scalar field with no defined axis, so it
#: has no actuator channel here.
_ACTUATOR_AXES: dict[str, str] = {
    "LScap": "XY",
    "RScap": "XY",
    "LS": "XYZ",
    "RS": "XYZ",
    "LF": "Z",
    "RF": "Z",
    "Spine": "XY",
}


def _actuator_cols(joint: str) -> tuple[str | None, str | None, str | None]:
    axes = _ACTUATOR_AXES[joint]
    a, b, c = (f"{joint}Logs_ActuatorTorque{x}" if x in axes else None for x in _XYZ)
    return (a, b, c)


def _joint_specs(joint: str) -> tuple[ChannelSpec, ...]:
    common = {
        "body": joint,
        "point_cols": _cols(f"{joint}Logs_GlobalPosition_"),
        "frame": "joint_local",
        "rotation_prefix": f"{joint}Logs_Rotation_Transform",
    }
    return (
        ChannelSpec(
            label=f"joint_reaction:{joint}",
            kind=WrenchKind.JOINT_REACTION,
            force_cols=_cols(f"{joint}Logs_ConstraintForceLocal_"),
            torque_cols=_cols(f"{joint}Logs_ConstraintTorqueLocal_"),
            **common,  # type: ignore[arg-type]
        ),
        ChannelSpec(
            label=f"joint_total:{joint}",
            kind=WrenchKind.EXTERNAL,
            force_cols=_cols(f"{joint}Logs_ForceLocal_"),
            torque_cols=_cols(f"{joint}Logs_TorqueLocal_"),
            **common,  # type: ignore[arg-type]
        ),
        *(
            [
                ChannelSpec(
                    label=f"joint_actuator:{joint}",
                    kind=WrenchKind.JOINT_ACTUATOR,
                    force_cols=None,
                    torque_cols=_actuator_cols(joint),
                    **common,  # type: ignore[arg-type]
                )
            ]
            if joint in _ACTUATOR_AXES
            else []
        ),
    )


_MP = _cols("MidpointCalcsLogs_MPGlobalPosition_")

SIMSCAPE_FORCE_CHANNELS: tuple[ChannelSpec, ...] = (
    *(spec for joint in SIMSCAPE_JOINTS for spec in _joint_specs(joint)),
    ChannelSpec(
        label="external:base_on_hip",
        kind=WrenchKind.EXTERNAL,
        body="pelvis",
        # Global columns only: BaseonHipForceHipBase is a different frame.
        force_cols=_cols("HipLogs_BaseonHipForceGlobal_"),
        torque_cols=_cols("HipLogs_BaseonHipTorqueGlobal_"),
        point_cols=_cols("HipLogs_HipGlobalPosition_dim"),
        frame="world",
    ),
    ChannelSpec(
        label="grip:total_hand",
        kind=WrenchKind.GRIP,
        body="club",
        force_cols=_cols("CalculatedSignalsLogs_TotalHandForceGlobal_"),
        torque_cols=_cols("CalculatedSignalsLogs_TotalHandTorqueGlobal_"),
        point_cols=_MP,
        frame="world",
    ),
    # Per-hand loading of the hand ON the club (#11715), as in grip_wrench.
    ChannelSpec(
        label="grip:hand_left",
        kind=WrenchKind.GRIP,
        body="club",
        force_cols=_cols("LWLogs_LHonClubFGlobal_"),
        torque_cols=_cols("LWLogs_LHonClubTGlobal_"),
        point_cols=_cols("LWLogs_LHGlobalPosition_"),
        frame="world",
    ),
    ChannelSpec(
        label="grip:hand_right",
        kind=WrenchKind.GRIP,
        body="club",
        force_cols=_cols("RWLogs_RHonClubFGlobal_"),
        torque_cols=_cols("RWLogs_RHonClubTGlobal_"),
        point_cols=_cols("RWLogs_RHGlobalPosition_"),
        frame="world",
    ),
    ChannelSpec(
        label="grip:lh_mof",
        kind=WrenchKind.GRIP,
        body="club",
        force_cols=None,
        torque_cols=_cols("MomentandCoupleLogs_LHMOFonClubGlobal_"),
        point_cols=_cols("LWLogs_LHGlobalPosition_"),
        frame="world",
    ),
    ChannelSpec(
        label="grip:rh_mof",
        kind=WrenchKind.GRIP,
        body="club",
        force_cols=None,
        torque_cols=_cols("MomentandCoupleLogs_RHMOFonClubGlobal_"),
        point_cols=_cols("RWLogs_RHGlobalPosition_"),
        frame="world",
    ),
    ChannelSpec(
        label="grip:midpoint_couple",
        kind=WrenchKind.GRIP,
        body="club",
        force_cols=None,
        torque_cols=_cols("MomentandCoupleLogs_EquivalentMidpointCoupleGlobal_"),
        point_cols=_MP,
        frame="world",
    ),
)


def _read_columns(path: Path) -> dict[str, list[str]]:
    """Read a dataset CSV into column name -> raw cell strings."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{path}: empty CSV")
        columns: dict[str, list[str]] = {name: [] for name in header}
        for row in reader:
            for name, cell in zip(header, row, strict=False):
                columns[name].append(cell)
    return columns


def _floats(cells: _Column, name: str) -> np.ndarray:
    try:
        arr = np.asarray([float(c) for c in cells], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"column {name}: non-numeric cell") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"column {name}: non-finite value")
    return arr


def _vec_array(
    columns: Mapping[str, _Column], names: tuple[str | None, ...]
) -> np.ndarray | None:
    """Stack three columns to (T, 3), or None when a named column is absent.

    A ``None`` name is an undriven axis and contributes exact zeros.
    """
    if not all(n is None or n in columns for n in names):
        return None
    rows = len(columns["time"])
    return np.stack(
        [np.zeros(rows) if n is None else _floats(columns[n], n) for n in names],
        axis=1,
    )


def _rotation_array(
    columns: Mapping[str, _Column], prefix: str, tol: float
) -> np.ndarray | None:
    """Return (T, 3, 3) R, validated orthonormal, or None when absent."""
    names = [[f"{prefix}_I{i}{j}" for j in (1, 2, 3)] for i in (1, 2, 3)]
    if not all(n in columns for r in names for n in r):
        return None
    rot = np.stack(
        [np.stack([_floats(columns[n], n) for n in r], axis=1) for r in names],
        axis=1,
    )
    joint = prefix.split("Logs_", 1)[0]
    ortho_err = np.abs(np.einsum("tji,tjk->tik", rot, rot) - np.eye(3)).max(axis=(1, 2))
    det_err = np.abs(np.linalg.det(rot) - 1.0)
    bad = np.flatnonzero((ortho_err > tol) | (det_err > tol))
    if bad.size:
        row = int(bad[0])
        raise ValueError(
            f"joint {joint}: rotation not orthonormal at row {row} "
            f"(|R^T R - I|={ortho_err[row]:.3e}, |det-1|={det_err[row]:.3e}, "
            f"tol={tol:g})"
        )
    return rot


def _to_world(rot: np.ndarray, local: np.ndarray) -> np.ndarray:
    """v_world = R @ v_local for every row."""
    return np.einsum("tij,tj->ti", rot, local)


def _half(
    columns: Mapping[str, _Column],
    spec: ChannelSpec,
    names: tuple[str | None, ...] | None,
    rotations: dict[str, np.ndarray | None],
) -> np.ndarray | None:
    if names is None:
        return None
    vec = _vec_array(columns, names)
    if vec is None or spec.frame == "world":
        return vec
    assert spec.rotation_prefix is not None  # guaranteed by ChannelSpec
    rot = rotations[spec.rotation_prefix]
    return None if rot is None else _to_world(rot, vec)


def _tuple3(row: np.ndarray) -> tuple[float, float, float]:
    return (float(row[0]), float(row[1]), float(row[2]))


def load_simscape_force_series(
    csv_path: str | Path, *, rotation_tol: float = DEFAULT_ROTATION_TOL
) -> tuple[ForceTorqueSeries, tuple[str, ...]]:
    """Load a Simscape dataset CSV as a world-frame ``ForceTorqueSeries``.

    Preconditions: ``csv_path`` is a dataset CSV with a ``time`` column and
    at least one row; ``rotation_tol`` is positive and finite.

    Postconditions: see :func:`force_series_from_columns`.

    Raises:
        TypeError: ``csv_path`` is not a path.
        ValueError: bad tolerance, no ``time`` column or rows, non-finite
            data, or a logged rotation that is not orthonormal (names the
            joint and row).
    """
    if not isinstance(csv_path, str | Path):
        raise TypeError("csv_path must be str or Path")
    path = Path(csv_path)
    return force_series_from_columns(
        _read_columns(path), rotation_tol=rotation_tol, source_name=path.name
    )


def force_series_from_columns(
    columns: Mapping[str, _Column],
    *,
    rotation_tol: float = DEFAULT_ROTATION_TOL,
    source_name: str = "columns",
    wrench_source: str = _SOURCE,
) -> tuple[ForceTorqueSeries, tuple[str, ...]]:
    """Build a world-frame ``ForceTorqueSeries`` from a column mapping.

    This is the single loader core shared by the CSV path and by
    :meth:`SimscapeOutput.to_force_series` (#11304). Keys are the dataset
    column names of :data:`SIMSCAPE_FORCE_CHANNELS`; values are per-sample
    sequences (numbers, or numeric strings from a CSV).

    Preconditions: ``columns`` has a ``time`` column with at least one row;
    ``rotation_tol`` is positive and finite; ``wrench_source`` is a non-empty
    provenance label stamped on every wrench (CSV: ``simscape_csv``).

    Postconditions: times are strictly increasing; every wrench is in the
    world frame (Z-up, SI). Each unavailable half is ``None`` and listed in
    the returned ``missing`` tuple as ``"<label>:force"`` / ``"<label>:torque"``.
    A joint-local channel with missing rotation columns is unavailable.

    Raises:
        ValueError: bad tolerance, no ``time`` column or rows, non-finite
            data, or a logged rotation that is not orthonormal.
    """
    if not (math.isfinite(rotation_tol) and rotation_tol > 0.0):
        raise ValueError("rotation_tol must be positive and finite")
    if "time" not in columns or len(columns["time"]) == 0:
        raise ValueError(f"{source_name}: needs a 'time' column with at least one row")
    times = _floats(columns["time"], "time")

    prefixes = {s.rotation_prefix for s in SIMSCAPE_FORCE_CHANNELS if s.rotation_prefix}
    rotations = {p: _rotation_array(columns, p, rotation_tol) for p in sorted(prefixes)}

    per_spec: list[
        tuple[ChannelSpec, np.ndarray | None, np.ndarray | None, np.ndarray | None]
    ] = []
    missing: list[str] = []
    for spec in SIMSCAPE_FORCE_CHANNELS:
        point = _vec_array(columns, spec.point_cols)
        force = (
            _half(columns, spec, spec.force_cols, rotations)
            if point is not None
            else None
        )
        torque = (
            _half(columns, spec, spec.torque_cols, rotations)
            if point is not None
            else None
        )
        if spec.force_cols is not None and force is None:
            missing.append(f"{spec.label}:force")
        if spec.torque_cols is not None and torque is None:
            missing.append(f"{spec.label}:torque")
        per_spec.append((spec, point, force, torque))

    frames = []
    for i, t in enumerate(times):
        wrenches = []
        for spec, point, force, torque in per_spec:
            if point is None or (force is None and torque is None):
                continue
            wrenches.append(
                OverlayWrench(
                    kind=spec.kind,
                    label=spec.label,
                    body=spec.body,
                    point_m=_tuple3(point[i]),
                    force_n=None if force is None else _tuple3(force[i]),
                    torque_nm=None if torque is None else _tuple3(torque[i]),
                    source=wrench_source,
                )
            )
        frames.append(
            ForceTorqueFrame(time_s=float(t), engine=_ENGINE, wrenches=tuple(wrenches))
        )
    if missing:
        logger.info(
            "Simscape force channels unavailable in %s: %s", source_name, missing
        )
    return ForceTorqueSeries(frames=tuple(frames), engine=_ENGINE), tuple(missing)
