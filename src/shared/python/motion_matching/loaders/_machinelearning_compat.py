"""Bidirectional adapter: MachineLearning DataFrame schema <-> canonical ClubTarget.

PR #3966's MachineLearning/ workflow uses two column conventions:

* ``clubface_x/y/z``, ``clubface_vx/vy/vz``, ``clubface_ax/ay/az`` -- the
  measured target produced by ``prepare_club_target_trajectory.py`` from the
  Wiffle workbook. Position is the clubhead/clubface; orientation and the
  butt-end position are not represented.
* ``ClubLogs_CHGlobalPosition_{1,2,3}``, ``ClubLogs_CHGlobalVelocity_{1,2,3}``,
  ``ClubLogs_CHGlobalAcceleration_{1,2,3}`` -- Simscape sim output
  (``extract_dynamics_dataset.py``). Same physical quantity (clubhead in the
  global frame) under different column names; orientation and butt are
  similarly absent.

The scaffold's canonical schema is the frozen :class:`ClubTarget` dataclass
with ``butt``, ``clubhead``, ``club_quat``, ``time``, ``impact_idx`` and
``source``.

LOSSY ROUND TRIPS
-----------------
Neither MachineLearning convention carries the butt-end position or the
orientation quaternion, so:

* ``ClubTarget -> clubface_*`` -> ``ClubTarget`` round-trips drop ``butt`` and
  ``club_quat``. The reconstructed target uses ``butt = clubhead - shaft *
  v_hat`` (or ``NaN`` where velocity is undefined) and a tangent-frame
  quaternion derived from velocity / acceleration. Both are recovered values,
  not the originals.
* ``ClubTarget -> ClubLogs_*`` -> ``ClubTarget`` is identical except for the
  column names; the same lossy fields apply.
* ``ClubTarget -> clubface/ClubLogs`` direction preserves all fields present
  in the flat schema (positions, derived velocities, derived accelerations,
  time) to within numpy's float64 round-off.

A ``UserWarning`` is emitted on every ``to_canonical_*`` call to remind the
caller that ``butt`` and ``club_quat`` are reconstructions.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.shared.python.core.contracts import postcondition, precondition

from ..club_target import ClubTarget, SourceProvenance
from ._quaternion import rotmat_to_quat

# Default geometry used to *recover* a butt position from a clubface position.
# 1.143 m is the canonical 45 in driver shaft length used elsewhere in the
# scaffold; tests pin against this exact value.
DEFAULT_SHAFT_LENGTH_M: float = 1.143

CLUBFACE_POSITION = ("clubface_x", "clubface_y", "clubface_z")
CLUBFACE_VELOCITY = ("clubface_vx", "clubface_vy", "clubface_vz")
CLUBFACE_ACCELERATION = ("clubface_ax", "clubface_ay", "clubface_az")

CLUBLOGS_POSITION = (
    "ClubLogs_CHGlobalPosition_1",
    "ClubLogs_CHGlobalPosition_2",
    "ClubLogs_CHGlobalPosition_3",
)
CLUBLOGS_VELOCITY = (
    "ClubLogs_CHGlobalVelocity_1",
    "ClubLogs_CHGlobalVelocity_2",
    "ClubLogs_CHGlobalVelocity_3",
)
CLUBLOGS_ACCELERATION = (
    "ClubLogs_CHGlobalAcceleration_1",
    "ClubLogs_CHGlobalAcceleration_2",
    "ClubLogs_CHGlobalAcceleration_3",
)

_LOSSY_MSG = (
    "MachineLearning -> ClubTarget conversion is lossy: butt and club_quat "
    "are reconstructed from the clubhead trajectory, not the original measured "
    "values."
)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Have: {sorted(df.columns.tolist())}"
        )


def _gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) > 1 and np.all(np.isfinite(time)) and len(np.unique(time)) > 1:
        return np.gradient(values, time, axis=0, edge_order=1)
    return np.full_like(values, np.nan, dtype=np.float64)


def _time_vector(df: pd.DataFrame, n: int) -> np.ndarray:
    if "time" in df.columns:
        return df["time"].to_numpy(dtype=np.float64)
    return np.arange(n, dtype=np.float64) / 1000.0


def _impact_idx(df: pd.DataFrame, default: int) -> int:
    if "impact_idx" in df.columns and len(df) > 0:
        return int(df["impact_idx"].iloc[0])
    return default


def _quat_from_velocity_acceleration(
    velocity: np.ndarray, acceleration: np.ndarray
) -> np.ndarray:
    """Build a tangent-frame rotation matrix per sample, return unit quats.

    The X-axis is taken along the velocity direction; the Y-axis is the
    component of acceleration orthogonal to X; Z = X x Y. Where the result is
    degenerate (zero velocity, parallel a/v), the row falls back to identity.
    """
    n = velocity.shape[0]
    rotmats = np.empty((n, 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for i in range(n):
        v = velocity[i]
        a = acceleration[i]
        v_norm = float(np.linalg.norm(v))
        if not np.isfinite(v_norm) or v_norm < 1.0e-9:
            rotmats[i] = identity
            continue
        x_hat = v / v_norm
        a_perp = a - np.dot(a, x_hat) * x_hat
        a_perp_norm = float(np.linalg.norm(a_perp))
        if not np.isfinite(a_perp_norm) or a_perp_norm < 1.0e-9:
            rotmats[i] = identity
            continue
        y_hat = a_perp / a_perp_norm
        z_hat = np.cross(x_hat, y_hat)
        rotmats[i] = np.column_stack([x_hat, y_hat, z_hat])
    return rotmat_to_quat(rotmats)


def _butt_from_clubhead(
    clubhead: np.ndarray, velocity: np.ndarray, shaft_length: float
) -> np.ndarray:
    """``butt = clubhead - shaft * v_hat``, NaN where ``v_hat`` is undefined."""
    speeds = np.linalg.norm(velocity, axis=1, keepdims=True)
    safe = speeds > 1.0e-9
    direction = np.where(safe, velocity / np.where(safe, speeds, 1.0), np.nan)
    return clubhead - shaft_length * direction


def _synthetic_provenance(label: str) -> SourceProvenance:
    return SourceProvenance(
        filename=f"{label}.dataframe",
        format="machinelearning",
        subject_id="ML",
        trial_id=label,
        sha256=hashlib.sha256(label.encode("utf-8")).hexdigest(),
    )


@precondition(
    lambda df, **_: isinstance(df, pd.DataFrame),
    "df must be a pandas DataFrame",
)
@postcondition(
    lambda result: isinstance(result, ClubTarget),
    "to_canonical_target_from_clubface must return a ClubTarget",
)
def to_canonical_target_from_clubface(
    df: pd.DataFrame,
    *,
    shaft_length: float = DEFAULT_SHAFT_LENGTH_M,
    source: SourceProvenance | None = None,
) -> ClubTarget:
    """Convert a clubface-style DataFrame to a canonical :class:`ClubTarget`.

    Position columns (``clubface_x/y/z``) become ``clubhead``. Velocity
    columns are used to derive a tangent-frame quaternion and, together with
    ``shaft_length``, a synthetic ``butt`` position. ``time`` defaults to
    1 kHz uniform spacing if absent.

    Raises:
        ValueError: if any required column is missing.
    """
    warnings.warn(_LOSSY_MSG, UserWarning, stacklevel=2)
    required = list(CLUBFACE_POSITION)
    _require_columns(df, required)
    n = len(df)
    if n < 2:
        raise ValueError(f"DataFrame must have >=2 rows (got {n})")

    time = _time_vector(df, n)
    time = time - float(time[0])
    clubhead = df[list(CLUBFACE_POSITION)].to_numpy(dtype=np.float64)
    if all(c in df.columns for c in CLUBFACE_VELOCITY):
        velocity = df[list(CLUBFACE_VELOCITY)].to_numpy(dtype=np.float64)
    else:
        velocity = _gradient(clubhead, time)
    if all(c in df.columns for c in CLUBFACE_ACCELERATION):
        acceleration = df[list(CLUBFACE_ACCELERATION)].to_numpy(dtype=np.float64)
    else:
        acceleration = _gradient(velocity, time)

    quat = _quat_from_velocity_acceleration(velocity, acceleration)
    butt = _butt_from_clubhead(clubhead, velocity, shaft_length)
    # ClubTarget validation forbids non-finite butt positions; if any NaN crept
    # in (zero velocity at first/last sample), fall back to clubhead - shaft*z.
    bad = ~np.all(np.isfinite(butt), axis=1)
    if np.any(bad):
        butt[bad] = clubhead[bad] - np.array([0.0, 0.0, shaft_length])

    impact = _impact_idx(
        df, default=int(np.argmax(np.linalg.norm(velocity, axis=1))) + 1
    )
    impact = max(1, min(impact, n))

    prov = source if source is not None else _synthetic_provenance("clubface")
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=impact,
        source=prov,
    )


@precondition(
    lambda df, **_: isinstance(df, pd.DataFrame),
    "df must be a pandas DataFrame",
)
@postcondition(
    lambda result: isinstance(result, ClubTarget),
    "to_canonical_target_from_clublogs must return a ClubTarget",
)
def to_canonical_target_from_clublogs(
    df: pd.DataFrame,
    *,
    shaft_length: float = DEFAULT_SHAFT_LENGTH_M,
    source: SourceProvenance | None = None,
) -> ClubTarget:
    """Convert a ClubLogs-style DataFrame to a canonical :class:`ClubTarget`.

    The ClubLogs schema also lacks butt and orientation; the recovery rules
    match :func:`to_canonical_target_from_clubface`.

    Raises:
        ValueError: if any required column is missing.
    """
    warnings.warn(_LOSSY_MSG, UserWarning, stacklevel=2)
    _require_columns(df, CLUBLOGS_POSITION)
    n = len(df)
    if n < 2:
        raise ValueError(f"DataFrame must have >=2 rows (got {n})")

    time = _time_vector(df, n)
    time = time - float(time[0])
    clubhead = df[list(CLUBLOGS_POSITION)].to_numpy(dtype=np.float64)
    if all(c in df.columns for c in CLUBLOGS_VELOCITY):
        velocity = df[list(CLUBLOGS_VELOCITY)].to_numpy(dtype=np.float64)
    else:
        velocity = _gradient(clubhead, time)
    if all(c in df.columns for c in CLUBLOGS_ACCELERATION):
        acceleration = df[list(CLUBLOGS_ACCELERATION)].to_numpy(dtype=np.float64)
    else:
        acceleration = _gradient(velocity, time)

    quat = _quat_from_velocity_acceleration(velocity, acceleration)
    butt = _butt_from_clubhead(clubhead, velocity, shaft_length)
    bad = ~np.all(np.isfinite(butt), axis=1)
    if np.any(bad):
        butt[bad] = clubhead[bad] - np.array([0.0, 0.0, shaft_length])

    impact = _impact_idx(
        df, default=int(np.argmax(np.linalg.norm(velocity, axis=1))) + 1
    )
    impact = max(1, min(impact, n))

    prov = source if source is not None else _synthetic_provenance("clublogs")
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=impact,
        source=prov,
    )


@precondition(
    lambda target: isinstance(target, ClubTarget),
    "target must be a ClubTarget",
)
@postcondition(
    lambda result: isinstance(result, pd.DataFrame),
    "to_machinelearning_clubface must return a DataFrame",
)
def to_machinelearning_clubface(target: ClubTarget) -> pd.DataFrame:
    """Project a :class:`ClubTarget` onto the ``clubface_*`` flat schema.

    Velocity and acceleration are computed from ``clubhead`` via central
    differences (matching :mod:`prepare_club_target_trajectory`). Butt and
    quaternion are not represented.
    """
    time = target.time.astype(np.float64)
    clubhead = target.clubhead.astype(np.float64)
    velocity = _gradient(clubhead, time)
    acceleration = _gradient(velocity, time)
    return pd.DataFrame(
        {
            "time": time,
            "clubface_x": clubhead[:, 0],
            "clubface_y": clubhead[:, 1],
            "clubface_z": clubhead[:, 2],
            "clubface_vx": velocity[:, 0],
            "clubface_vy": velocity[:, 1],
            "clubface_vz": velocity[:, 2],
            "clubface_ax": acceleration[:, 0],
            "clubface_ay": acceleration[:, 1],
            "clubface_az": acceleration[:, 2],
            "impact_idx": np.full(time.shape[0], target.impact_idx, dtype=np.int64),
        }
    )


@precondition(
    lambda target: isinstance(target, ClubTarget),
    "target must be a ClubTarget",
)
@postcondition(
    lambda result: isinstance(result, pd.DataFrame),
    "to_machinelearning_clublogs must return a DataFrame",
)
def to_machinelearning_clublogs(target: ClubTarget) -> pd.DataFrame:
    """Project a :class:`ClubTarget` onto the ``ClubLogs_CHGlobal*`` schema."""
    time = target.time.astype(np.float64)
    clubhead = target.clubhead.astype(np.float64)
    velocity = _gradient(clubhead, time)
    acceleration = _gradient(velocity, time)
    columns = {
        "time": time,
        CLUBLOGS_POSITION[0]: clubhead[:, 0],
        CLUBLOGS_POSITION[1]: clubhead[:, 1],
        CLUBLOGS_POSITION[2]: clubhead[:, 2],
        CLUBLOGS_VELOCITY[0]: velocity[:, 0],
        CLUBLOGS_VELOCITY[1]: velocity[:, 1],
        CLUBLOGS_VELOCITY[2]: velocity[:, 2],
        CLUBLOGS_ACCELERATION[0]: acceleration[:, 0],
        CLUBLOGS_ACCELERATION[1]: acceleration[:, 1],
        CLUBLOGS_ACCELERATION[2]: acceleration[:, 2],
        "impact_idx": np.full(time.shape[0], target.impact_idx, dtype=np.int64),
    }
    return pd.DataFrame(columns)
