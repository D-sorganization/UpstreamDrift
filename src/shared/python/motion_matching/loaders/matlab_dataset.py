"""MATLAB ``.mat`` club-target loader.

Loads ``.mat`` files matching the schema documented in issue #4477:

    data.time            (N,)        seconds
    data.midhands_xyz    (N, 3)      metres
    data.midhands_dircos (N, 9)      rotation, row-major flat 3x3
    data.clubface_xyz    (N, 3)      metres
    data.clubface_dircos (N, 9)      rotation, row-major flat 3x3
    params.Address       int         1-based frame index (swing-segment relative)
    params.TopOfBackswing int        1-based frame index (swing-segment relative)
    params.Impact        int         1-based frame index (swing-segment relative)
    params.Finish        int         1-based frame index (swing-segment relative)
    params.impact_frame  int         alias of Impact
    params.swing_start   int         1-based frame index
    params.backswing_start int       1-based frame index

Convention mapping into :class:`ClubTarget`:

    data.midhands_xyz    -> butt
    data.clubface_xyz    -> clubhead
    data.clubface_dircos -> club_quat (via ``rotmat_to_quat``)

Stamped impact: the time vector in these files centres ``t = 0`` on the
physical impact frame (verified by max clubhead speed across all canonical
files). The integer ``params.Impact`` field is preserved for traceability but
is referenced against a swing-segment subset rather than the row-index of the
full time-series, so it is not used as the row-index of the impact sample.
We use the row whose ``time`` is closest to zero as the stamped impact frame
and pass that into :func:`resample_target` instead of running the kinematic
peak-speed heuristic.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from src.shared.python.core.contracts import postcondition, precondition

from ..club_target import AlignOptions, ClubTarget, SourceProvenance
from ._align import resample_target
from ._quaternion import rotmat_to_quat

logger = logging.getLogger(__name__)

# Tolerances for rotation-matrix validation. The source data is single
# precision in places, so we cannot demand machine epsilon.
_ROT_DET_TOL = 1.0e-3
_ROT_ORTHO_TOL = 1.0e-3
_FORMAT_LABEL = "mat_dataset"

_REQUIRED_DATA_FIELDS = (
    "time",
    "midhands_xyz",
    "clubface_xyz",
    "clubface_dircos",
)


def _sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_struct(mat: dict[str, Any], name: str) -> Any:
    """Fetch a top-level struct, raising if missing."""
    if name not in mat:
        raise ValueError(f"Required top-level struct {name!r} missing from .mat file")
    return mat[name]


def _struct_field(struct: Any, field: str) -> Any:
    """Fetch a field from a scipy mat-struct or dict, raising if missing."""
    if isinstance(struct, dict):
        if field not in struct:
            raise ValueError(f"Required field {field!r} missing from struct")
        return struct[field]
    if hasattr(struct, field):
        return getattr(struct, field)
    raise ValueError(f"Required field {field!r} missing from struct")


def _reshape_dircos(dircos: np.ndarray, name: str) -> np.ndarray:
    """Reshape an ``(N, 9)`` or ``(N, 3, 3)`` dircos array to ``(N, 3, 3)``.

    The canonical files store rotation matrices flat-row-major in 9 columns.
    """
    arr = np.asarray(dircos, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return arr
    if arr.ndim == 2 and arr.shape[1] == 9:
        return arr.reshape(arr.shape[0], 3, 3)
    raise ValueError(f"{name} must have shape (N, 9) or (N, 3, 3); got {arr.shape}")


def _validate_rotation_stack(rot: np.ndarray, name: str) -> None:
    """Reject rotation stacks with non-unit determinant or non-orthonormal columns.

    Raises ``ValueError`` mentioning "rotation" so callers can pattern-match.
    """
    if not np.all(np.isfinite(rot)):
        raise ValueError(f"{name} contains non-finite rotation entries")
    dets = np.linalg.det(rot)
    bad_det = np.abs(dets - 1.0) > _ROT_DET_TOL
    if np.any(bad_det):
        first = int(np.argmax(bad_det))
        raise ValueError(
            f"{name} contains invalid rotation matrices (det != +1 within "
            f"{_ROT_DET_TOL}); first bad row {first} det={float(dets[first]):.4f}"
        )
    # Orthonormality: columns of R satisfy R.T @ R == I.
    eye = np.eye(3, dtype=np.float64)
    gram = np.einsum("nij,nik->njk", rot, rot)
    diffs = np.abs(gram - eye).reshape(gram.shape[0], -1).max(axis=1)
    bad_ortho = diffs > _ROT_ORTHO_TOL
    if np.any(bad_ortho):
        first = int(np.argmax(bad_ortho))
        raise ValueError(
            f"{name} contains non-orthonormal rotation columns (max |R^T R - I| "
            f"> {_ROT_ORTHO_TOL}); first bad row {first} dev={float(diffs[first]):.4f}"
        )


def _stamped_impact_index(time: np.ndarray, params: Any) -> int:
    """Return the 0-based row index of the stamped impact in the time vector.

    The canonical files stamp impact by centring ``t = 0`` on the physical
    impact frame. Use the row closest to ``t = 0`` if it lies inside the
    sampled range; otherwise fall back to the integer ``params.Impact``
    (1-based) field if it is a valid in-range row index.
    """
    n = int(time.shape[0])
    t_min, t_max = float(time[0]), float(time[-1])
    if t_min <= 0.0 <= t_max:
        return int(np.argmin(np.abs(time)))
    # Fallback path: try the integer field. We only accept it when it is a
    # valid in-range row index, otherwise the file is malformed.
    raw = _struct_field(params, "Impact")
    try:
        impact_1based = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot determine stamped impact: time vector does not span t=0 "
            "and params.Impact is not an integer"
        ) from exc
    impact_0based = impact_1based - 1
    if not (0 <= impact_0based < n):
        raise ValueError(
            f"params.Impact={impact_1based} is out of range [1, {n}] and the "
            "time vector does not span t=0"
        )
    return impact_0based


def _build_provenance(path: Path) -> SourceProvenance:
    """Build a :class:`SourceProvenance` for a ``.mat`` source file."""
    stem = path.stem
    subject_id = stem.split("_", 1)[0] if "_" in stem else stem
    return SourceProvenance(
        filename=path.name,
        format=_FORMAT_LABEL,
        subject_id=subject_id,
        trial_id=stem,
        sha256=_sha256_of(path),
    )


@precondition(
    lambda path, opts: Path(path).exists(),
    ".mat file must exist",
)
@precondition(
    lambda path, opts: opts.sample_rate_hz > 0,
    "sample_rate_hz must be > 0",
)
@postcondition(
    lambda result: isinstance(result, ClubTarget),
    "load_club_target_mat must return a ClubTarget",
)
def load_club_target_mat(path: Path | str, opts: AlignOptions) -> ClubTarget:
    """Load a ``.mat`` motion-capture file into a canonical ``ClubTarget``.

    The loader is agnostic to the recording subject: any ``.mat`` file
    matching the documented schema (``data.{time, midhands_xyz, clubface_xyz,
    clubface_dircos}`` plus ``params.Impact``) will load.

    Args:
        path: Path to the ``.mat`` source file.
        opts: Resampling / impact-alignment options.

    Returns:
        Validated :class:`ClubTarget` on the simulation timegrid.

    Raises:
        FileNotFoundError: If the file does not exist (via DbC precondition).
        ValueError: If a required struct/field is missing, the rotation stack
            fails validity (det != +1 or non-orthonormal columns), or the
            stamped impact is undeterminable.
    """
    path = Path(path)
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    data = _get_struct(mat, "data")
    params = _get_struct(mat, "params")

    for f in _REQUIRED_DATA_FIELDS:
        _struct_field(data, f)

    raw_time = np.asarray(_struct_field(data, "time"), dtype=np.float64)
    if raw_time.ndim != 1 or raw_time.shape[0] < 2:
        raise ValueError(
            f"data.time must be a 1-D array with at least 2 samples, got "
            f"shape {raw_time.shape}"
        )

    raw_butt = np.asarray(_struct_field(data, "midhands_xyz"), dtype=np.float64)
    raw_clubhead = np.asarray(_struct_field(data, "clubface_xyz"), dtype=np.float64)
    if raw_butt.shape != (raw_time.shape[0], 3):
        raise ValueError(
            f"data.midhands_xyz must have shape ({raw_time.shape[0]}, 3); "
            f"got {raw_butt.shape}"
        )
    if raw_clubhead.shape != (raw_time.shape[0], 3):
        raise ValueError(
            f"data.clubface_xyz must have shape ({raw_time.shape[0]}, 3); "
            f"got {raw_clubhead.shape}"
        )

    rot = _reshape_dircos(_struct_field(data, "clubface_dircos"), "clubface_dircos")
    if rot.shape[0] != raw_time.shape[0]:
        raise ValueError(
            f"clubface_dircos has {rot.shape[0]} rows; expected "
            f"{raw_time.shape[0]} to match time"
        )
    _validate_rotation_stack(rot, "clubface_dircos")
    raw_quat = rotmat_to_quat(rot)

    impact_raw = _stamped_impact_index(raw_time, params)
    sim_time, butt, clubhead, quat, impact_idx = resample_target(
        raw_time, raw_butt, raw_clubhead, raw_quat, impact_raw, opts
    )

    source = _build_provenance(path)
    logger.info(
        "Loaded ClubTarget from %s: %d output samples (impact=%d, raw_idx=%d)",
        path.name,
        sim_time.shape[0],
        impact_idx,
        impact_raw,
    )
    return ClubTarget(
        time=sim_time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=int(impact_idx),
        source=source,
    )
