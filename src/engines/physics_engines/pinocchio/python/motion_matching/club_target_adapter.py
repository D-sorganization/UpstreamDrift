"""Club-swing-dataset ``*.mat`` -> canonical :class:`ClubTarget` adapter.

The club-swing mocap dataset ships as paired MATLAB files:

* ``<name>.mat``                  -- raw fields (``data.time``, ``data.midhands_xyz``,
  ``data.midhands_dircos``, ``data.clubface_xyz``, ``data.clubface_dircos``) plus
  ``params`` with event indices (``Address``, ``TopOfBackswing``, ``Impact``,
  ``Finish``, ``impact_frame``).
* ``<name>_targetKinematics.mat`` -- resampled / cleaned version with fields
  ``Time``, ``MH`` (mid-hands xyz), ``MH_R`` (3x3xN rotation stack).

This adapter reads both, prefers the ``_targetKinematics`` arrays for the
canonical fields, but pulls event indices from the raw ``params`` struct and
keeps the clubface trajectory from the raw file (the resampled artifact does
not include it).

The output is the canonical ``ClubTarget`` from
``src/shared/python/motion_matching/club_target.py`` (frozen dataclass with a
``__post_init__`` validator). The shared module is guaranteed importable after
issue #4095 (PARITY-LOADERS, closed) promoted the dataclass to a top-level
package; a defensive local stub of ``ClubTarget`` / ``SourceProvenance`` is
kept below so the adapter still imports in stripped-down checkouts (e.g. a
sliced wheel that excludes the ``motion_matching`` package), but the canonical
path is always preferred.

Note: the engine-local Rob Neal ``.mat`` loader was *not* promoted by #4095
(only the canonical types were); a future refactor may move
``load_robneal_target`` to ``shared/python/motion_matching/loaders/`` for
symmetry with the C3D / Excel loaders that already live there.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as _sio

logger = logging.getLogger(__name__)

# Validation tolerances mirror the canonical ``ClubTarget`` defaults so the
# stub fallback below stays consistent with the upstream schema.
_QUAT_NORM_TOL = 1.0e-6
_MAX_POSITION_NORM_M = 5.0
_TIME_EPS = 1.0e-9


# ---------------------------------------------------------------------------
# ClubTarget import w/ defensive local-stub fallback.
#
# Post issue #4095 (PARITY-LOADERS, closed) the shared module is shipped at
# ``src/shared/python/motion_matching/club_target.py`` and the canonical path
# is always taken in normal checkouts. The stub below only fires for
# stripped-down environments (e.g. an engine wheel sliced without
# ``motion_matching``) so this adapter remains importable in isolation.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised by both branches via tests
    from src.shared.python.motion_matching.club_target import (  # type: ignore[import-not-found]
        ClubTarget,
        SourceProvenance,
    )

    _USING_STUB = False
except ImportError:  # pragma: no cover - fallback for stripped-down checkouts
    logger.warning(
        "src.shared.python.motion_matching.club_target unavailable; using "
        "local stub. This branch is only expected in stripped-down checkouts "
        "without the shared motion_matching package."
    )

    @dataclass(frozen=True)
    class SourceProvenance:  # type: ignore[no-redef]
        """Local stub mirroring the canonical ``SourceProvenance``."""

        filename: str
        format: str
        subject_id: str
        trial_id: str
        sha256: str

    @dataclass(frozen=True)
    class ClubTarget:  # type: ignore[no-redef]
        """Local stub mirroring the canonical ``ClubTarget``.

        Validation rules match ``CLUB_IK_SPEC.md`` so the stub round-trips
        with the upstream type whenever both are available.
        """

        time: np.ndarray
        butt: np.ndarray
        clubhead: np.ndarray
        club_quat: np.ndarray
        impact_idx: int
        source: SourceProvenance

        def __post_init__(self) -> None:
            _validate_stub_target(self)

    _USING_STUB = True


# ---------------------------------------------------------------------------
# Stub validation helpers (only used when the canonical type isn't available).
# ---------------------------------------------------------------------------


def _validate_stub_target(t: ClubTarget) -> None:  # noqa: C901
    """Replicate the canonical validation rules for the local stub."""
    time = np.asarray(t.time)
    if time.ndim != 1:
        raise ValueError(f"time must be 1-D, got shape {time.shape}")
    n = time.shape[0]
    if n < 2:
        raise ValueError(f"time must have at least 2 samples (got {n})")
    if abs(float(time[0])) > _TIME_EPS:
        raise ValueError(f"time[0] must be 0, got {time[0]!r}")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing")
    for name, arr, cols in (
        ("butt", t.butt, 3),
        ("clubhead", t.clubhead, 3),
        ("club_quat", t.club_quat, 4),
    ):
        if np.asarray(arr).shape != (n, cols):
            raise ValueError(
                f"{name} must have shape ({n}, {cols}), got {np.asarray(arr).shape}"
            )
    for name, arr in (("butt", t.butt), ("clubhead", t.clubhead)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf")
        norms = np.sqrt(np.einsum("ij,ij->i", arr, arr))
        if np.any(norms >= _MAX_POSITION_NORM_M):
            raise ValueError(
                f"{name} has |r| >= {_MAX_POSITION_NORM_M} m "
                f"(max {float(norms.max()):.3f})"
            )
    qnorms = np.sqrt(np.einsum("ij,ij->i", t.club_quat, t.club_quat))
    if np.any(np.abs(qnorms - 1.0) > _QUAT_NORM_TOL):
        max_dev = float(np.abs(qnorms - 1.0).max())
        raise ValueError(
            "club_quat rows must be unit-norm to within "
            f"{_QUAT_NORM_TOL} (max deviation {max_dev:.2e})"
        )
    if not (1 <= int(t.impact_idx) <= n):
        raise ValueError(f"impact_idx must be in [1, {n}], got {t.impact_idx}")
    if not isinstance(t.source, SourceProvenance):
        raise TypeError("source must be a SourceProvenance instance")


# ---------------------------------------------------------------------------
# .mat helpers
# ---------------------------------------------------------------------------


def _resolve_pair(path: Path) -> tuple[Path, Path]:
    """Return ``(raw_path, resampled_path)`` regardless of which one was given.

    Accepts either ``<name>.mat`` or ``<name>_targetKinematics.mat`` and locates
    the missing partner alongside it. Raises ``FileNotFoundError`` if either
    half of the pair is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() != ".mat":
        raise ValueError(f"Expected a .mat file, got {p.name}")

    name = p.name
    if name.lower().endswith("_targetkinematics.mat"):
        resampled = p
        raw_name = re.sub(
            r"_targetkinematics\.mat$",
            ".mat",
            name,
            flags=re.IGNORECASE,
        )
        raw = p.with_name(raw_name)
    else:
        raw = p
        resampled = p.with_name(p.stem + "_targetKinematics.mat")

    if not raw.exists():
        raise FileNotFoundError(f"Raw .mat partner not found: {raw}")
    if not resampled.exists():
        raise FileNotFoundError(f"Resampled .mat partner not found: {resampled}")
    return raw, resampled


def _sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of ``path``'s bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_mat(path: Path) -> dict[str, Any]:
    """``scipy.io.loadmat`` with squeezing and struct unwrapping enabled."""
    return _sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)


def _struct_field(obj: Any, *names: str) -> Any:
    """Return the first present attribute among ``names`` on a MATLAB struct.

    Raises ``KeyError`` if none of ``names`` are present.
    """
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise KeyError(f"None of fields {names!r} found on struct")


def _dircos_to_rotmat(dircos: np.ndarray) -> np.ndarray:
    """Reshape an ``(N, 9)`` direction-cosine table into ``(N, 3, 3)``.

    The Rob Neal convention is row-major: each row is
    ``[Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz]`` where ``X`` / ``Y`` / ``Z`` are
    the columns of the rotation matrix expressed in world coordinates.
    """
    arr = np.asarray(dircos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(
            f"direction-cosine table must be (N, 9); got shape {arr.shape}"
        )
    n = arr.shape[0]
    out = np.empty((n, 3, 3), dtype=np.float64)
    out[:, 0, 0] = arr[:, 0]  # Xx
    out[:, 1, 0] = arr[:, 1]  # Xy
    out[:, 2, 0] = arr[:, 2]  # Xz
    out[:, 0, 1] = arr[:, 3]  # Yx
    out[:, 1, 1] = arr[:, 4]  # Yy
    out[:, 2, 1] = arr[:, 5]  # Yz
    out[:, 0, 2] = arr[:, 6]  # Zx
    out[:, 1, 2] = arr[:, 7]  # Zy
    out[:, 2, 2] = arr[:, 8]  # Zz
    return out


def _rotmat_stack_to_quat(rotmats: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 3, 3)`` rotation stack to ``(N, 4)`` ``[w,x,y,z]``.

    Defers to the shared loader helper when available (single source of truth)
    and otherwise uses an inlined Shepperd's method so this adapter remains
    importable in stripped-down checkouts.
    """
    try:
        from src.shared.python.motion_matching.loaders._quaternion import (  # type: ignore[import-not-found]
            rotmat_to_quat,
        )

        return rotmat_to_quat(rotmats)
    except ImportError:  # pragma: no cover - exercised when shared is absent
        return _rotmat_to_quat_local(rotmats)


def _rotmat_to_quat_local(rotmats: np.ndarray) -> np.ndarray:
    """Fallback Shepperd's-method conversion (matches the shared helper)."""
    arr = np.asarray(rotmats, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"Expected (N, 3, 3); got {arr.shape}")
    n = arr.shape[0]
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        r = arr[i]
        m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
        m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
        m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]
        trace = m00 + m11 + m22
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif (m00 > m11) and (m00 > m22):
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        norm = np.linalg.norm(q)
        out[i] = q / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])
    flips = out[:, 0] < 0.0
    if np.any(flips):
        out[flips] = -out[flips]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_robneal_target(path: Path | str) -> ClubTarget:  # noqa: C901
    """Load a club-swing-dataset mocap pair into a canonical :class:`ClubTarget`.

    Args:
        path: Either the raw ``<name>.mat`` or the resampled
              ``<name>_targetKinematics.mat``. The partner file is auto-located
              alongside.

    Returns:
        A validated :class:`ClubTarget` whose ``source.format`` is
        ``"club_swing_mat"`` and whose ``source.filename`` is the *raw* file's
        basename (the resampled partner is recorded in ``trial_id``).

    Raises:
        FileNotFoundError: if either half of the pair is missing.
        ValueError: if the .mat structure is malformed, contains NaN/Inf
            frames, or otherwise fails ``ClubTarget`` validation.
        KeyError: if expected MATLAB struct fields are missing.

    Notes:
        * Time is rebased so ``time[0] == 0``.
        * Frames containing any NaN/Inf in mid-hands xyz, clubface xyz, or
          either direction-cosine table are dropped before validation. If the
          drop removes the impact frame the next surviving frame is used.
        * ``impact_idx`` is **1-based** to match the canonical schema.
    """
    raw_path, resampled_path = _resolve_pair(Path(path))

    raw = _load_mat(raw_path)
    resampled = _load_mat(resampled_path)

    if "data" not in raw or "params" not in raw:
        raise ValueError(
            f"{raw_path.name}: expected top-level 'data' and 'params' structs"
        )
    data = raw["data"]
    params = raw["params"]

    raw_time = np.asarray(_struct_field(data, "time"), dtype=np.float64).ravel()
    butt_xyz = np.asarray(_struct_field(data, "midhands_xyz"), dtype=np.float64)
    butt_dircos = np.asarray(_struct_field(data, "midhands_dircos"), dtype=np.float64)
    head_xyz = np.asarray(_struct_field(data, "clubface_xyz"), dtype=np.float64)
    head_dircos = np.asarray(_struct_field(data, "clubface_dircos"), dtype=np.float64)

    n_raw = raw_time.shape[0]
    for name, arr, cols in (
        ("midhands_xyz", butt_xyz, 3),
        ("clubface_xyz", head_xyz, 3),
        ("midhands_dircos", butt_dircos, 9),
        ("clubface_dircos", head_dircos, 9),
    ):
        if arr.shape != (n_raw, cols):
            raise ValueError(
                f"{raw_path.name}: {name} has shape {arr.shape}, "
                f"expected ({n_raw}, {cols})"
            )

    # Prefer the resampled mid-hands trajectory + rotation when shapes line up.
    use_resampled = False
    if "Time" in resampled and "MH" in resampled and "MH_R" in resampled:
        rs_time = np.asarray(resampled["Time"], dtype=np.float64).ravel()
        rs_mh = np.asarray(resampled["MH"], dtype=np.float64)
        rs_mh_r = np.asarray(resampled["MH_R"], dtype=np.float64)
        # MH_R is stored as (3, 3, N) in MATLAB; rearrange to (N, 3, 3).
        if rs_mh_r.ndim == 3 and rs_mh_r.shape[:2] == (3, 3):
            rs_mh_r = np.moveaxis(rs_mh_r, 2, 0)
        if (
            rs_time.shape[0] == n_raw
            and rs_mh.shape == (n_raw, 3)
            and rs_mh_r.shape == (n_raw, 3, 3)
        ):
            use_resampled = True
            base_time = rs_time
            butt_xyz = rs_mh
            butt_rotmats = rs_mh_r
            logger.debug(
                "%s: using resampled MH/MH_R from %s",
                raw_path.name,
                resampled_path.name,
            )
        else:
            logger.debug(
                "%s: resampled shapes %s/%s/%s incompatible with raw N=%d; "
                "falling back to raw arrays",
                raw_path.name,
                rs_time.shape,
                rs_mh.shape,
                rs_mh_r.shape,
                n_raw,
            )
    if not use_resampled:
        base_time = raw_time
        butt_rotmats = _dircos_to_rotmat(butt_dircos)
    head_rotmats = _dircos_to_rotmat(head_dircos)

    # Drop frames with any non-finite values across all source arrays.
    finite_mask = (
        np.all(np.isfinite(base_time[:, None]), axis=1)
        & np.all(np.isfinite(butt_xyz), axis=1)
        & np.all(np.isfinite(head_xyz), axis=1)
        & np.all(np.isfinite(butt_rotmats.reshape(n_raw, -1)), axis=1)
        & np.all(np.isfinite(head_rotmats.reshape(n_raw, -1)), axis=1)
    )
    if not np.any(finite_mask):
        raise ValueError(f"{raw_path.name}: no finite frames after NaN/Inf filter")
    n_dropped = int(n_raw - finite_mask.sum())
    if n_dropped > 0:
        logger.info(
            "%s: dropped %d non-finite frames out of %d",
            raw_path.name,
            n_dropped,
            n_raw,
        )

    keep_idx = np.flatnonzero(finite_mask)
    time = base_time[keep_idx].astype(np.float64)
    butt = butt_xyz[keep_idx].astype(np.float64)
    clubhead = head_xyz[keep_idx].astype(np.float64)
    butt_rotmats = butt_rotmats[keep_idx]
    # We expose the *clubhead* (a.k.a. clubface) orientation via club_quat
    # because that is what downstream IK consumes; the butt rotation is
    # available as raw data only and used here for impact-frame fallback.
    club_quat = _rotmat_stack_to_quat(head_rotmats[keep_idx])

    # Rebase time to start at 0 and confirm strict monotonicity (resampled
    # arrays are already monotonic; raw mocap occasionally has duplicate
    # frames at the boundaries which would fail validation downstream).
    time = time - float(time[0])
    if not np.all(np.diff(time) > 0):
        # Drop duplicate timestamps (keep the first).
        unique_mask = np.concatenate(([True], np.diff(time) > 0))
        if unique_mask.sum() < 2:
            raise ValueError(
                f"{raw_path.name}: fewer than 2 strictly increasing samples"
            )
        time = time[unique_mask]
        butt = butt[unique_mask]
        clubhead = clubhead[unique_mask]
        club_quat = club_quat[unique_mask]
        keep_idx = keep_idx[unique_mask]
        logger.info(
            "%s: dropped %d duplicate-timestamp frames",
            raw_path.name,
            int((~unique_mask).sum()),
        )

    # Resolve the impact frame: prefer ``params.Impact``, fall back to
    # ``params.impact_frame``. Map the original 1-based MATLAB index back
    # through the keep mask.
    raw_impact_1b = int(_struct_field(params, "Impact", "impact_frame"))
    if raw_impact_1b < 1 or raw_impact_1b > n_raw:
        raise ValueError(
            f"{raw_path.name}: impact index {raw_impact_1b} out of range [1, {n_raw}]"
        )
    raw_impact_0b = raw_impact_1b - 1
    surviving = np.flatnonzero(keep_idx >= raw_impact_0b)
    if surviving.size == 0:
        raise ValueError(
            f"{raw_path.name}: impact frame {raw_impact_1b} dropped and no "
            "later surviving frame is available"
        )
    impact_idx_0b = int(surviving[0])
    impact_idx = impact_idx_0b + 1  # 1-based per CLUB_IK_SPEC

    subject_id = raw_path.stem.split("_")[0] or "unknown"
    source = SourceProvenance(
        filename=raw_path.name,
        format="club_swing_mat",
        subject_id=subject_id,
        trial_id=raw_path.stem,
        sha256=_sha256_of(raw_path),
    )
    logger.info(
        "Loaded ClubTarget from %s (+ %s): %d samples, impact_idx=%d, subject=%s",
        raw_path.name,
        resampled_path.name,
        time.shape[0],
        impact_idx,
        subject_id,
    )
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=club_quat,
        impact_idx=int(impact_idx),
        source=source,
    )


__all__ = [
    "ClubTarget",
    "SourceProvenance",
    "load_robneal_target",
]
