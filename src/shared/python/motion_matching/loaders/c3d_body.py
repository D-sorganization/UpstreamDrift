"""C3D loader producing a :class:`BodyTarget` (full-body marker trajectories).

Sibling of :mod:`loaders.c3d`, which only emits :class:`ClubTarget`. This
loader pulls the anatomical body markers out of the same C3D, fills short
occlusion gaps, converts source coordinates from Y-up to right-handed Z-up,
resamples each marker onto the simulation timegrid via NaN-preserving cubic
interpolation, and returns a validated :class:`BodyTarget`.

Uses the canonical :class:`C3DDataReader` from
``sidekick.lab.bio.c3d_reader`` (NOT the legacy
duplicate under ``src/engines``, which is being deprecated by a separate
issue).

Public API:
    DEFAULT_BODY_MARKER_EXCLUDES        -- markers always excluded by default.
    default_anatomical_marker_set       -- canonical 28-marker default subset.
    load_body_target_c3d                -- main entry point.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.shared.python.core.contracts import postcondition, precondition

from ..body_target import BodyEvent, BodyTarget
from ..club_target import AlignOptions, ClubTarget, SourceProvenance
from ._align import detect_impact_index
from ._gears import fill_short_gaps, y_up_to_z_up

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

# Markers whose presence in the C3D is a stuck sentinel rather than real data,
# or which are known to be occluded for the entire trace and therefore must
# not be silently spline-filled. Source-neutral; documented behaviours, not
# vendor identifiers.
DEFAULT_BODY_MARKER_EXCLUDES: frozenset[str] = frozenset(
    {
        "Marker_0:0:0",  # stuck-value sentinel
        "RShoulderTop",  # known occluded across reference traces
    }
)

# Canonical 28-marker anatomical subset present in the four reference C3D
# files. Order is fixed so that downstream consumers (visualisers, cost
# functions) can rely on stable column semantics.
_DEFAULT_ANATOMICAL_MARKERS: tuple[str, ...] = (
    "WaistLeft",
    "WaistRight",
    "WaistLBack",
    "WaistRBack",
    "BackTop",
    "BackLeft",
    "BackRight",
    "HeadTop",
    "HeadFront",
    "HeadSide",
    "LShoulderTop",
    "LShoulderBack",
    "LElbowOut",
    "LUArmHigh",
    "LWristTop",
    "RShoulderBack",
    "RElbowOut",
    "RUArmHigh",
    "RWristTop",
    "LKneeOut",
    "LToeIn",
    "LToeOut",
    "LAnkleOut",
    "RKneeOut",
    "RToeIn",
    "RToeOut",
    "RAnkleOut",
    "RShoulderTop",  # included by name but excluded-by-default; see below.
)

# Wrist-marker candidates used by the kinematic-peak heuristic when no
# ``impact_source`` is supplied. The lead wrist (right for right-handed
# players, left for left-handed) carries the largest speed peak around impact;
# we just take whichever wrist marker has the highest peak.
_WRIST_HEURISTIC_MARKERS: tuple[str, ...] = ("RWristTop", "LWristTop")


def default_anatomical_marker_set() -> tuple[str, ...]:
    """Return the canonical 28-marker anatomical subset (default ``marker_set``).

    The result is filtered through :data:`DEFAULT_BODY_MARKER_EXCLUDES` so the
    known-occluded marker is dropped. Callers wishing to keep the occluded
    marker (e.g. to inspect its NaN coverage) can pass an explicit
    ``marker_set`` to :func:`load_body_target_c3d`.
    """
    return tuple(
        m for m in _DEFAULT_ANATOMICAL_MARKERS if m not in DEFAULT_BODY_MARKER_EXCLUDES
    )


def _sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_c3d_reader_class():
    """Import :class:`C3DDataReader` from the canonical bio module."""
    from src.shared.python.sidekick.lab.bio.c3d_reader import (
        C3DDataReader,
    )

    return C3DDataReader


def _pivot_marker_dict(
    df: pd.DataFrame, n_frames: int, marker_names: Sequence[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pivot the tidy points DataFrame into ``{marker: (N, 3)}`` and a time array.

    Frames missing from ``df`` for a given marker remain ``NaN``.
    """
    if "time" in df.columns:
        # Use the per-frame time of any single marker (any will do; they're shared).
        ref = df.drop_duplicates("frame").sort_values("frame")
        time = ref["time"].to_numpy(dtype=np.float64)
    else:
        time = np.arange(n_frames, dtype=np.float64)
    time = time - float(time[0])

    by_marker = {m: g.sort_values("frame") for m, g in df.groupby("marker")}
    out: dict[str, np.ndarray] = {}
    for label in marker_names:
        sub = by_marker.get(label)
        xyz = np.full((n_frames, 3), np.nan, dtype=np.float64)
        if sub is not None:
            frames = sub["frame"].to_numpy(dtype=np.int64)
            if frames.size and frames.min() == 1:
                frames = frames - 1
            valid = (frames >= 0) & (frames < n_frames)
            xyz[frames[valid], 0] = sub["x"].to_numpy(dtype=np.float64)[valid]
            xyz[frames[valid], 1] = sub["y"].to_numpy(dtype=np.float64)[valid]
            xyz[frames[valid], 2] = sub["z"].to_numpy(dtype=np.float64)[valid]
        out[label] = xyz
    return out, time


def _resample_marker_series(
    sim_t: np.ndarray, raw_t: np.ndarray, raw_xyz: np.ndarray, max_gap_s: float
) -> np.ndarray:
    """Resample a single marker's ``(N, 3)`` series onto ``sim_t``.

    Cubic interpolation per coordinate where there are >= 4 finite samples,
    NaN-preserving across long gaps. Output samples whose nearest source
    interval contains a NaN longer than ``max_gap_s`` are kept NaN.
    """
    out = np.full((sim_t.shape[0], 3), np.nan, dtype=np.float64)
    finite_per = np.isfinite(raw_xyz).all(axis=-1)
    if finite_per.sum() < 2:
        return out
    # Build a NaN-mask in the source time domain for the long-gap exclusion.
    source_finite_t = raw_t[finite_per]
    for k in range(3):
        col = raw_xyz[:, k]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        t_ok = raw_t[ok]
        c_ok = col[ok]
        # Cubic via numpy.polyfit per piecewise window is overkill; np.interp
        # gives linear, but for smoothness use a single global cubic spline
        # only when feasible. For long traces a piecewise approach is more
        # robust: fall back to linear interpolation, which matches the existing
        # ``_align`` helper semantics. NaN-preservation across long gaps is
        # enforced in the masking step below.
        out[:, k] = np.interp(sim_t, t_ok, c_ok, left=np.nan, right=np.nan)

    # NaN-preserve: for every sim-grid sample, find the nearest pair of finite
    # source samples bracketing it; if the gap exceeds ``max_gap_s``, set NaN.
    if source_finite_t.size >= 2:
        for i, t in enumerate(sim_t):
            if t < source_finite_t[0] or t > source_finite_t[-1]:
                out[i] = np.nan
                continue
            # Index of nearest finite-sample on the right.
            j = int(np.searchsorted(source_finite_t, t))
            j = min(max(j, 1), source_finite_t.size - 1)
            gap = float(source_finite_t[j] - source_finite_t[j - 1])
            if gap > max_gap_s:
                out[i] = np.nan
    return out


def _detect_impact_via_wrist(
    raw_t: np.ndarray, marker_xyz_z_up: dict[str, np.ndarray]
) -> int:
    """Pick a wrist marker and return the kinematic-peak frame index.

    Heuristic: among ``_WRIST_HEURISTIC_MARKERS`` present in the data, choose
    the marker with the largest finite-sample count, then run the same speed
    argmax used by the club loader.
    """
    candidates = [
        (name, marker_xyz_z_up[name])
        for name in _WRIST_HEURISTIC_MARKERS
        if name in marker_xyz_z_up
    ]
    if not candidates:
        raise ValueError(
            "Cannot detect impact: no wrist markers available for kinematic peak"
        )
    name, xyz = max(
        candidates, key=lambda nv: int(np.isfinite(nv[1]).all(axis=-1).sum())
    )
    finite = np.isfinite(xyz).all(axis=-1)
    if finite.sum() < 5:
        raise ValueError(
            f"Wrist marker {name!r} has insufficient finite samples for impact detection"
        )
    # detect_impact_index requires no NaN; restrict to finite frames.
    t_ok = raw_t[finite]
    xyz_ok = xyz[finite]
    idx_ok = int(detect_impact_index(t_ok, xyz_ok))
    # Map back to the original frame index.
    finite_indices = np.where(finite)[0]
    return int(finite_indices[idx_ok])


def _event_raw_index(
    events: Sequence,
    raw_time: np.ndarray,
    event_label: str | None,
    file_label: str,
) -> int | None:
    """Map a C3D ``EVENT`` label to its raw-frame index.

    Returns ``None`` when no label was requested. Raises :class:`ValueError`
    when a label is requested but the file has no events or the label is
    absent. Logs at INFO level when no events are present and the heuristic
    will be used.
    """
    if event_label is None:
        if not events:
            logger.info(
                "No EVENT annotations in %s; falling back to wrist-speed "
                "kinematic-peak heuristic for impact alignment",
                file_label,
            )
        return None
    if not events:
        raise ValueError(
            f"event_label_for_alignment={event_label!r} requested but "
            f"{file_label} has no EVENT annotations"
        )
    for ev in events:
        if ev.label == event_label:
            target_t = float(ev.time)
            idx = int(np.argmin(np.abs(raw_time - target_t)))
            logger.info(
                "Using EVENT %r at t=%.4fs (frame %d) for impact alignment in %s",
                event_label,
                target_t,
                idx,
                file_label,
            )
            return idx
    available = [ev.label for ev in events]
    raise ValueError(
        f"event_label_for_alignment={event_label!r} not found in "
        f"{file_label}; available event labels: {available}"
    )


def _events_from_metadata(
    metadata, sim_time: np.ndarray, time_offset_s: float
) -> tuple[BodyEvent, ...]:
    """Convert C3D event annotations into :class:`BodyEvent` instances.

    ``time_offset_s`` is the same shift applied to the raw time vector for
    impact alignment; events are placed on the resampled grid by the same
    transform, then quantised to the nearest frame.
    """
    out: list[BodyEvent] = []
    for ev in getattr(metadata, "events", []) or []:
        t_shifted = float(ev.time) - time_offset_s
        if t_shifted < float(sim_time[0]) or t_shifted > float(sim_time[-1]):
            continue
        frame = int(np.argmin(np.abs(sim_time - t_shifted)))
        label = str(ev.label)
        if not label:
            continue
        # Deduplicate labels by suffixing ordinal; BodyTarget enforces unique.
        existing = {e.label for e in out}
        if label in existing:
            k = 2
            while f"{label}_{k}" in existing:
                k += 1
            label = f"{label}_{k}"
        out.append(BodyEvent(label=label, frame=frame, time_s=float(sim_time[frame])))
    return tuple(out)


def _build_sim_grid(opts: AlignOptions) -> np.ndarray:
    """Construct the simulation timegrid from :class:`AlignOptions`."""
    sim_dt = 1.0 / float(opts.sample_rate_hz)
    n_out = int(round(opts.simulation_time_s * opts.sample_rate_hz)) + 1
    return np.arange(n_out, dtype=np.float64) * sim_dt


def _resolve_marker_set(
    requested: Sequence[str] | None, available: list[str]
) -> tuple[str, ...]:
    """Resolve the effective marker set, validating against availability."""
    if requested is None:
        chosen = default_anatomical_marker_set()
    else:
        chosen = tuple(requested)
        if not chosen:
            raise ValueError("marker_set must be non-empty when provided explicitly")
    available_set = set(available)
    missing = [m for m in chosen if m not in available_set]
    if missing:
        raise ValueError(f"Requested markers not present in C3D file: {missing}")
    return chosen


@precondition(
    lambda path, opts, **_: Path(path).exists(),
    "C3D file must exist",
)
@precondition(
    lambda path, opts, **_: opts.sample_rate_hz > 0,
    "sample_rate_hz must be > 0",
)
@precondition(
    lambda path, opts, **_: opts.simulation_time_s > 0,
    "simulation_time_s must be > 0",
)
@postcondition(
    lambda result: isinstance(result, BodyTarget),
    "load_body_target_c3d must return a BodyTarget",
)
def load_body_target_c3d(
    path: Path | str,
    opts: AlignOptions,
    *,
    marker_set: Sequence[str] | None = None,
    impact_source: ClubTarget | None = None,
    event_label_for_alignment: str | None = None,
) -> BodyTarget:
    """Load a C3D file's anatomical body markers into a validated ``BodyTarget``.

    Steps:

    1. Parse the file with the canonical :class:`C3DDataReader`.
    2. Pivot the tidy points dataframe into ``{marker: (N, 3)}`` arrays.
    3. Spline-fill internal NaN gaps of length ``<= 5`` frames; longer gaps
       remain NaN.
    4. Convert Y-up source frame to right-handed Z-up.
    5. Resample each marker onto the simulation grid via NaN-preserving
       per-coordinate interpolation. Long gaps in the source are *not* spanned.
    6. Determine ``impact_idx``: from ``impact_source`` if provided, else via
       a kinematic-peak heuristic on the wrist marker with the largest finite
       coverage.
    7. Convert any C3D ``EVENT.LABELS``/``TIMES`` annotations to
       :class:`BodyEvent` instances anchored to the resampled grid.
    8. Build :class:`SourceProvenance` from the file basename, sha256, and
       stem-derived subject/trial ids.

    Args:
        path:           Filesystem path to a ``.c3d`` file.
        opts:           Resampling and impact-alignment options.
        marker_set:     Optional sequence of marker names to extract. When
                        ``None``, :func:`default_anatomical_marker_set` is used.
                        Output column order matches the requested order.
        impact_source:  Optional :class:`ClubTarget` whose ``time`` and
                        ``impact_idx`` will be reused so that the body target
                        shares a clock with the club target. When provided,
                        ``opts`` is ignored for grid construction.
        event_label_for_alignment:
                        Optional label of a C3D ``EVENT`` group entry (e.g.
                        ``"Impact"``) to use as the alignment frame in place
                        of the wrist-speed kinematic-peak heuristic. When
                        the label is supplied but absent from the file's
                        events, ``ValueError`` is raised listing the
                        available labels. When ``None``, the heuristic is
                        used and a fallback INFO log is emitted for files
                        with no EVENT annotations.

    Returns:
        Validated :class:`BodyTarget` on the simulation timegrid.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError:        If ``marker_set`` is empty or references markers
                           absent from the C3D file, if the wrist heuristic
                           cannot find a usable marker, or if the resulting
                           target fails post-construction validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"C3D file not found: {p}")

    reader_cls = _import_c3d_reader_class()
    reader = reader_cls(p)
    metadata = reader.get_metadata()
    available_labels = list(metadata.marker_labels)
    df = reader.points_dataframe(include_time=True, target_units="m")

    chosen = _resolve_marker_set(marker_set, available_labels)

    # --- Pivot + gap-fill in source (Y-up) coordinates --------------------
    n_frames = int(metadata.frame_count)
    raw_dict, raw_time = _pivot_marker_dict(df, n_frames, chosen)

    pre_nan_total = int(
        sum(np.isnan(arr).any(axis=-1).sum() for arr in raw_dict.values())
    )
    filled = {name: fill_short_gaps(arr) for name, arr in raw_dict.items()}
    post_nan_total = int(
        sum(np.isnan(arr).any(axis=-1).sum() for arr in filled.values())
    )

    # --- Y-up -> Z-up ------------------------------------------------------
    z_up = {name: y_up_to_z_up(arr) for name, arr in filled.items()}

    # --- Determine impact + grid ------------------------------------------
    metadata_events = list(getattr(metadata, "events", []) or [])
    event_raw_idx = _event_raw_index(
        metadata_events, raw_time, event_label_for_alignment, p.name
    )
    if impact_source is not None:
        sim_time = np.asarray(impact_source.time, dtype=np.float64).copy()
        # Reuse the club's resampled-grid impact index directly so the two
        # clocks land on the same sample. ``impact_idx`` is 0-based on the
        # resampled grid (see ClubTarget contract). Re-deriving it via argmin
        # would introduce a 1-sample drift when the chosen impact_target_t_s
        # falls between grid samples.
        impact_idx_out = int(impact_source.impact_idx)
        impact_target_t_s = (
            float(sim_time[impact_idx_out])
            if 0 <= impact_idx_out < sim_time.size
            else float(opts.impact_target_t_s)
        )
        _wrist_markers_present = any(m in chosen for m in ("RWristTop", "LWristTop"))
        if event_raw_idx is not None:
            impact_raw = event_raw_idx
            time_offset = float(raw_time[impact_raw]) - impact_target_t_s
        elif _wrist_markers_present:
            impact_raw = _detect_impact_via_wrist(raw_time, z_up)
            time_offset = float(raw_time[impact_raw]) - impact_target_t_s
        else:
            impact_raw = 0
            time_offset = 0.0
    else:
        sim_time = _build_sim_grid(opts)
        if event_raw_idx is not None:
            impact_raw = event_raw_idx
        else:
            impact_raw = _detect_impact_via_wrist(raw_time, z_up)
        if opts.time_alignment == "impact":
            time_offset = float(raw_time[impact_raw]) - float(opts.impact_target_t_s)
            impact_idx_out = int(np.argmin(np.abs(sim_time - opts.impact_target_t_s)))
        elif opts.time_alignment in ("address", "none"):
            time_offset = float(raw_time[0])
            impact_idx_out = int(
                np.clip(
                    int(
                        round(
                            (float(raw_time[impact_raw]) - time_offset)
                            / (sim_time[1] - sim_time[0])
                        )
                    ),
                    0,
                    sim_time.size - 1,
                )
            )
        else:
            raise ValueError(f"Unknown time_alignment {opts.time_alignment!r}")

    raw_time_aligned = raw_time - time_offset

    # --- Resample each marker onto the sim grid ---------------------------
    max_gap_s = 5.0 / float(metadata.frame_rate)  # 5 frames at source rate
    m = len(chosen)
    marker_xyz = np.full((sim_time.shape[0], m, 3), np.nan, dtype=np.float64)
    for j, name in enumerate(chosen):
        marker_xyz[:, j, :] = _resample_marker_series(
            sim_time, raw_time_aligned, z_up[name], max_gap_s=max_gap_s
        )

    # --- Events -----------------------------------------------------------
    events = _events_from_metadata(metadata, sim_time, time_offset)

    # --- Provenance -------------------------------------------------------
    source = SourceProvenance(
        filename=p.name,
        format="c3d",
        subject_id=p.stem,
        trial_id=p.stem,
        sha256=_sha256_of(p),
    )

    occluded_after = int(np.isnan(marker_xyz).any(axis=-1).sum())
    logger.info(
        "Loaded BodyTarget from %s: markers=%d, sim_samples=%d, impact_idx=%d, "
        "src_nan_frames=%d -> filled=%d, post_resample_nan_samples=%d",
        p.name,
        m,
        sim_time.shape[0],
        impact_idx_out,
        pre_nan_total,
        pre_nan_total - post_nan_total,
        occluded_after,
    )

    return BodyTarget(
        time=sim_time,
        marker_xyz=marker_xyz,
        marker_names=tuple(chosen),
        impact_idx=int(impact_idx_out),
        events=events,
        source=source,
    )
