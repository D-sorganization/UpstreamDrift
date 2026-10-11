"""Alignment of force/torque series with video frames and trace import (FTO-24, #11309).

Authorities and conventions:
- Space: canonical_z_up_to_adr0041_world maps canonical Z-up to ADR-0041 camera world.
- Time: ReferenceRegistration.time_mapping.scene_to_reference.
- Vectors: forces transform as polar vectors (F_world = A @ F_can);
  torques transform as axial vectors / pseudovectors (tau_world = det(A) * (A @ tau_can)).
- Viewport payload: sums contact and external wrenches about origin (T, 6).
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.motion_capture.reference.registration import (
    ReferenceRegistration,
)
from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.contracts import precondition
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.motion_matching.force_torque import validate_vec3

__all__ = [
    "force_frame_for_video",
    "load_trace_forces",
    "series_to_viewport_payload_wrench",
    "write_trace_forces",
]

_CANONICAL_TO_ADR0041 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
_CANONICAL_TO_ADR0041_MIRRORED = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _compute_direction_matrix(
    registration: ReferenceRegistration,
) -> tuple[np.ndarray, float]:
    """Compute the 3x3 direction transformation matrix and its determinant."""
    c_eff = (
        _CANONICAL_TO_ADR0041_MIRRORED
        if registration.mirror_lateral
        else _CANONICAL_TO_ADR0041
    )
    r = np.asarray(registration.transform.rotation, dtype=np.float64)
    a = r @ c_eff
    det_a = float(np.linalg.det(a))
    return a, det_a


def _transform_wrench(
    w: OverlayWrench,
    point_world: tuple[float, float, float],
    a_matrix: np.ndarray,
    det_a: float,
) -> OverlayWrench:
    """Transform an individual wrench to ADR-0041 world coordinates."""
    force_n: tuple[float, float, float] | None = None
    if w.force_n is not None:
        f_vec = a_matrix @ np.asarray(w.force_n, dtype=np.float64)
        force_n = (float(f_vec[0]), float(f_vec[1]), float(f_vec[2]))

    torque_nm: tuple[float, float, float] | None = None
    if w.torque_nm is not None:
        t_vec = det_a * (a_matrix @ np.asarray(w.torque_nm, dtype=np.float64))
        torque_nm = (float(t_vec[0]), float(t_vec[1]), float(t_vec[2]))

    return OverlayWrench(
        kind=w.kind,
        label=w.label,
        body=w.body,
        point_m=point_world,
        force_n=force_n,
        torque_nm=torque_nm,
        source=w.source,
    )


def force_frame_for_video(
    series: ForceTorqueSeries,
    *,
    video_time_s: float,
    registration: ReferenceRegistration,
    max_gap_s: float = 0.1,
) -> ForceTorqueFrame | None:
    """Align and transform a force/torque frame onto a video timestamp."""
    if not math.isfinite(video_time_s):
        raise ValueError("video_time_s must be finite")

    ref_time_s = float(registration.time_mapping.scene_to_reference(video_time_s))
    frame = series.frame_at(ref_time_s, max_gap_s=max_gap_s)
    if frame is None:
        return None

    a_matrix, det_a = _compute_direction_matrix(registration)

    wrenches: list[OverlayWrench] = []
    if frame.wrenches:
        raw_points = np.array([w.point_m for w in frame.wrenches], dtype=np.float64)
        world_points = registration.place_points(raw_points)
        for w, pt in zip(frame.wrenches, world_points, strict=True):
            p_tuple = (float(pt[0]), float(pt[1]), float(pt[2]))
            wrenches.append(_transform_wrench(w, p_tuple, a_matrix, det_a))

    axial: AxialLoadFrame | None = None
    if frame.axial_loads is not None:
        axial = AxialLoadFrame(
            time_s=float(video_time_s),
            values_n=frame.axial_loads.values_n,
            source=frame.axial_loads.source,
        )

    metadata = dict(frame.metadata)
    metadata["registration_scale"] = float(registration.transform.scale)
    metadata["reference_time_s"] = float(ref_time_s)
    metadata["mirrored"] = bool(registration.mirror_lateral)

    return ForceTorqueFrame(
        time_s=float(video_time_s),
        engine=frame.engine,
        world_frame="adr0041_world",
        wrenches=tuple(wrenches),
        axial_loads=axial,
        metadata=metadata,
    )


def write_trace_forces(path: Path, series: ForceTorqueSeries) -> None:
    """Persist a ForceTorqueSeries into an HDF5 trace file under force_torque_series."""
    mode = "a" if path.exists() else "w"
    buf = io.BytesIO()
    series.to_npz(buf)
    buf.seek(0)

    with np.load(buf, allow_pickle=False) as npz, h5py.File(path, mode) as handle:
        if "force_torque_series" in handle:
            del handle["force_torque_series"]
        grp = handle.create_group("force_torque_series")
        grp.attrs["schema_version"] = "force-torque-frame-v1"
        for key in npz.files:
            val = npz[key]
            if val.dtype.kind == "U":
                grp.create_dataset(
                    key, data=val.astype(h5py.string_dtype(encoding="utf-8"))
                )
            else:
                grp.create_dataset(key, data=val)


def _decode_hdf5_array(raw: Any) -> np.ndarray:
    """Decode HDF5 byte string or array to Unicode string array or numpy array."""
    if isinstance(raw, (bytes, str)):
        decoded = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return np.array(decoded, dtype=str)
    arr = np.asarray(raw)
    if arr.dtype.kind in ("S", "O"):
        if arr.ndim == 0:
            val = arr.item()
            decoded = val.decode("utf-8") if isinstance(val, bytes) else str(val)
            return np.array(decoded, dtype=str)
        decoded_list = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr.flat
        ]
        return np.array(decoded_list, dtype=str).reshape(arr.shape)
    return arr


def _load_group_forces(grp: h5py.Group) -> ForceTorqueSeries:
    """Deserialize ForceTorqueSeries from an HDF5 group."""
    data: dict[str, np.ndarray] = {}
    for key in grp:
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            data[key] = _decode_hdf5_array(item[()])
    buf = io.BytesIO()
    np.savez(buf, **data)  # type: ignore[arg-type]
    buf.seek(0)
    return ForceTorqueSeries.from_npz(buf)


def _parse_meta_point(handle: h5py.File) -> tuple[float, float, float] | None:
    """Extract declared wrench_point or root_point from HDF5 metadata attributes."""
    point_val = (
        handle.attrs.get("meta_wrench_point")
        or handle.attrs.get("meta_root_point")
        or handle.attrs.get("wrench_point")
        or handle.attrs.get("root_point")
    )
    if point_val is None:
        return None
    if isinstance(point_val, bytes):
        point_val = point_val.decode("utf-8")
    if isinstance(point_val, str):
        try:
            point_data = json.loads(point_val)
        except json.JSONDecodeError:
            return None
    else:
        point_data = point_val
    try:
        return validate_vec3(point_data, "wrench_point")
    except (ValueError, TypeError):
        return None


def _load_fallback_trace_forces(handle: h5py.File) -> ForceTorqueSeries | None:
    """Construct ForceTorqueSeries from root wrench dataset and declared point."""
    if "wrench" not in handle or "t" not in handle:
        return None
    wrench_arr = np.asarray(handle["wrench"][()], dtype=np.float64)
    if wrench_arr.ndim != 2 or wrench_arr.shape[1] != 6:
        return None

    point_m = _parse_meta_point(handle)
    if point_m is None:
        return None

    times = np.asarray(handle["t"][()], dtype=np.float64)
    if len(times) != wrench_arr.shape[0] or len(times) == 0:
        return None
    if np.any(np.diff(times) <= 0):
        return None

    backend = str(handle.attrs.get("backend", "simulation"))
    body = str(handle.attrs.get("meta_wrench_body", "root"))
    frames: list[ForceTorqueFrame] = []
    for t_val, row in zip(times, wrench_arr, strict=True):
        w = OverlayWrench(
            kind=WrenchKind.EXTERNAL,
            label="external:trace_wrench",
            body=body,
            point_m=point_m,
            force_n=(float(row[0]), float(row[1]), float(row[2])),
            torque_nm=(float(row[3]), float(row[4]), float(row[5])),
            source=backend,
        )
        frames.append(
            ForceTorqueFrame(time_s=float(t_val), engine=backend, wrenches=(w,))
        )
    return ForceTorqueSeries(frames=tuple(frames), engine=backend)


def load_trace_forces(path: Path) -> ForceTorqueSeries | None:
    """Load ForceTorqueSeries from trace HDF5 group or declared fallback root wrench."""
    if not path.is_file():
        return None
    try:
        with h5py.File(path, "r") as handle:
            if "force_torque_series" in handle:
                grp = handle["force_torque_series"]
                if isinstance(grp, h5py.Group):
                    return _load_group_forces(grp)
                return None
            return _load_fallback_trace_forces(handle)
    except (OSError, ValueError, TypeError, KeyError):
        return None


@precondition(lambda series: series is not None, "Valid ForceTorqueSeries required")
def series_to_viewport_payload_wrench(series: ForceTorqueSeries) -> np.ndarray:
    """Compute (T, 6) net wrench of contact and external wrenches about world origin."""
    n_frames = len(series.frames)
    payload = np.zeros((n_frames, 6), dtype=np.float64)

    for i, frame in enumerate(series.frames):
        for w in frame.wrenches:
            if w.kind not in (WrenchKind.CONTACT, WrenchKind.EXTERNAL):
                continue
            f_vec = (
                np.asarray(w.force_n, dtype=np.float64)
                if w.force_n is not None
                else np.zeros(3, dtype=np.float64)
            )
            t_vec = (
                np.asarray(w.torque_nm, dtype=np.float64)
                if w.torque_nm is not None
                else np.zeros(3, dtype=np.float64)
            )
            r_vec = np.asarray(w.point_m, dtype=np.float64)
            # Moment of force about origin: r x F
            t_origin = t_vec + np.cross(r_vec, f_vec)
            payload[i, 0:3] += f_vec
            payload[i, 3:6] += t_origin

    return payload
