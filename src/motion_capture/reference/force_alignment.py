"""Force/torque series video alignment, trace import, and viewport projection (FTO-24, #11309).

Coordinate Conventions and Vector Transformation Rules
======================================================
| Operation / Frame        | X-axis         | Y-axis              | Z-axis               | Handedness / Parity | Notes                                                    |
|--------------------------|----------------|---------------------|----------------------|---------------------|----------------------------------------------------------|
| Canonical Z-up           | +X (forward)   | +Y (golfer's left)  | +Z (up)              | Right-handed (+1)   | ADR-0026 simulation coordinate convention                |
| ADR-0041 World           | +X (forward)   | +Y (up)             | +Z (golfer's right)  | Right-handed (+1)   | x_adr = x_can, y_adr = z_can, z_adr = -y_can             |
| Lateral Mirror (Points)  | x_can          | -y_can              | z_can                | Left-handed (-1)    | Flips lateral coordinate across sagittal plane           |
| Lateral Mirror (Forces)  | F_x            | -F_y                | F_z                  | Polar vector        | Transforms as coordinates: F_m = M_can @ F_can           |
| Lateral Mirror (Torques) | -tau_x         | tau_y               | -tau_z               | Axial pseudovector  | Extra parity flip: tau_m = det(M) * M_can @ tau_can     |
| Scene Rotation           | R_reg @ v      | R_reg @ v           | R_reg @ v            | Proper rotation (+1)| Applied to points and rotated vectors identically        |
| Scene Translation/Scale  | s*R@p + t      | s*R@p + t           | s*R@p + t            | Similarity          | Points scaled and translated; force/torque vectors unscaled|
| Viewport Origin Wrench   | sum(F_x)       | sum(F_y)            | sum(F_z)             | Spatial wrench      | tau_origin = tau_pt + r x F summed over EXTERNAL/CONTACT |
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    canonical_z_up_to_adr0041_world,
)
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.series import ForceTorqueSeries
from src.shared.python.motion_matching.force_torque import (
    SpatialWrench,
    transform_wrench,
)
from src.shared.python.simulation_backends.protocol import Trace
from src.shared.python.simulation_backends.trace_io import read_trace

logger = logging.getLogger(__name__)

HDF5_GROUP_NAME = "force_torque_series"
_KNOWN_POINT_KEYS = (
    "root_point_m",
    "wrench_point_m",
    "external_wrench_point_m",
    "root_position_m",
    "root_position",
    "wrench_point",
)


def force_frame_for_video(
    series: ForceTorqueSeries,
    *,
    video_time_s: float,
    registration: ReferenceRegistration,
    max_gap_s: float | None = None,
) -> ForceTorqueFrame | None:
    """Sample and spatially register a ForceTorqueSeries for a specific video timestamp.

    Applies:
      1. Time mapping: video_time_s -> scene_time -> reference_time -> series.frame_at
      2. Spatial registration:
         - Points: canonical_z_up_to_adr0041_world + registration point placement
         - Forces: rotated by canonical-to-ADR-0041 and registration rotation (polar vector)
         - Torques: rotated with axial pseudovector parity under lateral mirroring
    """
    v_time = float(video_time_s)
    if not math.isfinite(v_time):
        raise ValueError(f"video_time_s must be finite, got {video_time_s}")

    scene_t = registration.scene_time(v_time)
    ref_t = float(registration.time_mapping.scene_to_reference(scene_t))
    eff_max_gap = (
        float(max_gap_s) if max_gap_s is not None else float(registration.max_gap_s)
    )

    source_frame = series.frame_at(ref_t, max_gap_s=eff_max_gap)
    if source_frame is None:
        return None

    r_reg = np.asarray(registration.transform.rotation, dtype=np.float64)
    mirror = registration.mirror_lateral

    transformed_wrenches: list[OverlayWrench] = []
    for w in source_frame.wrenches:
        pt_scene = registration.place_points(np.asarray(w.point_m, dtype=np.float64))
        pt_tuple = (float(pt_scene[0]), float(pt_scene[1]), float(pt_scene[2]))

        f_tuple: tuple[float, float, float] | None = None
        if w.force_n is not None:
            f_can = np.asarray(w.force_n, dtype=np.float64)
            f_can_m = (
                np.array([f_can[0], -f_can[1], f_can[2]], dtype=np.float64)
                if mirror
                else f_can
            )
            f_adr = canonical_z_up_to_adr0041_world(f_can_m)
            f_scene = r_reg @ f_adr
            f_tuple = (float(f_scene[0]), float(f_scene[1]), float(f_scene[2]))

        tau_tuple: tuple[float, float, float] | None = None
        if w.torque_nm is not None:
            tau_can = np.asarray(w.torque_nm, dtype=np.float64)
            # Axial vector under lateral reflection gains extra -1 factor:
            # tau_m = det(M) * (M @ tau) = (-1) * [tau_x, -tau_y, tau_z] = [-tau_x, tau_y, -tau_z]
            tau_can_m = (
                np.array([-tau_can[0], tau_can[1], -tau_can[2]], dtype=np.float64)
                if mirror
                else tau_can
            )
            tau_adr = canonical_z_up_to_adr0041_world(tau_can_m)
            tau_scene = r_reg @ tau_adr
            tau_tuple = (float(tau_scene[0]), float(tau_scene[1]), float(tau_scene[2]))

        transformed_wrenches.append(
            OverlayWrench(
                kind=w.kind,
                label=w.label,
                body=w.body,
                point_m=pt_tuple,
                force_n=f_tuple,
                torque_nm=tau_tuple,
                source=w.source,
            )
        )

    axial_frame: AxialLoadFrame | None = None
    if source_frame.axial_loads is not None:
        axial_frame = AxialLoadFrame(
            time_s=v_time,
            values_n=dict(source_frame.axial_loads.values_n),
            source=source_frame.axial_loads.source,
        )

    merged_meta = dict(source_frame.metadata)
    merged_meta["registration_scale"] = float(registration.transform.scale)

    return ForceTorqueFrame(
        time_s=v_time,
        engine=source_frame.engine,
        wrenches=tuple(transformed_wrenches),
        axial_loads=axial_frame,
        world_frame="adr0041_world",
        units=source_frame.units,
        metadata=merged_meta,
    )


def write_trace_forces(path: Path | str, series: ForceTorqueSeries) -> None:
    """Store ForceTorqueSeries in a Trace HDF5 file under group 'force_torque_series'."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file does not exist: {p}")

    json_str = json.dumps(series.to_dict())
    json_bytes = np.frombuffer(json_str.encode("utf-8"), dtype=np.uint8)
    times = np.array([f.time_s for f in series.frames], dtype=np.float64)

    with h5py.File(p, "a") as h5:
        if HDF5_GROUP_NAME in h5:
            del h5[HDF5_GROUP_NAME]
        grp = h5.create_group(HDF5_GROUP_NAME)
        grp.attrs["schema_version"] = series.SCHEMA_VERSION
        grp.attrs["engine"] = series.engine
        grp.create_dataset("json_bytes", data=json_bytes, compression="gzip")
        grp.create_dataset("time_s", data=times)


def _parse_declared_point(value: Any) -> tuple[float, float, float] | None:
    """Parse declared 3D point from string, array, or sequence."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                return (float(parsed[0]), float(parsed[1]), float(parsed[2]))
        except (json.JSONDecodeError, ValueError, TypeError):
            parts = [s.strip() for s in value.split(",")]
            if len(parts) == 3:
                try:
                    return (float(parts[0]), float(parts[1]), float(parts[2]))
                except ValueError:
                    return None
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


def load_trace_forces(path: Path | str) -> ForceTorqueSeries | None:
    """Load ForceTorqueSeries from a Trace HDF5 file, or fallback to Trace.wrench."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file does not exist: {p}")

    with h5py.File(p, "r") as h5:
        if HDF5_GROUP_NAME in h5:
            grp = h5[HDF5_GROUP_NAME]
            if "json_bytes" in grp:
                raw_bytes = bytes(grp["json_bytes"][()])
                data = json.loads(raw_bytes.decode("utf-8"))
                return ForceTorqueSeries.from_dict(data)

    trace = read_trace(p)
    if not isinstance(trace, Trace) or trace.wrench is None:
        return None

    # Check if trace.meta declares a wrench point
    declared_point: tuple[float, float, float] | None = None
    for key in _KNOWN_POINT_KEYS:
        val = trace.meta.get(key)
        if val is not None:
            declared_point = _parse_declared_point(val)
            if declared_point is not None:
                break

    if declared_point is None:
        logger.debug(
            "Trace carries wrench array but declares no wrench point in meta; skipping"
        )
        return None

    w_arr = np.asarray(trace.wrench, dtype=np.float64)
    if w_arr.ndim != 2 or w_arr.shape[1] != 6:
        return None

    frames: list[ForceTorqueFrame] = []
    for i, t_val in enumerate(trace.t):
        w_row = w_arr[i]
        f_vec = (float(w_row[0]), float(w_row[1]), float(w_row[2]))
        tau_vec = (float(w_row[3]), float(w_row[4]), float(w_row[5]))
        w = OverlayWrench(
            kind=WrenchKind.EXTERNAL,
            label="external:root",
            body="root",
            point_m=declared_point,
            force_n=f_vec,
            torque_nm=tau_vec,
            source=trace.backend,
        )
        frames.append(
            ForceTorqueFrame(
                time_s=float(t_val),
                engine=trace.backend,
                wrenches=(w,),
                world_frame="world_Zup",
            )
        )

    return ForceTorqueSeries(engine=trace.backend, frames=tuple(frames))


def series_to_viewport_payload_wrench(
    series: ForceTorqueSeries,
) -> npt.NDArray[np.float64]:
    """Compute total external and contact wrench about the world origin for each frame.

    Returns array of shape (T, 6) with layout [fx, fy, fz, tx, ty, tz].
    """
    t_count = len(series.frames)
    payload = np.zeros((t_count, 6), dtype=np.float64)
    origin_pt = (0.0, 0.0, 0.0)

    for i, frame in enumerate(series.frames):
        f_net = np.zeros(3, dtype=np.float64)
        tau_net = np.zeros(3, dtype=np.float64)

        for w in frame.wrenches:
            if w.kind not in (WrenchKind.EXTERNAL, WrenchKind.CONTACT):
                continue

            f_vec = w.force_n if w.force_n is not None else (0.0, 0.0, 0.0)
            tau_vec = w.torque_nm if w.torque_nm is not None else (0.0, 0.0, 0.0)

            sw = SpatialWrench(
                application_frame="world",
                point_m=w.point_m,
                force_n=f_vec,
                torque_nm=tau_vec,
            )
            sw_origin = transform_wrench(
                sw,
                target_frame="world_origin",
                new_point_m=origin_pt,
                rotation_matrix=np.eye(3),
            )
            f_net += np.asarray(sw_origin.force_n, dtype=np.float64)
            tau_net += np.asarray(sw_origin.torque_nm, dtype=np.float64)

        payload[i, :3] = f_net
        payload[i, 3:] = tau_net

    return payload
