"""Swing read-out of both wrists in wrist-sensor (HackMotion) quantities (#15275)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from src.shared.python.anatomical_angles.full_body_v1 import (
    calibrate_neutral,
    full_body_v1_wrist_forearm,
    side_chain,
)
from src.shared.python.anatomical_angles.wrist import WristForearm, check_side
from src.shared.python.model_appearance.skinned_body import spec_body_poses

DEFAULT_LEAD_SIDE = "L"
MODEL_JOINT_OBSERVABILITY = (
    "model_joint_angles: derived from model joint angles, not sensor data"
)


def _to_json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("i", "u"):
            return [int(v) for v in obj.tolist()]
        return [None if math.isnan(float(v)) else float(v) for v in obj.tolist()]
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _to_json_ready(getattr(obj, f.name)) for f in fields(obj)}
    return obj


@dataclass(frozen=True)
class SwingWristReadoutSettings:
    """Configuration for :func:`swing_wrist_readout`."""

    lead_side: str = DEFAULT_LEAD_SIDE
    dt_s: float | None = None
    address_frame: int = 0
    receipt_path: str = "spec_joint_kinematics"
    observability: str = MODEL_JOINT_OBSERVABILITY
    pelvis_body: str = "LowerTorso"
    trajectory_ndim: int = 2
    min_frames: int = 1
    rot_dim: int = 3
    min_dt_s: float = 0.0


@dataclass(frozen=True)
class SideWristReadout:
    """Per-side wrist and forearm series in degrees, absolute and vs. address."""

    role: str
    side: str
    flexion_deg: np.ndarray
    ulnar_deg: np.ndarray
    pronation_deg: np.ndarray
    flexion_rel_address_deg: np.ndarray
    ulnar_rel_address_deg: np.ndarray
    pronation_rel_address_deg: np.ndarray
    observability: str

    def as_mapping(self) -> dict[str, Any]:
        """Plain JSON-ready dictionary for this side."""
        return dict(_to_json_ready(self))


@dataclass(frozen=True)
class SwingWristReadout:
    """Both-wrist swing read-out in degrees with provenance and JSON export."""

    lead: SideWristReadout
    trail: SideWristReadout
    frame_index: np.ndarray
    time_s: np.ndarray | None
    lead_side: str
    trail_side: str
    address_frame: int
    observability: str

    def as_mapping(self) -> dict[str, Any]:
        """Plain JSON-ready representation of the read-out."""
        return dict(_to_json_ready(self))


def _short_rotations(
    spec: Mapping[str, Any], q_arr: np.ndarray, dim: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    order = list(spec["coordinate_order"])
    neutral = {
        k.rsplit("/", 1)[-1]: v[:dim, :dim] for k, v in spec_body_poses(spec).items()
    }
    chains = (side_chain("L"), side_chain("R"))
    needed = {b for c in chains for b in (c.humerus, c.forearm_body, c.hand)}
    series = {b: np.empty((len(q_arr), dim, dim)) for b in needed}
    for i, row in enumerate(q_arr):
        poses = spec_body_poses(spec, dict(zip(order, row.tolist(), strict=True)))
        for full_name, mat in poses.items():
            if (short := full_name.rsplit("/", 1)[-1]) in series:
                series[short][i] = mat[:dim, :dim]
    return neutral, series


def _side_readout(
    role: str, side: str, wf: WristForearm, addr: int, obs: str
) -> SideWristReadout:
    f, u = wf.wrist.flexion_deg, wf.wrist.ulnar_deviation_deg
    p = wf.forearm.pronation_deg
    return SideWristReadout(
        role, side, f, u, p, f - f[addr], u - u[addr], p - p[addr], obs
    )


def swing_wrist_readout(
    spec: Mapping[str, Any],
    q_trajectory: Any,
    *,
    settings: SwingWristReadoutSettings | None = None,
) -> SwingWristReadout:
    """Compute per-frame lead/trail wrist and forearm angles in degrees.

    Preconditions: ``spec`` has ``coordinate_order`` and ``joints``; ``q_trajectory``
    is a finite ``(N, D)`` array. Postcondition: returns :class:`SwingWristReadout`.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"spec must be a Mapping, got {type(spec).__name__}")
    if "coordinate_order" not in spec or "joints" not in spec:
        raise ValueError("spec must contain 'coordinate_order' and 'joints'")
    cfg = SwingWristReadoutSettings() if settings is None else settings
    lead = check_side(cfg.lead_side)
    trail = "R" if lead == "L" else "L"
    q_arr = np.asarray(q_trajectory, dtype=float)
    n_coords = len(spec["coordinate_order"])
    if q_arr.ndim != cfg.trajectory_ndim:
        raise ValueError(f"q_trajectory must be 2D, got shape {q_arr.shape}")
    if q_arr.shape[0] < cfg.min_frames:
        raise ValueError("q_trajectory must contain at least one frame")
    if q_arr.shape[1] != n_coords:
        raise ValueError(f"q_trajectory column count {q_arr.shape[1]} != {n_coords}")
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q_trajectory must contain only finite values")
    n_frames = q_arr.shape[0]
    if not (0 <= cfg.address_frame < n_frames):
        raise ValueError(f"address_frame {cfg.address_frame!r} out of range")
    time_s: np.ndarray | None = None
    if cfg.dt_s is not None:
        if not (math.isfinite(cfg.dt_s) and cfg.dt_s > cfg.min_dt_s):
            raise ValueError(f"dt_s must be finite and positive, got {cfg.dt_s!r}")
        time_s = np.arange(n_frames, dtype=float) * float(cfg.dt_s)
    neutral, series = _short_rotations(spec, q_arr, cfg.rot_dim)
    pelvis = neutral.get(cfg.pelvis_body)
    cal_l = calibrate_neutral("L", neutral, pelvis_rotation=pelvis)
    cal_r = calibrate_neutral("R", neutral, pelvis_rotation=pelvis)
    by_side = {
        "L": full_body_v1_wrist_forearm(cal_l, series, receipt_path=cfg.receipt_path),
        "R": full_body_v1_wrist_forearm(cal_r, series, receipt_path=cfg.receipt_path),
    }
    addr, obs = cfg.address_frame, cfg.observability
    return SwingWristReadout(
        lead=_side_readout("lead", lead, by_side[lead], addr, obs),
        trail=_side_readout("trail", trail, by_side[trail], addr, obs),
        frame_index=np.arange(n_frames, dtype=int),
        time_s=time_s,
        lead_side=lead,
        trail_side=trail,
        address_frame=addr,
        observability=obs,
    )
