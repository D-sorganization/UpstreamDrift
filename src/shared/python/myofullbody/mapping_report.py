"""Mapping receipts: key frames, ROM exceedance, hinge checks, muscle lengths (#11644).

Everything here is evidence about :class:`mapping.MyoMapper` poses.  The
primary mapping uses the ``extend`` ROM policy so each segment reaches its
target orientation; the demand beyond MyoFullBody's own limits is reported per
joint (the "ROM clamps"), and a second ``clamp``-policy run at the key frames
quantifies what honouring the limits would cost in orientation error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.myofullbody import anatomy
from src.shared.python.myofullbody.mapping import (
    SEGMENT_JOINTS,
    MappedPose,
    MyoMapper,
)

Array = np.ndarray
CLUB_BODY = "Clubface Vector"
THREE_DOF = ("pelvis", "thorax", "humerus_l", "humerus_r", "femur_l", "femur_r")
TOLERANCE_DEG_THREE_DOF = 1.0
TOLERANCE_DEG_STRUCTURAL = 15.0
# (spec coordinate, Myo joint, sign): flexion hinges whose physical axis both models
# define.  Spec knee flexion is negative and elbow flexion runs opposite to the
# MyoFullBody sign; ankle dorsiflexion agrees.  The signs are fixed by convention,
# not fitted, so a wrong mapping cannot hide behind a sign flip.
HINGE_PAIRS = (
    ("knee_angle_l", "knee_angle_l", -1.0),
    ("knee_angle_r", "knee_angle_r", -1.0),
    ("ankle_angle_l", "ankle_angle_l", 1.0),
    ("ankle_angle_r", "ankle_angle_r", 1.0),
    ("LEInput", "elbow_flexion_l", -1.0),
    ("REInput", "elbow_flexion_r", -1.0),
)
LENGTH_OUT_TOL = 0.02


@dataclass(frozen=True)
class KeyFrames:
    """Indices of the swing events used in receipts."""

    address: int
    top: int
    impact: int
    finish: int

    def as_dict(self) -> dict[str, int]:
        return {
            "address": self.address,
            "top": self.top,
            "impact": self.impact,
            "finish": self.finish,
        }


def key_frames(mapper: MyoMapper, q: Array, dt_s: float) -> KeyFrames:
    """Address, top (peak torso turn in the first 60 %), impact and finish.

    Impact is the maximum speed of the club body origin, as in the OpenSim
    musculoskeletal pipeline.

    Raises:
        ValueError: if ``q`` is not ``(frames, 44)`` with at least 10 frames.
    """
    require(q.ndim == 2 and q.shape[0] >= 10, "q must be (frames >= 10, nv)")
    order = mapper.spec.coordinate_order
    torso = q[:, order.index("TorsoInput")]
    n = q.shape[0]
    top = int(np.argmax(np.abs(torso[: int(0.6 * n)] - torso[0])))
    body = [
        i
        for i in range(mapper.spec.model.nbody)
        if (mapper.spec.model.body(i).name or "").endswith(CLUB_BODY)
    ][0]
    stride = max(1, n // 400)
    idx = np.arange(0, n, stride)
    pos = np.empty((len(idx), 3))
    for j, k in enumerate(idx):
        mapper.spec.set(q[k])
        pos[j] = mapper.spec.position(body)
    speed = np.linalg.norm(np.gradient(pos, idx * dt_s, axis=0), axis=1)
    impact = int(idx[int(np.argmax(speed))])
    return KeyFrames(0, top, impact, n - 2)  # last step midpoint


def map_sequence(
    mapper: MyoMapper, q: Array, indices: Array, *, bounded: bool = True
) -> list[MappedPose]:
    """Map the frames ``indices`` in order, warm-starting each from the previous."""
    poses: list[MappedPose] = []
    guess: Array | None = None
    for k in indices:
        pose = mapper.map_pose(q[int(k)], guess=guess, bounded=bounded)
        poses.append(pose)
        guess = pose.qpos
    return poses


def segment_errors(poses: list[MappedPose]) -> dict[str, dict[str, float]]:
    """Mean, 95th percentile and max orientation error (deg) per segment."""
    out: dict[str, dict[str, float]] = {}
    for name in anatomy.SEGMENTS:
        values = np.array([p.error_deg[name] for p in poses])
        out[name] = {
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    return out


def tolerance_for(segment: str) -> float:
    """Orientation tolerance (deg): tight for 3-DOF chains, loose where a 1-2 DOF
    MyoFullBody chain cannot represent the spec's 3-D segment orientation."""
    return TOLERANCE_DEG_THREE_DOF if segment in THREE_DOF else TOLERANCE_DEG_STRUCTURAL


def rom_exceedance(
    mapper: MyoMapper, poses: list[MappedPose]
) -> dict[str, dict[str, float]]:
    """Per solved joint: fraction of frames beyond the limit and the peak overshoot (deg).

    ``poses`` must come from the extended-ROM mapping.  Only joints that exceed
    their limit at least once are returned.
    """
    out: dict[str, dict[str, float]] = {}
    model = mapper.model
    for joints in SEGMENT_JOINTS.values():
        for name in joints:
            lo, hi = mapper.bounds((name,), "clamp")
            adr = int(model.joint(name).qposadr[0])
            values = np.array([p.qpos[adr] for p in poses])
            over = np.maximum(values - hi[0], lo[0] - values)
            if (over > 1e-9).any():
                out[name] = {
                    "fraction_beyond": float((over > 1e-9).mean()),
                    "peak_overshoot_deg": float(np.degrees(over.max())),
                    "range_deg": [float(np.degrees(lo[0])), float(np.degrees(hi[0]))],
                    "demand_deg": [
                        float(np.degrees(values.min())),
                        float(np.degrees(values.max())),
                    ],
                }
    return out


def hinge_agreement(
    mapper: MyoMapper, q: Array, indices: Array, poses: list[MappedPose]
) -> dict[str, dict[str, float]]:
    """Independent check: solved MyoFullBody hinge angle versus the spec coordinate.

    The RMS of ``myo - sign * spec`` in degrees (with the conventional signs of
    :data:`HINGE_PAIRS`) measures whether the orientation mapping reproduces the
    same physical flexion; the mean offset is the structural carrying angle.
    """
    order = mapper.spec.coordinate_order
    out: dict[str, dict[str, float]] = {}
    for spec_name, myo_name, sign in HINGE_PAIRS:
        spec = q[np.asarray(indices, dtype=int), order.index(spec_name)]
        adr = int(mapper.model.joint(myo_name).qposadr[0])
        myo = np.array([p.qpos[adr] for p in poses])
        diff = np.degrees(myo - sign * spec)
        out[f"{spec_name}->{myo_name}"] = {
            "sign": sign,
            "rms_deg": float(np.sqrt(np.mean(diff**2))),
            "max_abs_deg": float(np.abs(diff).max()),
            "mean_offset_deg": float(diff.mean()),
        }
    return out


def muscle_lengths(model: Any, data: Any, poses: list[MappedPose]) -> dict[str, Array]:
    """Actuator (tendon) length and normalised length for each pose, ``(frames, nu)``."""
    import mujoco

    lengths = np.empty((len(poses), model.nu))
    for k, pose in enumerate(poses):
        data.qpos[:] = pose.qpos
        mujoco.mj_fwdPosition(model, data)
        lengths[k] = data.actuator_length
    lr = np.asarray(model.actuator_lengthrange)
    span = np.where(lr[:, 1] > lr[:, 0], lr[:, 1] - lr[:, 0], np.nan)
    return {"length": lengths, "normalised": (lengths - lr[:, 0]) / span}


def length_report(model: Any, data: Any, poses: list[MappedPose]) -> dict[str, Any]:
    """Finite/positive check and the muscles outside their joint-limit length range."""
    import mujoco

    result = muscle_lengths(model, data, poses)
    length, norm = result["length"], result["normalised"]
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    outside = (norm < -LENGTH_OUT_TOL) | (norm > 1.0 + LENGTH_OUT_TOL)
    frac = outside.mean(axis=0)
    worst = np.argsort(frac)[::-1]
    return {
        "frames": int(length.shape[0]),
        "all_finite": bool(np.isfinite(length).all()),
        "all_positive": bool((length > 0.0).all()),
        "min_length_m": float(length.min()),
        "muscles_ever_outside_range": int((frac > 0).sum()),
        "muscles_outside_over_10pct_of_frames": int((frac > 0.1).sum()),
        "worst": {names[i]: float(frac[i]) for i in worst[:25] if frac[i] > 0},
        "range_definition": (
            "normalised = (length - lengthrange[0]) / span; lengthrange is MuJoCo's "
            "length over the model's own joint limits, so 'outside' marks paths "
            "evaluated beyond MyoFullBody's validated range of motion"
        ),
    }
