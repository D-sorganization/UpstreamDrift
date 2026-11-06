"""Refit capture club pose per trail-grip option (SHOW-A20, #14796).

In the G2 swing comparison (#14751), tracking residuals for trail-hand options
(a), (b15), and (c) were measured by solving arm postures against a captured
club pose fitted through the copy-grip model (both hands sharing one rotation).
That biased the reference toward the copy grip: any non-copy trail rotation was
forced to track a club pose built for identical rotations, artificially
inflating residuals (copy 84.7 mm, (a) +55 mm, (b15) +131 mm, (c) +147 mm).

This module implements joint refitting of the capture club pose and bilateral
arm postures per grip option. By allowing the club pose to adjust to the
bilateral grip midpoint and orientation while holding hand-club rotation fixed
per option, the copy-grip reference bias is eliminated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from src.shared.python.motion_matching.execution.grip_rotation import (
    apply_grip_rotation,
)
from docs.development.full_body_models.evidence.grip_roll.solve_neutral_grip import (
    Rig,
)

ROOT = Path(__file__).resolve().parents[4]
ARM_KEYWORDS = ("SInput", "EInput", "FInput", "WInput", "Scap")

#: Canonical trail-hand Euler angles (extrinsic x-y-z, deg) for G2 comparison
OPTION_TRAIL_EULER_DEG: dict[str, tuple[float, float, float]] = {
    "copy": (-75.3894, -9.7161, 4.5795),
    "(a)": (85.5, 43.2, -154.0),
    "(b15)": (75.8, 13.6, -173.5),
    "(c)": (85.5, -43.2, 154.0),
}


@dataclass(frozen=True, slots=True)
class RefitSummary:
    """Summary of joint club-pose refit for one grip option."""

    grip_option: str
    trail_euler_deg: tuple[float, float, float]
    closure_pos_max_mm: float
    closure_pos_p95_mm: float
    closure_rot_max_deg: float
    closure_rot_p95_deg: float
    club_shift_pos_max_mm: float
    club_shift_pos_p95_mm: float
    club_shift_rot_max_deg: float
    club_shift_rot_p95_deg: float
    frames_at_limit: int
    binding_joints: dict[str, int]
    runtime_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "grip_option": self.grip_option,
            "trail_euler_deg": list(self.trail_euler_deg),
            "closure_pos_max_mm": self.closure_pos_max_mm,
            "closure_pos_p95_mm": self.closure_pos_p95_mm,
            "closure_rot_max_deg": self.closure_rot_max_deg,
            "closure_rot_p95_deg": self.closure_rot_p95_deg,
            "club_shift_pos_max_mm": self.club_shift_pos_max_mm,
            "club_shift_pos_p95_mm": self.club_shift_pos_p95_mm,
            "club_shift_rot_max_deg": self.club_shift_rot_max_deg,
            "club_shift_rot_p95_deg": self.club_shift_rot_p95_deg,
            "frames_at_limit": self.frames_at_limit,
            "binding_joints": self.binding_joints,
            "runtime_s": self.runtime_s,
        }


def get_arm_joint_indices_and_bounds(
    rig_order: Sequence[str],
    coordinate_ranges_deg: Mapping[str, Sequence[float]],
) -> tuple[list[int], list[str], NDArray[np.float64], NDArray[np.float64]]:
    """Extract arm joint indices in rig order and corresponding lower/upper bounds in radians."""
    idx = [
        i
        for i, n in enumerate(rig_order)
        if n[0] in "LR" and any(k in n for k in ARM_KEYWORDS)
    ]
    idx_names = [rig_order[i] for i in idx]
    lb = np.array(
        [
            np.radians(coordinate_ranges_deg[n][0])
            if n in coordinate_ranges_deg
            else -np.inf
            for n in idx_names
        ],
        dtype=np.float64,
    )
    ub = np.array(
        [
            np.radians(coordinate_ranges_deg[n][1])
            if n in coordinate_ranges_deg
            else np.inf
            for n in idx_names
        ],
        dtype=np.float64,
    )
    return idx, idx_names, lb, ub


def refit_joint_club_pose(
    spec: Mapping[str, Any],
    grip_option: str,
    q_swing: NDArray[np.float64],
    *,
    trail_euler_deg: Sequence[float] | None = None,
    max_frames: int | None = None,
    xtol: float = 1e-4,
    ftol: float = 1e-4,
    max_nfev: int = 50,
) -> RefitSummary:
    """Refit club pose and bilateral arm postures jointly for ``grip_option``."""
    t0 = time.time()
    if trail_euler_deg is None:
        if grip_option not in OPTION_TRAIL_EULER_DEG:
            raise ValueError(
                f"Unknown grip_option {grip_option!r}, must be one of {sorted(OPTION_TRAIL_EULER_DEG)}"
            )
        target_euler = OPTION_TRAIL_EULER_DEG[grip_option]
    else:
        target_euler = tuple(float(v) for v in trail_euler_deg)  # type: ignore[assignment]

    old_rig = Rig(json.dumps(spec).encode())
    new_spec, _ = apply_grip_rotation(spec, {"R": target_euler})
    new_rig = Rig(json.dumps(new_spec, sort_keys=True).encode())

    ranges = spec.get("coordinate_ranges_deg", {})
    idx, idx_names, lb, ub = get_arm_joint_indices_and_bounds(old_rig.order, ranges)

    frames = len(q_swing) if max_frames is None else min(len(q_swing), max_frames)
    closure_pos = np.zeros(frames, dtype=np.float64)
    closure_rot = np.zeros(frames, dtype=np.float64)
    club_shift_pos = np.zeros(frames, dtype=np.float64)
    club_shift_rot = np.zeros(frames, dtype=np.float64)

    joint_violations = {n: 0 for n in idx_names if n in ranges}
    frames_at_limit = 0

    guess = np.clip(q_swing[0, idx], lb, ub)

    for t in range(frames):
        (rp, rR), _, _ = old_rig.sites(q_swing[t])

        def resid_joint(x: NDArray[np.float64], t_idx: int = t) -> NDArray[np.float64]:
            row = q_swing[t_idx].copy()
            row[idx] = x
            (p, R), a, b = new_rig.sites(row)
            return np.concatenate(
                [
                    (a[0] - b[0]) * 100.0,
                    Rot.from_matrix(a[1].T @ b[1]).as_rotvec() * 17.2,
                    1.0 * (x - q_swing[t_idx, idx]) * 57.3,
                ]
            )

        sol = least_squares(
            resid_joint,
            guess,
            bounds=(lb, ub),
            xtol=xtol,
            ftol=ftol,
            max_nfev=max_nfev,
        )
        guess = sol.x

        row = q_swing[t].copy()
        row[idx] = sol.x
        (p, R), a, b = new_rig.sites(row)

        closure_pos[t] = np.linalg.norm(a[0] - b[0]) * 1000.0
        closure_rot[t] = np.degrees(
            np.linalg.norm(Rot.from_matrix(a[1].T @ b[1]).as_rotvec())
        )
        club_shift_pos[t] = np.linalg.norm(p - rp) * 1000.0
        club_shift_rot[t] = np.degrees(
            np.linalg.norm(Rot.from_matrix(rR.T @ R).as_rotvec())
        )

        deg = np.degrees(sol.x)
        hit_any = False
        for i, n in enumerate(idx_names):
            if n in ranges:
                lo, hi = ranges[n]
                if deg[i] <= lo + 0.1 or deg[i] >= hi - 0.1:
                    joint_violations[n] += 1
                    hit_any = True
        if hit_any:
            frames_at_limit += 1

    t1 = time.time()
    return RefitSummary(
        grip_option=grip_option,
        trail_euler_deg=target_euler,
        closure_pos_max_mm=round(float(np.max(closure_pos)), 1),
        closure_pos_p95_mm=round(float(np.percentile(closure_pos, 95)), 1),
        closure_rot_max_deg=round(float(np.max(closure_rot)), 1),
        closure_rot_p95_deg=round(float(np.percentile(closure_rot, 95)), 1),
        club_shift_pos_max_mm=round(float(np.max(club_shift_pos)), 1),
        club_shift_pos_p95_mm=round(float(np.percentile(club_shift_pos, 95)), 1),
        club_shift_rot_max_deg=round(float(np.max(club_shift_rot)), 1),
        club_shift_rot_p95_deg=round(float(np.percentile(club_shift_rot, 95)), 1),
        frames_at_limit=frames_at_limit,
        binding_joints={k: v for k, v in joint_violations.items() if v > 0},
        runtime_s=round(t1 - t0, 2),
    )
