"""Sampled total-applied joint work audit for optimization epic #15493.

Power is tau_applied * qd_rollout in a common joint order and SI units.
The audit integrates a piecewise-linear *power* reconstruction, splitting
zero crossings before accumulating positive and negative work per joint.
It does not certify between-sample dynamics, physiological safety or replay.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.optimization.solution_bundle import load_solution_bundle


@dataclass(frozen=True)
class JointWorkAudit:
    """Per-joint signed/positive/negative work in joules; negative work <= 0."""

    signed_work_j: NDArray[np.float64]
    positive_work_j: NDArray[np.float64]
    negative_work_j: NDArray[np.float64]

    @property
    def total_positive_work_j(self) -> float:
        """Sum positive work across joints without inter-joint cancellation."""
        return float(np.sum(self.positive_work_j))


def audit_joint_work(
    time_s: ArrayLike, tau_applied_nm: ArrayLike, velocity_rad_s: ArrayLike
) -> JointWorkAudit:
    """Return signed and sign-separated work for a finite sampled history.

    Preconditions: >=2 strictly increasing times, finite torque/velocity arrays
    of identical shape (samples, joints), >=1 joint, matching frames/order.
    Postcondition: signed work equals positive work plus negative work under
    piecewise-linear sampled power, including exact linear zero crossings.
    Total applied torque must include feedback and saturation effects.
    """
    time = np.asarray(time_s, dtype=float)
    torque = np.asarray(tau_applied_nm, dtype=float)
    velocity = np.asarray(velocity_rad_s, dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise ValueError("time must contain >=2 finite samples")
    if (
        torque.ndim != 2
        or torque.shape != velocity.shape
        or torque.shape[0] != time.size
        or torque.shape[1] < 1
        or not np.all(np.isfinite(torque))
        or not np.all(np.isfinite(velocity))
    ):
        raise ValueError("torque and velocity must be finite (samples, joints) arrays")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dt = np.diff(time)[:, None]
        power = torque * velocity
        if np.any(dt <= 0) or not np.all(np.isfinite(dt)):
            raise ValueError("time must strictly increase with finite intervals")
        if not np.all(np.isfinite(power)):
            raise ValueError("derived power must be finite")
        left, right = power[:-1], power[1:]
        positive_left, positive_right = np.maximum(left, 0), np.maximum(right, 0)
        area = (positive_left / 2 + positive_right / 2) * dt
        crossing = ((left < 0) & (right > 0)) | ((left > 0) & (right < 0))
        # Fraction of each crossing interval on the positive side. Scaling
        # avoids overflow when adding large opposite-sign endpoint magnitudes.
        scale = np.maximum(np.abs(left), np.abs(right))
        scale = np.where(scale == 0, 1, scale)
        fraction = (positive_left / scale + positive_right / scale) / (
            np.abs(left) / scale + np.abs(right) / scale
        )
        area = np.where(crossing, area * fraction, area)
        positive = np.sum(area, axis=0)
        signed = np.sum((left / 2 + right / 2) * dt, axis=0)
        negative = signed - positive
        total = np.sum(positive)
    if not all(np.all(np.isfinite(v)) for v in (positive, signed, negative, total)):
        raise ValueError("integrated work must be finite")
    return JointWorkAudit(signed, positive, negative)


def main() -> None:
    """Load an integrity-checked Solution Bundle and emit a strict JSON audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = load_solution_bundle(args.bundle)
    arrays = bundle.arrays
    audit = audit_joint_work(
        arrays["time"], arrays["tau_applied"], arrays["qd_rollout"]
    )
    print(
        json.dumps(
            {
                "solution_id": bundle.solution_id,
                "joint_order": bundle.joint_order,
                "units": "J",
                "power_reconstruction": "piecewise_linear",
                "signed_work_j": audit.signed_work_j.tolist(),
                "positive_work_j": audit.positive_work_j.tolist(),
                "negative_work_j": audit.negative_work_j.tolist(),
                "total_positive_work_j": audit.total_positive_work_j,
                "qualification": "sampled_work_only",
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
