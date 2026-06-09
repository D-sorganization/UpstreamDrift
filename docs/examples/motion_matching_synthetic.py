#!/usr/bin/env python3
"""End-to-end example: motion-matching cost on a synthetic trajectory.

Motion matching scores how closely a *candidate* joint trajectory tracks a
*reference* (e.g. a mocap-derived target). The production pipeline lives in
``src/engines/physics_engines/drake/python/motion_matching/`` and depends on
Drake; this example reproduces the core scoring idea with ``numpy`` only so it
runs anywhere.

Run it directly from the repository root::

    python docs/examples/motion_matching_synthetic.py

The cost combines a per-frame pose error with a velocity-profile error, which
is the same decomposition the real ``compute_cost`` adapters use.
"""

from __future__ import annotations

import numpy as np


def make_reference(num_frames: int = 120, num_joints: int = 7) -> np.ndarray:
    """Build a smooth synthetic reference swing trajectory.

    Args:
        num_frames: Number of time samples.
        num_joints: Number of joints.

    Returns:
        Array of shape ``(num_frames, num_joints)``.
    """
    t = np.linspace(0.0, 1.0, num_frames)
    phases = np.linspace(0.0, np.pi / 2, num_joints)
    # Each joint follows a phase-shifted half-sine "downswing".
    return np.sin(np.pi * t[:, None] + phases[None, :])


def motion_match_cost(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    velocity_weight: float = 0.5,
    dt: float = 1.0 / 120.0,
) -> float:
    """Return a scalar tracking cost between two joint trajectories.

    Args:
        reference: Target trajectory, shape ``(frames, joints)``.
        candidate: Trajectory to score, same shape as ``reference``.
        velocity_weight: Relative weight of the velocity-profile term.
        dt: Sample spacing used for the finite-difference velocities.

    Returns:
        Non-negative scalar cost (``0.0`` for a perfect match).
    """
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have matching shapes")
    if dt <= 0:
        raise ValueError("dt must be positive")

    pose_error = float(np.mean((reference - candidate) ** 2))
    ref_vel = np.gradient(reference, dt, axis=0)
    cand_vel = np.gradient(candidate, dt, axis=0)
    vel_error = float(np.mean((ref_vel - cand_vel) ** 2))
    return pose_error + velocity_weight * vel_error


def main() -> int:
    """Score a noisy candidate against the reference and print the cost."""
    rng = np.random.default_rng(seed=0)
    reference = make_reference()

    perfect = motion_match_cost(reference, reference)
    noisy = motion_match_cost(
        reference, reference + rng.normal(0, 0.05, reference.shape)
    )

    print(f"Cost for a perfect match: {perfect:.6f}")
    print(f"Cost for a noisy candidate: {noisy:.6f}")
    assert perfect == 0.0, "a trajectory must match itself exactly"
    assert noisy > perfect, "a noisier candidate must cost more"
    print("Motion-matching example completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
