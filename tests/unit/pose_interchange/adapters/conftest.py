"""Shared fixtures for adapter parity tests.

Each adapter ships its own ``test_<engine>_*.py`` files. The fixtures
here give them a uniform way to draw random :class:`CanonicalPose`
samples from a seeded RNG so the round-trip parity tests are
deterministic.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.canonical import CanonicalPose


def random_canonical_pose(rng: np.random.Generator) -> CanonicalPose:
    """Draw a single random canonical pose from *rng*.

    Pelvis rotation Y is clamped to ``[-80, 80]`` deg to stay outside
    the gimbal-lock singularity at ``y = +-90`` deg; that matches the
    canonical golfer envelope.
    """
    translation = rng.uniform(low=-1.0, high=1.0, size=3)
    rot_x = float(rng.uniform(-180.0, 180.0))
    rot_y = float(rng.uniform(-80.0, 80.0))
    rot_z = float(rng.uniform(-180.0, 180.0))
    angles = {name: float(rng.uniform(-90.0, 90.0)) for name in REFERENCE_GOLFER_FIELDS}
    return CanonicalPose(
        pelvis_translation_m=translation,
        pelvis_rotation_xyz_deg=np.array([rot_x, rot_y, rot_z], dtype=float),
        joint_angles_deg=angles,
    )


@pytest.fixture()
def rng() -> Iterator[np.random.Generator]:
    """Seeded RNG so tests are reproducible."""
    yield np.random.default_rng(seed=20260509)


@pytest.fixture()
def random_poses(rng: np.random.Generator) -> list[CanonicalPose]:
    """100 random canonical poses, one fresh draw per test invocation."""
    return [random_canonical_pose(rng) for _ in range(100)]
