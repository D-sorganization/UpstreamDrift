"""Round-trip parity for ``save_initial_state`` / ``load_initial_state``.

For each of the five supported engines, save then load must reproduce
the original :class:`CanonicalPose` to ``1e-9`` tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.canonical import (
    CanonicalPose,
    canonical_from_reference_setup,
)
from src.shared.python.pose_interchange.pose_io import (
    SUPPORTED_ENGINES,
    load_initial_state,
    save_initial_state,
)

pytestmark = pytest.mark.unit


def _make_richly_populated_pose() -> CanonicalPose:
    rng = np.random.default_rng(seed=20260509)
    angles = {name: float(rng.uniform(-45.0, 45.0)) for name in REFERENCE_GOLFER_FIELDS}
    return CanonicalPose(
        pelvis_translation_m=np.array([0.1, -0.2, 0.95], dtype=float),
        pelvis_rotation_xyz_deg=np.array([3.5, -7.25, 11.0], dtype=float),
        joint_angles_deg=angles,
    )


def _assert_pose_close(a: CanonicalPose, b: CanonicalPose, *, tol: float) -> None:
    np.testing.assert_allclose(
        a.pelvis_translation_m, b.pelvis_translation_m, atol=tol, rtol=0
    )
    np.testing.assert_allclose(
        a.pelvis_rotation_xyz_deg, b.pelvis_rotation_xyz_deg, atol=tol, rtol=0
    )
    full_a = a.angles_full_dict_deg()
    full_b = b.angles_full_dict_deg()
    assert set(full_a) == set(full_b)
    for key in full_a:
        assert full_a[key] == pytest.approx(full_b[key], abs=tol)


@pytest.mark.parametrize("engine", sorted(SUPPORTED_ENGINES))
def test_engine_roundtrip_reference_pose(engine: str, tmp_path: Path) -> None:
    pose = canonical_from_reference_setup()
    out = tmp_path / f"initial_state_{engine}"
    save_initial_state(pose, engine, out)
    # pinocchio appends .npz; everything else writes the path verbatim.
    candidate = out if out.exists() else out.with_suffix(".npz")
    recovered = load_initial_state(engine, candidate)
    _assert_pose_close(pose, recovered, tol=1e-9)


@pytest.mark.parametrize("engine", sorted(SUPPORTED_ENGINES))
def test_engine_roundtrip_random_pose(engine: str, tmp_path: Path) -> None:
    pose = _make_richly_populated_pose()
    out = tmp_path / f"random_state_{engine}"
    save_initial_state(pose, engine, out)
    candidate = out if out.exists() else out.with_suffix(".npz")
    recovered = load_initial_state(engine, candidate)
    _assert_pose_close(pose, recovered, tol=1e-9)
