"""TDD tests for frozen MuJoCo canonical-v2 / capture-A fixture (#15793 A1)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import src.engines.physics_engines.opensim.python.motion_matching.canonical_v2_reference_fixture as mod
from src.shared.python.model_appearance.canonical_golfer_gate import (
    NonCanonicalGolferError,
)
from src.shared.python.motion_matching.execution.golfer_reference_specs import (
    reference_spec,
)
from src.shared.python.swing_comparison.motion import CAPTURE_A_MARKER_LABELS

pytestmark = pytest.mark.unit
_SHA = "cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d"


def _tx() -> mod.FrameTransform3D:
    """Return a valid proper-orthogonal MuJoCo Z-up to OpenSim Y-up transform."""
    rot = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    return mod.FrameTransform3D("mujoco_world_Zup", "opensim_ground_Yup", rot)


def _traj(n: int = 5) -> mod.CanonicalV2ReferenceTrajectory:
    """Build a deterministic valid reference trajectory on the capture-A rate."""
    dt, f = 1.0 / 360.0, np.float64
    q = np.linspace(-0.2, 0.4, n * 2, dtype=f).reshape(n, 2)
    qd = np.linspace(0.1, 0.7, n * 2, dtype=f).reshape(n, 2)
    pos = np.linspace(-1.0, 1.0, n * 9, dtype=f).reshape(n, 3, 3)
    t, lbls = np.arange(n, dtype=f) * dt, CAPTURE_A_MARKER_LABELS[:3]
    return mod.CanonicalV2ReferenceTrajectory(
        t, dt, ("pelvis_tx", "pelvis_ty"), q, qd, lbls, pos
    )


def test_non_canonical_golfer_spec_rejected_fail_closed() -> None:
    """Non-canonical golfer specs must fail closed via require_canonical_golfer."""
    cases: tuple[tuple[str, str, Any, str], ...] = (
        ("subject", "hand_visual_backend", "capsule", "capsule"),
        ("subject", "body_visual_backend", "loft", "loft"),
        ("", "spec_revision", {"configuration": "old"}, "spec_revision"),
        ("club", "shaft", {"model": "rigid"}, "shaft"),
    )
    for section, key, val, part in cases:
        spec = copy.deepcopy(reference_spec("driver"))
        (spec[section] if section else spec)[key] = val
        with pytest.raises(NonCanonicalGolferError) as exc:
            mod.build_canonical_v2_reference_fixture(spec, _traj(), _tx(), _SHA)
        assert exc.value.part == part


def test_non_orthogonal_or_improper_frame_transform_rejected() -> None:
    """FrameTransform3D rejects det != +1, R.T @ R != I, and non-finite inputs."""
    bad_rots: tuple[mod.RotationTuple, ...] = (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        ((1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 0.25)),
        ((1.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.0, 1.0)),
    )
    for bad_rot in bad_rots:
        with pytest.raises(ValueError):
            mod.FrameTransform3D("mujoco_world_Zup", "opensim_ground_Yup", bad_rot)


def test_invalid_timebase_rejected_fail_closed() -> None:
    """Non-monotone, non-finite, or dt_s <= 0 timebases must fail closed."""
    bad_cases: tuple[tuple[list[float], float], ...] = (
        ([0.0, 0.02, 0.01, 0.03], 0.01),
        ([0.0, 0.01, 0.01, 0.02], 0.01),
        ([0.0, float("nan"), 0.02], 0.01),
        ([0.0, 0.01, float("inf")], 0.01),
        ([0.0, 0.01, 0.02], 0.0),
        ([0.0, 0.01, 0.02], -0.01),
        ([0.0, 0.01, 0.02], float("nan")),
        ([0.0, 0.01, 0.02], 0.02),
    )
    for times, dt_s in bad_cases:
        n = len(times)
        z1, z3 = (
            np.zeros((n, 1), dtype=np.float64),
            np.zeros((n, 1, 3), dtype=np.float64),
        )
        t = np.asarray(times, dtype=np.float64)
        with pytest.raises(ValueError):
            mod.CanonicalV2ReferenceTrajectory(
                t, dt_s, ("pelvis_tx",), z1, z1, ("WaistLeft",), z3
            )


def test_deterministic_json_roundtrip_and_sha256_verification() -> None:
    """Fixture serializes deterministically and passes SHA-256 verification."""
    spec = copy.deepcopy(reference_spec("driver"))
    fix = mod.build_canonical_v2_reference_fixture(
        spec, _traj(), _tx(), _SHA, metadata={"source": "capture-A"}
    )
    assert fix.schema_version == mod.SCHEMA_VERSION and fix.capture_id == "capture-A"
    assert len(fix.fixture_sha256) == 64
    mod.verify_fixture_integrity(fix)
    raw = fix.to_json()
    assert raw == fix.to_json()
    loaded = mod.CanonicalV2ReferenceFixture.from_json(raw)
    assert loaded == fix and loaded.to_json() == raw
    np.testing.assert_allclose(loaded.trajectory.q, fix.trajectory.q)
    np.testing.assert_allclose(
        loaded.trajectory.marker_positions_m, fix.trajectory.marker_positions_m
    )


def test_tampering_trajectory_or_markers_raises_integrity_error_on_load() -> None:
    """Tampering trajectory, marker arrays, or digest raises FixtureIntegrityError."""
    fix = mod.build_canonical_v2_reference_fixture(
        copy.deepcopy(reference_spec("driver")), _traj(), _tx(), _SHA
    )
    payload = json.loads(fix.to_json())
    muts = (
        lambda d: d["trajectory"]["q"][0].__setitem__(0, 99.0),
        lambda d: d["trajectory"]["marker_positions_m"][0][0].__setitem__(2, 9.9),
        lambda d: d["trajectory"]["marker_labels"].__setitem__(0, "TamperedMarker"),
    )
    for mut in muts:
        tampered = copy.deepcopy(payload)
        mut(tampered)
        with pytest.raises(mod.FixtureIntegrityError):
            mod.CanonicalV2ReferenceFixture.from_json(json.dumps(tampered))
    with pytest.raises(mod.FixtureIntegrityError):
        mod.verify_fixture_integrity(replace(fix, fixture_sha256="f" * 64))
