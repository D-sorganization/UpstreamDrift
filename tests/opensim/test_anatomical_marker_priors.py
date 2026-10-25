"""Tests for anatomical marker priors and plausibility checks (#13434)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.anatomical_marker_priors import (
    AnatomicalPrior,
    _ANCHOR_JOINTS,
    _joint_anchors,
    anatomical_marker_priors,
    clamp_to_priors,
    lateral_side_ok,
    marker_side,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_plausibility import (
    assess_marker_placements,
)
from src.shared.python.contracts import PreconditionError
from tests.helpers.opensim_support import require_opensim

ROOT = Path(__file__).resolve().parents[2]
OS7_MODEL_PATH = (
    ROOT
    / "docs/development/opensim_tour_matching/evidence/os7_moco_g1/inputs/golf_humanoid_scaled_tour_markers.osim"
)


# ─── Unit tests (no OpenSim required) ─────────────────────────────────────────


@pytest.mark.unit
def test_marker_side() -> None:
    assert marker_side("WaistLeft") == -1
    assert marker_side("WaistRBack") == 1
    assert marker_side("LKneeOut") == -1
    assert marker_side("RToeIn") == 1
    assert marker_side("HeadTop") == 0
    assert marker_side("BackTop") == 0
    assert marker_side("Marker_2:2:1") == 0


@pytest.mark.unit
def test_lateral_side_ok() -> None:
    # Lateral markers
    assert lateral_side_ok("LKneeOut", (0.0, 0.0, -0.05))
    assert not lateral_side_ok("LKneeOut", (0.0, 0.0, 0.05))
    assert lateral_side_ok("RKneeOut", (0.0, 0.0, 0.05))
    assert not lateral_side_ok("RKneeOut", (0.0, 0.0, -0.05))

    # Midline markers (always True regardless of z)
    assert lateral_side_ok("HeadTop", (0.0, 0.0, 0.10))
    assert lateral_side_ok("HeadTop", (0.0, 0.0, -0.10))
    assert lateral_side_ok("BackTop", (0.0, 0.0, 0.0))

    # Medial markers (ToeIn): allowed within max(tol_m, 0.05) across midline
    assert lateral_side_ok("RToeIn", (0.0, 0.0, 0.02))
    assert lateral_side_ok("RToeIn", (0.0, 0.0, -0.04))  # within 0.05 m tolerance
    assert not lateral_side_ok("RToeIn", (0.0, 0.0, -0.06))  # exceeds tolerance
    assert lateral_side_ok("LToeIn", (0.0, 0.0, -0.02))
    assert lateral_side_ok("LToeIn", (0.0, 0.0, 0.04))  # within 0.05 m tolerance
    assert not lateral_side_ok("LToeIn", (0.0, 0.0, 0.06))  # exceeds tolerance


@pytest.mark.unit
def test_clamp_to_priors() -> None:
    prior = AnatomicalPrior("femur_r", (0.10, 0.20, 0.30), 0.04, "test landmark")

    # Far marker is projected exactly onto the ball surface
    far_offset = ("femur_r", (0.10, 0.20, 0.50))  # 0.20 m away > 0.04 m
    clamped = clamp_to_priors({"RKneeOut": far_offset}, {"RKneeOut": prior})
    clamped_loc = np.asarray(clamped["RKneeOut"][1], dtype=float)
    centre = np.asarray(prior.offset_m, dtype=float)
    dist = float(np.linalg.norm(clamped_loc - centre))
    assert abs(dist - 0.04) < 1e-12

    # Inside marker is left unchanged
    inside_offset = ("femur_r", (0.10, 0.20, 0.31))  # 0.01 m away < 0.04 m
    clamped_inside = clamp_to_priors({"RKneeOut": inside_offset}, {"RKneeOut": prior})
    assert clamped_inside["RKneeOut"] == inside_offset

    # Marker without prior is left unchanged
    club_offset = ("club", (1.0, 2.0, 3.0))
    clamped_club = clamp_to_priors({"ClubMarker": club_offset}, {"RKneeOut": prior})
    assert clamped_club["ClubMarker"] == club_offset

    # Wrong body raises PreconditionError
    wrong_body_offset = ("tibia_r", (0.10, 0.20, 0.30))
    with pytest.raises(PreconditionError):
        clamp_to_priors({"RKneeOut": wrong_body_offset}, {"RKneeOut": prior})


# ─── Live OpenSim tests ───────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.requires_opensim
def test_anatomical_marker_priors_os7_scaled_model() -> None:
    opensim = require_opensim()
    assert OS7_MODEL_PATH.is_file(), f"Model not found: {OS7_MODEL_PATH}"

    priors = anatomical_marker_priors(OS7_MODEL_PATH)
    assert len(priors) == 28

    for name, p in priors.items():
        assert lateral_side_ok(name, p.offset_m), (
            f"{name} failed lateral_side_ok: {p.offset_m}"
        )
        assert 0.0 < p.radius_m <= 0.10, f"{name} radius {p.radius_m} outside (0, 0.1]"

    # Symmetric left/right pairs: AnkleOut, ShoulderTop, WaistLeft/WaistRight mirror to 1e-9
    exact_mirror_pairs = [
        ("LAnkleOut", "RAnkleOut"),
        ("LShoulderTop", "RShoulderTop"),
        ("WaistLeft", "WaistRight"),
    ]
    for left_name, right_name in exact_mirror_pairs:
        z_l = priors[left_name].offset_m[2]
        z_r = priors[right_name].offset_m[2]
        assert abs(z_l - (-z_r)) < 1e-9, (
            f"{left_name}/{right_name} did not mirror to 1e-9: z_L={z_l}, -z_R={-z_r}"
        )

    # KneeOut and ElbowOut: their offsets relative to joint anchor mirror to 1e-9
    model = opensim.Model(str(OS7_MODEL_PATH))
    anchors = _joint_anchors(model, _ANCHOR_JOINTS)

    dz_knee_l = priors["LKneeOut"].offset_m[2] - anchors["walker_knee_l"][2]
    dz_knee_r = priors["RKneeOut"].offset_m[2] - anchors["walker_knee_r"][2]
    assert abs(dz_knee_l - (-dz_knee_r)) < 1e-9

    dz_elbow_l = priors["LElbowOut"].offset_m[2] - anchors["elbow_l"][2]
    dz_elbow_r = priors["RElbowOut"].offset_m[2] - anchors["elbow_r"][2]
    assert abs(dz_elbow_l - (-dz_elbow_r)) < 1e-9

    # Due to subject scaling in OS7, the joint anchors have slight asymmetry
    # (knee ~1e-5 m, elbow ~2.04e-3 m). We verify the parent-frame offsets mirror within tolerance.
    assert (
        abs(priors["LKneeOut"].offset_m[2] - (-priors["RKneeOut"].offset_m[2])) < 2e-5
    )
    assert (
        abs(priors["LElbowOut"].offset_m[2] - (-priors["RElbowOut"].offset_m[2])) < 3e-3
    )


@pytest.mark.integration
@pytest.mark.requires_opensim
def test_old_marker_offsets_fail_plausibility_defect_13434() -> None:
    opensim = require_opensim()
    assert OS7_MODEL_PATH.is_file(), f"Model not found: {OS7_MODEL_PATH}"

    priors = anatomical_marker_priors(OS7_MODEL_PATH)
    model = opensim.Model(str(OS7_MODEL_PATH))
    marker_set = model.getMarkerSet()

    offsets: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for i in range(marker_set.getSize()):
        m = marker_set.get(i)
        name = str(m.getName())
        body = str(m.getParentFrameName()).rsplit("/", 1)[-1]
        loc = m.get_location()
        offsets[name] = (body, (loc.get(0), loc.get(1), loc.get(2)))

    checks = assess_marker_placements(offsets, priors)
    failed_names = {c.name for c in checks if not c.passed}

    # Document defect #13434: old calibrated model has gross registration errors on LKneeOut and LToeOut
    assert "LKneeOut" in failed_names, "LKneeOut expected to fail plausibility"
    assert "LToeOut" in failed_names, "LToeOut expected to fail plausibility"

    lknee = next(c for c in checks if c.name == "LKneeOut")
    assert not lknee.passed
    assert not lknee.side_ok  # Old LKneeOut had z = +0.3759 m on left femur
    assert lknee.deviation_m > lknee.radius_m  # dev = ~0.43 m > 0.04 m

    ltoe = next(c for c in checks if c.name == "LToeOut")
    assert not ltoe.passed
    assert not ltoe.side_ok  # Old LToeOut had z = +0.4753 m on left foot
    assert ltoe.deviation_m > ltoe.radius_m  # dev = ~0.54 m > 0.04 m
