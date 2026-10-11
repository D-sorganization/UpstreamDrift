"""Grip interface description: frames, bushing and contact parameters (#11739)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    BushingParameters,
    ContactMaterial,
    GripFrame,
    GripInterface,
    default_bushing,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
EYE = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def test_frames_come_from_spec_closure_and_wrist_joint(spec: dict) -> None:
    gi = GripInterface.from_spec(spec)
    closure = np.asarray(spec["closure"]["placement_b"])
    np.testing.assert_allclose(gi.right.position_m, closure[:3, 3])
    np.testing.assert_allclose(gi.right.matrix()[:3, :3], closure[:3, :3])
    wrist = next(j for j in spec["joints"] if j["child"] == spec["closure"]["body_b"])
    np.testing.assert_allclose(
        gi.left.position_m, np.asarray(wrist["child_to_follower"])[:3, 3]
    )
    # x axis of the grip frame is the shaft axis (club y) for both hands
    np.testing.assert_allclose(gi.left.matrix()[:3, 0], (0.0, 1.0, 0.0), atol=1e-12)
    assert 0.05 < gi.hand_separation_m < 0.12


def test_defaults_are_positive_and_scalable() -> None:
    bp = default_bushing()
    assert all(k > 0 for k in bp.translational_stiffness_n_m)
    soft = bp.scaled(0.01)
    assert soft.translational_stiffness_n_m[0] == pytest.approx(
        0.01 * bp.translational_stiffness_n_m[0]
    )

    def ratio(p: BushingParameters) -> float:
        return p.translational_damping_ns_m[0] / math.sqrt(
            p.translational_stiffness_n_m[0]
        )

    assert ratio(soft) == pytest.approx(ratio(bp))
    assert "engineering default" in bp.source


@pytest.mark.parametrize("bad", [(0, 1, 1), (1, 1), (1, math.nan, 1), (-1, 1, 1)])
def test_bushing_rejects_invalid_stiffness(bad: tuple) -> None:
    ok = (1.0, 1.0, 1.0)
    with pytest.raises((ValueError, TypeError)):
        BushingParameters(bad, ok, ok, ok)


def test_bushing_rejects_negative_damping_and_bad_scale() -> None:
    ok = (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        BushingParameters(ok, ok, (-1.0, 0.0, 0.0), ok)
    with pytest.raises(ValueError):
        default_bushing().scaled(0.0)


def test_frame_validation() -> None:
    with pytest.raises(ValueError):
        GripFrame("X", (0, 0, 0), EYE)
    with pytest.raises(ValueError):
        GripFrame("L", (0, 0, math.inf), EYE)
    with pytest.raises(ValueError):
        GripFrame("L", (0, 0, 0), ((2, 0, 0), (0, 1, 0), (0, 0, 1)))
    with pytest.raises(ValueError):  # reflection
        GripFrame("L", (0, 0, 0), ((-1, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_interface_validation(spec: dict) -> None:
    a = GripFrame("L", (0, 0, 0), EYE)
    with pytest.raises(ValueError):
        GripInterface(a, GripFrame("R", (0, 0, 0), EYE))
    with pytest.raises(ValueError):
        GripInterface(GripFrame("R", (0, 0, 0), EYE), a)
    with pytest.raises(ValueError):
        GripInterface.from_spec({**spec, "closure": None})
    with pytest.raises(ValueError):
        GripInterface.from_spec(spec).frame("Q")


def test_contact_material_placeholder_validation() -> None:
    assert ContactMaterial().status.startswith("placeholder")
    with pytest.raises(ValueError):
        ContactMaterial(dynamic_friction=2.0)
    with pytest.raises(ValueError):
        ContactMaterial(stiffness_n_m2=0.0)
