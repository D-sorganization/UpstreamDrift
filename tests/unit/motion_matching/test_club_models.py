"""Tests for the typical-club parameters and their application to a document."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching import club_models as module

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def _wrist_to_head(doc: dict) -> tuple[float, float]:
    adapter = NativeMujocoFullBodyModel(json.dumps(doc).encode())
    poses = adapter.frame_poses(dict.fromkeys(doc["coordinate_order"], 0.0))
    head, lw, rw = poses["Clubhead"][:3, 3], poses["LW"][:3, 3], poses["RW"][:3, 3]
    return float(np.linalg.norm(head - lw)), float(np.linalg.norm(lw - rw))


def test_typical_clubs_have_typical_masses_and_lengths() -> None:
    assert 0.29 < module.DRIVER.total_mass_kg < 0.34
    assert 0.40 < module.IRON_7.total_mass_kg < 0.46
    assert module.DRIVER.length_m > module.IRON_7.length_m
    with pytest.raises(ValueError):
        module.ClubSpec(
            "x", 1.0, 0.2, 0.1, 0.05, 1.2, 0.006, "box", (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises(ValueError):
        module.ClubSpec(
            "x", 1.0, 0.2, 0.1, 0.05, 0.2, 0.006, "cone", (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.parametrize("club", [module.DRIVER, module.IRON_7])
def test_apply_club_sets_masses_and_moves_the_hands_along_the_shaft(
    club: module.ClubSpec,
) -> None:
    doc = json.loads(SPEC.read_text())
    before_head, before_hands = _wrist_to_head(doc)
    club_body = next(b for b in doc["bodies"] if b["name"].endswith("Clubface Vector"))
    before_wrist = -next(j for j in doc["joints"] if j["child"] == club_body["name"])[
        "child_to_follower"
    ][1][3]
    out = module.apply_club(doc, club)
    body = next(b for b in out["bodies"] if b["name"].endswith("Clubface Vector"))
    club_mass = sum(
        s["mass_kg"] for s in body["solids"] if "Hand" not in s["name"].rsplit("/")[-1]
    )
    assert club_mass == pytest.approx(club.total_mass_kg)
    for solid in body["solids"]:
        if solid["mass_kg"] > 0:
            assert np.all(np.linalg.eigvalsh(np.array(solid["inertia_com_kg_m2"])) > 0)
    after_head, after_hands = _wrist_to_head(out)
    # The LW frame sits a little off the shaft axis, so compare the change.
    assert after_head - before_head == pytest.approx(
        club.wrist_to_head_m - before_wrist, abs=2e-3
    )
    assert after_hands == pytest.approx(before_hands, abs=1e-9)  # hand relation kept
    assert out["club"]["name"] == club.name
    assert out["visual_hints"]["shapes"][body["name"] + "/Clubhead"]["shape"] == (
        club.head_shape
    )
    assert json.loads(SPEC.read_text()) == doc  # input untouched


def test_specs_from_the_club_database_agree_with_the_typical_constants() -> None:
    driver = module.from_database("driver")
    iron = module.from_database("7_iron")
    assert driver.length_m == pytest.approx(module.DRIVER.length_m, abs=1e-3)
    assert driver.head_mass_kg == pytest.approx(module.DRIVER.head_mass_kg, abs=0.01)
    assert driver.head_shape == "ellipsoid" and iron.head_shape == "box"
    assert iron.length_m == pytest.approx(module.IRON_7.length_m, abs=1e-3)
    assert 0.03 < driver.head_gyration_m[2] < 0.06
    assert abs(driver.total_mass_kg - module.DRIVER.total_mass_kg) < 0.02
    with pytest.raises(ValueError):
        module.from_database("no_such_club")
