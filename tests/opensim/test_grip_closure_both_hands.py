"""Both hands are constrained in every OpenSim golf model (OSV-2, #11728)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.grip_closure import (
    constrained_hands,
    weld_closure_series,
)
from src.engines.physics_engines.opensim.python.tour_matching.address import (
    FROZEN_ADDRESS_TOLERANCE_PROFILE,
)
from src.shared.python.grip_contact import load_coordinate_swing
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "src/engines/physics_engines/opensim/models"
BUILDER = ROOT / "scripts/build_humanoid_osim.py"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPECS = ROOT / "docs/development/full_body_models"
GENERATED = {
    "driver": MODELS / "generated/full_body_anthro_driver.osim",
    "iron7": MODELS / "generated/full_body_anthro_iron7.osim",
}


def test_golf_humanoid_constrains_both_hands() -> None:
    hands = constrained_hands(MODELS / "golf_humanoid.osim")
    assert set(hands) == {"lead", "trail"}
    assert hands["lead"].startswith("WeldConstraint")  # DEFAULT_CLOSURE_TYPE == weld


@pytest.mark.parametrize("club", sorted(GENERATED))
def test_generated_full_body_models_constrain_both_hands(club: str) -> None:
    hands = constrained_hands(GENERATED[club])
    assert set(hands) == {"lead", "trail"}


def test_a_model_with_only_the_trail_hand_is_reported_as_such(tmp_path: Path) -> None:
    text = (MODELS / "golf_humanoid.osim").read_text()
    start = text.index('<WeldConstraint name="hand_l_to_club">')
    end = text.index("</WeldConstraint>") + len("</WeldConstraint>")
    stripped = tmp_path / "one_hand.osim"
    stripped.write_text(text[:start] + text[end:])
    assert set(constrained_hands(stripped)) == {"trail"}


def _builder():
    spec = importlib.util.spec_from_file_location("build_humanoid_osim", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_places_the_lead_weld_at_the_shared_offset() -> None:
    (weld,) = _builder()._make_lead_hand_closure("weld")
    club_frame = next(
        f
        for f in weld.iter("PhysicalOffsetFrame")
        if f.get("name") == "club_lead_grip_offset"
    )
    y = float(club_frame.findtext("translation", "").split()[1])
    assert y == pytest.approx(DEFAULT_GRIP_POSE.club_y_m(Hand.LEAD))


def test_builder_point_pair_option_and_rejection() -> None:
    builder = _builder()
    pair = builder._make_lead_hand_closure("point_pair")
    assert [e.tag for e in pair] == ["PointConstraint", "PointConstraint"]
    with pytest.raises(ValueError, match="closure_type"):
        builder._make_lead_hand_closure("bushing")


def test_right_weld_puts_the_trail_hand_at_the_shared_trail_offset() -> None:
    joint = _builder()._make_club_weld_joint()
    frame = next(
        f
        for f in joint.iter("PhysicalOffsetFrame")
        if f.get("name") == "club_grip_offset"
    )
    y = float(frame.findtext("translation", "").split()[1])
    assert y == pytest.approx(DEFAULT_GRIP_POSE.club_y_m(Hand.TRAIL))


def test_committed_model_matches_the_builder(tmp_path: Path) -> None:
    builder = _builder()
    if not Path(builder.BASE_OSIM).is_file():
        pytest.skip("opensim-models submodule not checked out")
    out = tmp_path / "golf_humanoid.osim"
    builder.build(output_path=out)
    assert out.read_bytes() == (MODELS / "golf_humanoid.osim").read_bytes()


def test_golf_humanoid_loads_with_the_lead_constraint() -> None:
    osim = pytest.importorskip("opensim")
    model = osim.Model(str(MODELS / "golf_humanoid.osim"))
    model.initSystem()
    constraints = model.getConstraintSet()
    assert constraints.get("hand_l_to_club").get_isEnforced()


@pytest.mark.parametrize("club", sorted(GENERATED))
def test_generated_weld_closure_stays_within_tolerance_on_the_canned_swing(
    club: str,
) -> None:
    pytest.importorskip("opensim")
    spec = json.loads((SPECS / f"full_body_spec_anthro_{club}.json").read_bytes())
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz",
        FIXTURES / "address_poses.json",
        club,
        spec["coordinate_order"],
    )
    series = weld_closure_series(GENERATED[club], swing.names, swing.q)
    assert series.available, series.reason
    assert series.residual_m is not None and series.residual_m.size == swing.q.shape[0]
    assert series.within(FROZEN_ADDRESS_TOLERANCE_PROFILE.max_grip_closure_m) is True
    assert np.isfinite(series.max_m)
