"""Unit tests for OpenSim lifts, showcase renderer, and additional movements.

Covers:
- LIFT-6 (#11746): OpenSim full-body lift composition and kinetics analyzer
- LIFT-9 (#11749): Lifting showcase video planner and headless smoke renderer
- LIFT-10 (#11750): Additional movements (split jerk, push press, overhead press,
  front squat, lunge) with stance and phase specifications
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from scripts import render_lifting_showcase
from src.shared.python.lifting import (
    ADDITIONAL_LIFTS,
    AdditionalMovementSpec,
    OpenSimFullBodyLiftSpec,
    OpenSimLiftKineticsResult,
    OpenSimLiftModelComposition,
    additional_movement_spec,
    compose_opensim_full_body_lift,
    evaluate_opensim_lift_kinetics,
    is_additional_movement,
    lift_grip,
    lift_phase_names,
    lift_phases,
    lift_root_joint,
    lift_via_pose,
    lift_via_pose_names,
)
from src.shared.python.lifting.pack_audit.names import CANONICAL_COORDINATES


# ---------------------------------------------------------------------------
# LIFT-10: Additional Movements
# ---------------------------------------------------------------------------


def test_all_five_additional_movements_defined() -> None:
    expected = ("split_jerk", "push_press", "overhead_press", "front_squat", "lunge")
    assert expected == ADDITIONAL_LIFTS
    for name in expected:
        assert is_additional_movement(name)
        spec = additional_movement_spec(name)
        assert isinstance(spec, AdditionalMovementSpec)
        assert spec.name == name
        assert spec.phase_count == len(spec.phases)


@pytest.mark.parametrize("lift", ADDITIONAL_LIFTS)
def test_additional_movement_phases_strictly_increase(lift: str) -> None:
    phases = lift_phases(lift)
    assert len(phases) >= 4
    fractions = [p.t_frac for p in phases]
    assert fractions[0] == 0.0
    assert fractions[-1] < 1.0
    assert all(b > a for a, b in zip(fractions, fractions[1:], strict=False))
    assert all(p.definition for p in phases)
    assert lift_phase_names(lift) == tuple(p.name for p in phases)


@pytest.mark.parametrize("lift", ADDITIONAL_LIFTS)
def test_additional_movement_root_joint_policy(lift: str) -> None:
    assert lift_root_joint(lift) == "free"


@pytest.mark.parametrize("lift", ADDITIONAL_LIFTS)
def test_additional_movement_grip(lift: str) -> None:
    grip = lift_grip(lift)
    assert grip.attach == "palm_centre"
    assert grip.width_range_m[0] <= grip.width_m <= grip.width_range_m[1]


@pytest.mark.parametrize("lift", ADDITIONAL_LIFTS)
def test_additional_movement_via_poses(lift: str) -> None:
    pose_names = lift_via_pose_names(lift)
    assert "start" in pose_names
    assert "lockout" in pose_names
    for name in pose_names:
        via = lift_via_pose(lift, name)
        assert via.name == name
        assert via.phase in lift_phase_names(lift)
        assert len(via.root_position_m) == 3
        assert len(via.bar_position_m) == 3
        unknown = set(via.coordinates) - set(CANONICAL_COORDINATES)
        assert not unknown, (
            f"{lift}/{name} has non-canonical coordinates {sorted(unknown)}"
        )


def test_split_stance_geometry_for_split_jerk_and_lunge() -> None:
    for name in ("split_jerk", "lunge"):
        spec = additional_movement_spec(name)
        assert spec.stance.is_split is True
        assert spec.stance.split_length_m > 0.5
        assert spec.stance.lead_foot == "l"
        assert spec.stance.trail_foot == "r"
        positions = spec.stance.foot_positions()
        assert "l" in positions and "r" in positions
        # Asymmetric along X axis
        assert positions["l"][0] > 0.0  # Lead foot forward
        assert positions["r"][0] < 0.0  # Trail foot back


def test_symmetric_stance_for_presses_and_squat() -> None:
    for name in ("push_press", "overhead_press", "front_squat"):
        spec = additional_movement_spec(name)
        assert spec.stance.is_split is False
        positions = spec.stance.foot_positions()
        assert positions["l"][0] == 0.0
        assert positions["r"][0] == 0.0
        assert positions["l"][1] == -positions["r"][1]  # Bilaterally symmetric


# ---------------------------------------------------------------------------
# LIFT-6: OpenSim Full-Body Lifts
# ---------------------------------------------------------------------------


def test_opensim_full_body_lift_spec_validation() -> None:
    spec = OpenSimFullBodyLiftSpec(lift="split_jerk", bar_mass_kg=100.0)
    assert spec.lift == "split_jerk"
    assert spec.bar_mass_kg == 100.0
    assert spec.body_model == "Rajagopal2015"

    with pytest.raises(ValueError, match="unsupported lift"):
        OpenSimFullBodyLiftSpec(lift="invalid_lift")

    with pytest.raises(ValueError, match="unsupported body_model"):
        OpenSimFullBodyLiftSpec(lift="squat", body_model="InvalidBody")

    with pytest.raises(ValueError, match="body_mass_kg must be positive"):
        OpenSimFullBodyLiftSpec(lift="squat", body_mass_kg=0.0)

    with pytest.raises(ValueError, match="reserve_actuator_threshold"):
        OpenSimFullBodyLiftSpec(lift="squat", reserve_actuator_threshold=1.5)


def test_opensim_composition_fails_closed_when_runtime_unavailable() -> None:
    spec = OpenSimFullBodyLiftSpec(lift="squat")
    composition = compose_opensim_full_body_lift(spec)
    assert isinstance(composition, OpenSimLiftModelComposition)
    # When native OpenSim is absent, status must be unqualified
    assert composition.status == "unqualified"
    assert composition.is_composed is False
    assert composition.reason is not None
    assert "OpenSim runtime" in composition.reason


def test_opensim_kinetics_evaluation_fails_closed_when_unavailable() -> None:
    spec = OpenSimFullBodyLiftSpec(lift="squat")
    kinetics = evaluate_opensim_lift_kinetics(spec)
    assert isinstance(kinetics, OpenSimLiftKineticsResult)
    assert kinetics.status == "unqualified"
    assert kinetics.is_qualified is False
    assert kinetics.hand_wrench.split_method == "unavailable"
    assert bool(kinetics.hand_wrench.unavailable_reason)
    assert kinetics.reason is not None


# ---------------------------------------------------------------------------
# LIFT-9: Showcase Videos and Renderer
# ---------------------------------------------------------------------------


def test_showcase_plan_creation_and_validation(tmp_path: Path) -> None:
    plan = render_lifting_showcase.plan_lifting_showcase(
        lift="split_jerk",
        engines=["mujoco", "drake"],
        views=["side", "front"],
        speeds=[1.0],
        overlays=["clean", "forces"],
        out_dir=tmp_path,
    )
    assert plan.lift == "split_jerk"
    assert plan.engines == ("mujoco", "drake")
    assert plan.views == ("side", "front")
    assert plan.fps == 60

    with pytest.raises(ValueError, match="unknown lift"):
        render_lifting_showcase.plan_lifting_showcase(lift="unknown_lift")

    with pytest.raises(ValueError, match="unknown engine"):
        render_lifting_showcase.plan_lifting_showcase(
            lift="squat", engines=["invalid_engine"]
        )


def test_headless_showcase_smoke_render(tmp_path: Path) -> None:
    plan = render_lifting_showcase.plan_lifting_showcase(
        lift="overhead_press",
        engines=["opensim"],
        views=["side"],
        speeds=[1.0],
        overlays=["clean"],
        out_dir=tmp_path,
    )
    manifest = render_lifting_showcase.render_lifting_showcase_smoke(
        plan, frames_per_variant=1
    )
    assert manifest["schema"] == "lifting-showcase-manifest/v1"
    assert manifest["lift"] == "overhead_press"
    assert len(manifest["entries"]) == 1
    # opensim is unavailable, should report unavailable
    entry = manifest["entries"][0]
    assert entry["engine"] == "opensim"
    assert entry["status"] == "unavailable"

    manifest_file = tmp_path / "overhead_press" / "manifest.json"
    assert manifest_file.exists()
    loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert loaded["lift"] == "overhead_press"
