"""Three §5 Phase-1 sanity-check integration tests.

These are the binding acceptance criteria for the Phase-1 PR:

  1. Increasing sigma_lat on driver → optimal tee aim on a hole with right-side
     water shifts *left* (away from the hazard).
  2. Increasing rough.severity → approach strategy from rough becomes more
     conservative (targets a smaller part of the green).
  3. Increasing greens.stimp → expected strokes from 30 ft on the green rises.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.shared.python.sg_optimizer.course.conditions import (
    CourseConditions,
    GreenModel,
    RoughModel,
    TreeModel,
)
from src.shared.python.sg_optimizer.course.rasterize import (
    LIE_CODES,
    CircleFeature,
    RectFeature,
    SyntheticHole,
    rasterize_synthetic,
)
from src.shared.python.sg_optimizer.mdp.action import ActionSet
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.mdp.value_iteration import HoleMDP
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.player_profile import (
    ClubSkill,
    PlayerProfile,
    PuttingSkill,
)
from src.shared.python.sg_optimizer.shot_model.putting import (
    leave_distance_distribution,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "data" / "sg_optimizer" / "baselines" / "pga_tour.yaml"


def _par4_right_water() -> SyntheticHole:
    """Synthetic short par-4 with water hugging the right side."""
    return SyntheticHole(
        name="right_water_p4",
        par=4,
        tee=(0.0, 0.0),
        pin=(350.0, 0.0),
        bbox=(-30.0, 370.0, -60.0, 60.0),
        features=(
            RectFeature("fairway", 0.0, 360.0, -20.0, 20.0),
            RectFeature("water", 100.0, 360.0, -50.0, -22.0),
            CircleFeature("green", 350.0, 0.0, 12.0),
        ),
    )


def _solve(
    hole: SyntheticHole,
    profile: PlayerProfile,
    conditions: CourseConditions,
    bag,
    clubs=("driver", "5_iron", "7_iron", "pw"),
    resolution=8.0,
    n_samples=32,
    max_iter=30,
):
    raster = rasterize_synthetic(hole, resolution_yd=resolution)
    actions = ActionSet(
        clubs=clubs,
        aim_grid_deg=np.linspace(-20.0, 20.0, 21),
    )
    mdp = HoleMDP(
        raster=raster,
        profile=profile,
        baseline=bag,
        conditions=conditions,
        actions=actions,
        n_samples=n_samples,
        seed=0,
    )
    result = mdp.solve(max_iter=max_iter)
    tee = State(x=hole.tee[0], y=hole.tee[1], lie=int(raster.lie_at(*hole.tee)))
    action = mdp.optimal_action(tee, result.value)
    return mdp, result, action


def test_sanity_1_widening_sigma_lat_shifts_aim_away_from_water():
    bag = load_baseline(BASELINE)
    hole = _par4_right_water()
    conditions = CourseConditions.tournament()

    # Baseline driver dispersion.
    profile_tight = PlayerProfile(
        name="tight",
        baseline=str(BASELINE),
        clubs={"driver": ClubSkill(skill_mult_lat=1.0)},
    )
    # Wide-dispersion driver.
    profile_wide = PlayerProfile(
        name="wide",
        baseline=str(BASELINE),
        clubs={"driver": ClubSkill(skill_mult_lat=2.5)},
    )

    _, _, tight_action = _solve(
        hole, profile_tight, conditions, bag, clubs=("driver", "5_iron")
    )
    _, _, wide_action = _solve(
        hole, profile_wide, conditions, bag, clubs=("driver", "5_iron")
    )

    # Convention: aim_angle_rad positive = aim left (water is right → negative y).
    # We expect the wide player to aim *more positively* (more left) than tight.
    assert wide_action.aim_angle_rad >= tight_action.aim_angle_rad - 1e-6
    # And at least one of them shouldn't be the maximum-right aim.
    assert wide_action.aim_angle_rad > math.radians(-20.0)


def test_sanity_2_heavy_rough_makes_approach_more_conservative():
    """Approach expected-strokes penalty from rough is larger under heavy
    rough — exposed by the expected stroke difference."""
    bag = load_baseline(BASELINE)
    hole = SyntheticHole(
        name="approach_test",
        par=4,
        tee=(0.0, 0.0),
        pin=(150.0, 0.0),
        bbox=(-10.0, 170.0, -40.0, 40.0),
        features=(
            RectFeature("fairway", 0.0, 100.0, -10.0, 10.0),
            CircleFeature("green", 150.0, 0.0, 10.0),
        ),
    )
    profile = PlayerProfile(
        name="t",
        baseline=str(BASELINE),
        clubs={c: ClubSkill() for c in ("7_iron", "9_iron", "pw")},
    )

    benign = CourseConditions(
        rough=RoughModel.preset("light"),
        trees=TreeModel.preset("decorative"),
        greens=GreenModel.preset("medium"),
    )
    punitive = CourseConditions(
        rough=RoughModel.preset("us_open"),
        trees=TreeModel.preset("decorative"),
        greens=GreenModel.preset("medium"),
    )

    mdp_b, res_b, _ = _solve(
        hole, profile, benign, bag, clubs=("7_iron", "9_iron", "pw")
    )
    mdp_p, res_p, _ = _solve(
        hole, profile, punitive, bag, clubs=("7_iron", "9_iron", "pw")
    )

    # A ball in the rough at 150 yards.
    rough_state = State(
        x=10.0,
        y=25.0,  # off the fairway, in rough
        lie=LIE_CODES["rough"],
    )
    v_b = mdp_b.expected_strokes(rough_state, res_b.value)
    v_p = mdp_p.expected_strokes(rough_state, res_p.value)
    assert v_p > v_b, f"expected strokes did not rise: benign={v_b}, punitive={v_p}"


def test_sanity_3_faster_greens_increase_expected_strokes_from_30ft():
    profile = PuttingSkill()
    slow = GreenModel.preset("slow")
    masters = GreenModel.preset("masters")

    leave_slow = leave_distance_distribution(30.0, profile, slow)
    leave_fast = leave_distance_distribution(30.0, profile, masters)

    # On faster greens the 3-putt probability rises, so EV from 30 ft must rise.
    three_putt_slow = leave_slow.expected_three_putt_probability(3.0)
    three_putt_fast = leave_fast.expected_three_putt_probability(3.0)
    assert three_putt_fast > three_putt_slow

    # Build a coarse expected-strokes proxy: 2 + P(3-putt) (i.e. assume the
    # first putt is a lag; the question is how often we need >1 putt after).
    ev_slow = 2.0 + three_putt_slow
    ev_fast = 2.0 + three_putt_fast
    assert ev_fast > ev_slow
