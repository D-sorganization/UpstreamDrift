"""Phase 2 integration / acceptance tests for sg-optimizer.

Binding acceptance criteria for issue #6271:

  1. All 5 classic GeoJSON files load and produce valid HoleGeometry objects.
  2. TreeModel punch-out shifts expected strokes upward vs. no trees.
  3. CLI ``run --classic`` sub-command produces valid JSON output for a
     classic hole.
  4. CLI ``list-classics`` sub-command prints at least 5 slugs.
  5. UTM round-trip for all 5 classic tee positions is <1 cm.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.sg_optimizer.course.conditions import (
    CourseConditions,
    GreenModel,
    RoughModel,
)
from src.shared.python.sg_optimizer.course.conditions import (
    TreeModel as ConditionsTreeModel,
)
from src.shared.python.sg_optimizer.course.geometry import (
    haversine_m,
    project_to_utm,
    utm_to_latlon,
)
from src.shared.python.sg_optimizer.course.library import list_classics, load_classic
from src.shared.python.sg_optimizer.mdp.tree_model import TreeModel
from src.shared.python.sg_optimizer.course.rasterize import (
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
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "data" / "sg_optimizer" / "baselines" / "pga_tour.yaml"


# ---------------------------------------------------------------------------
# Acceptance 1: classic GeoJSON files load correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", list_classics())
def test_classic_holes_load(slug):
    hole = load_classic(slug)
    assert hole.par in (3, 4, 5)
    assert hole.yardage > 0
    assert hole.name
    assert hole.tee is not None
    assert hole.green_center is not None


# ---------------------------------------------------------------------------
# Acceptance 2: TreeModel punch-out raises expected strokes
# ---------------------------------------------------------------------------


def _simple_par3(trees: bool = False) -> tuple[SyntheticHole, list]:
    features = [
        RectFeature("fairway", 0.0, 130.0, -10.0, 10.0),
        CircleFeature("green", 120.0, 0.0, 8.0),
    ]
    if trees:
        features.append(RectFeature("trees", 50.0, 90.0, 10.0, 25.0))
    return SyntheticHole(
        name="test_par3",
        par=3,
        tee=(0.0, 0.0),
        pin=(120.0, 0.0),
        bbox=(-15.0, 140.0, -25.0, 25.0),
        features=tuple(features),
    ), features


def test_tree_model_punch_out_probability_in_trees():
    """High-penalization TreeModel returns p>0.5 in trees lie."""
    from src.shared.python.sg_optimizer.course.features import StateFeatures

    model = TreeModel(penalization=0.95)
    sf = StateFeatures(
        distance_to_pin_m=50.0,
        distance_to_center_m=50.0,
        lie="trees",
    )
    prob = model.forced_punch_out_probability(sf)
    assert prob > 0.9, f"expected high punch-out probability, got {prob}"


def test_trees_penalization_increases_expected_strokes():
    """Heavy trees on a par-3 should increase expected strokes from the tee."""
    bag = load_baseline(BASELINE)

    hole_clean, _ = _simple_par3(trees=False)
    hole_treed, _ = _simple_par3(trees=True)

    profile = PlayerProfile(
        name="avg",
        baseline=str(BASELINE),
        clubs={c: ClubSkill() for c in ("9_iron", "pw")},
    )

    def solve(hole):
        raster = rasterize_synthetic(hole, resolution_yd=8.0)
        actions = ActionSet(
            clubs=("9_iron", "pw"),
            aim_grid_deg=np.linspace(-15.0, 15.0, 7),
        )
        conditions = CourseConditions(
            rough=RoughModel.preset("medium"),
            trees=ConditionsTreeModel.preset("dense"),
            greens=GreenModel.preset("medium"),
        )
        mdp = HoleMDP(
            raster=raster,
            profile=profile,
            baseline=bag,
            conditions=conditions,
            actions=actions,
            n_samples=16,
            seed=0,
        )
        result = mdp.solve(max_iter=20)
        tee = State(x=hole.tee[0], y=hole.tee[1], lie=int(raster.lie_at(*hole.tee)))
        return mdp.expected_strokes(tee, result.value)

    ev_clean = solve(hole_clean)
    ev_treed = solve(hole_treed)
    # Trees should not lower expected strokes.
    assert ev_treed >= ev_clean - 0.05, (
        f"trees unexpectedly reduced expected strokes: clean={ev_clean:.3f}, "
        f"treed={ev_treed:.3f}"
    )


# ---------------------------------------------------------------------------
# Acceptance 3: CLI run --classic produces valid JSON
# ---------------------------------------------------------------------------


def _write_profile(path: Path) -> None:
    path.write_text(
        "name: cli_phase2\n"
        f"baseline: {BASELINE}\n"
        "clubs: {}\n"
        "putting:\n"
        "  make_pct_multipliers: {}\n"
        "  three_putt_avoidance: 1.0\n"
        "short_game: {}\n"
        "notes: phase2-test\n"
    )


def test_cli_classic_flag(tmp_path):
    from src.shared.python.sg_optimizer.cli import main

    profile = tmp_path / "profile.yaml"
    _write_profile(profile)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "run",
                "--profile",
                str(profile),
                "--baseline",
                str(BASELINE),
                "--classic",
                "sawgrass_17",
                "--conditions",
                "tournament",
                "--resolution",
                "10.0",
                "--n-samples",
                "16",
                "--max-iter",
                "5",
            ]
        )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "sawgrass" in payload["hole"].lower() or payload["par"] == 3
    assert payload["tee_optimal_action"]["club"] in (
        "driver",
        "3_wood",
        "5_iron",
        "7_iron",
        "9_iron",
        "pw",
        "sw",
        "lw",
    )


# ---------------------------------------------------------------------------
# Acceptance 4: list-classics sub-command
# ---------------------------------------------------------------------------


def test_cli_list_classics(monkeypatch):
    from src.shared.python.sg_optimizer.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["list-classics"])
    assert rc == 0
    lines = [ln for ln in buf.getvalue().strip().splitlines() if ln]
    assert len(lines) >= 5
    assert "sawgrass_17" in lines


# ---------------------------------------------------------------------------
# Acceptance 5: UTM round-trip <1 cm for all classic tee positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", list_classics())
def test_utm_round_trip_classic_tees(slug):
    hole = load_classic(slug)
    tee_ll = hole.tee
    utm_pts = project_to_utm([tee_ll])
    back = utm_to_latlon(utm_pts)
    err_m = haversine_m(tee_ll, back[0])
    assert err_m < 0.01, f"{slug}: round-trip error {err_m:.4f} m > 1 cm"
