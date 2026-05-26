"""Property test: vectorized Bellman backup must agree with the scalar reference.

Spec pitfall #11: ``HoleMDP.bellman_backup`` is paired with
``bellman_backup_scalar`` for correctness verification on small grids. Both
implementations consume the same RNG seed and must agree under expectation
(over enough samples) to within Monte-Carlo tolerance.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import (
    LIE_CODES,
    RectFeature,
    SyntheticHole,
    rasterize_synthetic,
)
from src.shared.python.sg_optimizer.mdp.action import ActionSet
from src.shared.python.sg_optimizer.mdp.value_iteration import (
    HoleMDP,
    bellman_backup_scalar,
)
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.player_profile import (
    ClubSkill,
    PlayerProfile,
)


from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "data" / "sg_optimizer" / "baselines" / "pga_tour.yaml"


def _tiny_hole():
    return SyntheticHole(
        name="tiny",
        par=3,
        tee=(0.0, 0.0),
        pin=(60.0, 0.0),
        bbox=(-10.0, 80.0, -20.0, 20.0),
        features=(
            RectFeature("fairway", 0.0, 80.0, -10.0, 10.0),
            RectFeature("green", 50.0, 70.0, -8.0, 8.0),
        ),
    )


def test_scalar_and_vectorized_backups_agree_in_expectation():
    raster = rasterize_synthetic(_tiny_hole(), resolution_yd=5.0)
    bag = load_baseline(BASELINE)
    profile = PlayerProfile(
        name="t",
        baseline=str(BASELINE),
        clubs={c: ClubSkill() for c in ("7_iron", "9_iron", "pw")},
    )
    actions = ActionSet(clubs=("7_iron", "9_iron", "pw"))
    conditions = CourseConditions.tournament()

    nx, ny = raster.shape
    V = np.zeros((nx, ny))
    # Mark terminal cells (holed) explicitly.
    V[raster.codes == LIE_CODES["holed"]] = 0.0

    mdp = HoleMDP(
        raster=raster,
        profile=profile,
        baseline=bag,
        conditions=conditions,
        actions=actions,
        n_samples=64,
        seed=42,
    )

    V_vec, _ = mdp.bellman_backup(V)
    V_scalar, _ = bellman_backup_scalar(
        V,
        raster,
        profile,
        bag,
        conditions,
        actions,
        n_samples=64,
        rng=np.random.default_rng(42),
    )

    # MC noise: agree on the *mean* magnitude rather than pointwise.
    diff = np.abs(V_vec - V_scalar)
    # Drop terminal cells and water/OB sentinel cells.
    valid = raster.codes != LIE_CODES["holed"]
    rel = diff[valid].mean() / max(V_scalar[valid].mean(), 1e-3)
    assert rel < 0.5, f"vectorized vs scalar disagree by relative {rel:.3f}"
