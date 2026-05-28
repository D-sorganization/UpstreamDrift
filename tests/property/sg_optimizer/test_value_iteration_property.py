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


def test_action_q_responds_to_shot_bias():
    """Regression for #6537.

    ``HoleMDP._action_q`` must apply ``ClubSkill.bias_long``/``bias_lat``
    to the vectorized landing offsets. Previously the bias was dropped
    on the vectorized path, so two profiles differing *only* by a large
    chronic miss produced identical Q-values for the same action.

    We check this at the ``_action_q`` level (single fixed action) so
    the optimizer cannot "aim around" the bias and hide the bug.
    """
    from src.shared.python.sg_optimizer.mdp.action import ShotAction

    # Use a wide hole so iron-length shots actually land in-bounds —
    # otherwise OOB hazard masking would hide the bias signal.
    wide_hole = SyntheticHole(
        name="wide",
        par=4,
        tee=(0.0, 0.0),
        pin=(180.0, 0.0),
        bbox=(-20.0, 240.0, -60.0, 60.0),
        features=(
            RectFeature("fairway", 0.0, 240.0, -40.0, 40.0),
            RectFeature("green", 170.0, 195.0, -10.0, 10.0),
        ),
    )
    raster = rasterize_synthetic(wide_hole, resolution_yd=5.0)
    bag = load_baseline(BASELINE)
    unbiased = PlayerProfile(
        name="unbiased",
        baseline=str(BASELINE),
        clubs={c: ClubSkill() for c in ("7_iron", "9_iron", "pw")},
    )
    biased = PlayerProfile(
        name="biased",
        baseline=str(BASELINE),
        clubs={c: ClubSkill(bias_lat=12.0) for c in ("7_iron", "9_iron", "pw")},
    )
    actions = ActionSet(clubs=("7_iron", "9_iron", "pw"))
    conditions = CourseConditions.tournament()

    nx, ny = raster.shape
    # Non-trivial V: cost grows away from centerline so the lateral bias
    # actually shifts the expected cost.
    cell_y = raster.origin[1] + (np.arange(ny) + 0.5) * raster.resolution_yd
    V = np.abs(cell_y)[None, :].repeat(nx, axis=0).astype(np.float64)
    V[raster.codes == LIE_CODES["holed"]] = 0.0

    def _q(profile):
        mdp = HoleMDP(
            raster=raster,
            profile=profile,
            baseline=bag,
            conditions=conditions,
            actions=actions,
            n_samples=64,
            seed=42,
        )
        return mdp._action_q(V, ShotAction(club="7_iron", aim_angle_rad=0.0))

    q_unb = _q(unbiased)
    q_bia = _q(biased)
    valid = raster.codes != LIE_CODES["holed"]
    delta = float(np.abs(q_unb - q_bia)[valid].mean())
    assert delta > 0.05, (
        f"biased vs unbiased _action_q differ by only {delta:.4g} — the vectorized "
        "backup appears to ignore ClubSkill.bias_lat (issue #6537)."
    )
