"""Arms refitted to the moved wrist joints with the club held still (#15255).

Moving the wrist joints to the skinned hands' wrists (#15253) leaves a pose
fitted on the unmoved model with the club centimetres out of place.
``refit_arms_to_club`` solves the arm coordinates again so the club is where
the pose had it, the trail hand closes on the grip and nothing else moves.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.skinned_arm_refit import (  # noqa: E402
    ARM_COORDINATE_KEYS,
    ArmRefit,
    ArmRefitSettings,
    refit_arms_to_club,
    refit_arms_with,
)
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (  # noqa: E402
    apply_skinned_hand_wrists,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.timeout(900)]
ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
#: Position (m) and angle (deg) the solver must close to.
TIGHT_M, TIGHT_DEG = 1e-4, 0.05
#: An upper arm may turn this far (deg) about its own axis in a refit, and the
#: two arms' turns may differ by this much (#15297).
TWIST_DEG, TWIST_APART_DEG = 10.0, 10.0
#: A refitted wrist coordinate stays this far (deg) inside its range.
WRIST_MARGIN_DEG = 1.0
#: Wrist ranges (deg) of specs built before the evidence ranges of #14795.
LEGACY_WRIST_DEG = {"X": (-40.0, 25.0), "Y": (-70.0, 70.0)}


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def closed(base: dict[str, Any]) -> ArmRefit:
    """The spec's address seed with the trail hand closed on the club.

    The seed itself leaves the hand off the grip, so the club is let go
    (``hold_club=False``) and both arms bring the hand and the club together.
    """
    order = base["coordinate_order"]
    seed = np.zeros(len(order))
    for name, degrees in base["address_seed_deg"].items():
        seed[order.index(name)] = np.radians(degrees)
    return refit_arms_to_club(base, base, seed, alignment_weight=0.0, hold_club=False)


@pytest.fixture(scope="module")
def refit(base: dict[str, Any], closed: ArmRefit) -> ArmRefit:
    return refit_arms_to_club(base, apply_skinned_hand_wrists(base), closed.q)


def test_a_pose_closes_when_the_club_is_let_go(closed: ArmRefit) -> None:
    assert closed.closure_before_m > 0.03
    assert closed.closure_m <= TIGHT_M
    assert closed.closure_deg <= TIGHT_DEG
    # The club was not held, and it went to meet the hand.
    assert closed.club_offset_m > 0.03


def test_the_club_stays_where_the_pose_had_it(refit: ArmRefit) -> None:
    # Unrefitted, the moved joints carry the club centimetres away.
    assert refit.club_offset_before_m > 0.03
    assert refit.club_offset_m <= TIGHT_M
    assert refit.club_turn_deg <= TIGHT_DEG


def test_the_trail_hand_closes_on_the_grip(refit: ArmRefit) -> None:
    assert refit.closure_before_m > 0.03
    assert refit.closure_m <= TIGHT_M
    assert refit.closure_deg <= TIGHT_DEG


def test_only_the_arms_move_and_stay_in_range(base, closed, refit) -> None:
    order = base["coordinate_order"]
    arms = [
        i
        for i, name in enumerate(order)
        if name[0] in "LR" and any(key in name for key in ARM_COORDINATE_KEYS)
    ]
    assert len(arms) == 18
    others = [i for i in range(len(order)) if i not in arms]
    np.testing.assert_array_equal(refit.q[others], closed.q[others])
    assert float(np.abs(refit.q[arms] - closed.q[arms]).max()) > np.radians(1.0)
    ranges = base["coordinate_ranges_deg"]
    for i in arms:
        low, high = ranges.get(order[i], (-180.0, 180.0))
        assert low - 1e-6 <= np.degrees(refit.q[i]) <= high + 1e-6


def test_the_forearms_turn_toward_the_hands(base, closed, refit) -> None:
    moved = apply_skinned_hand_wrists(base)
    free = refit_arms_to_club(base, moved, closed.q, alignment_weight=0.0)
    for side in ("lead", "trail"):
        assert refit.forearm_to_cuff_deg[side] <= free.forearm_to_cuff_deg[side] + 0.5
    assert sum(refit.forearm_to_cuff_deg.values()) < sum(
        free.forearm_to_cuff_deg.values()
    )


def test_named_trunk_coordinates_move_with_the_arms(base, closed) -> None:
    order = base["coordinate_order"]
    trunk = ("lumbar_flexion", "thoracic_flexion")
    moved = apply_skinned_hand_wrists(base)
    freed = refit_arms_to_club(base, moved, closed.q, also_free=trunk)
    assert freed.club_offset_m <= TIGHT_M
    assert freed.closure_m <= TIGHT_M
    changed = {order[i] for i in np.flatnonzero(freed.q != closed.q)}
    assert changed & set(trunk)
    assert all(
        name in trunk or any(key in name for key in ARM_COORDINATE_KEYS)
        for name in changed
    )
    with pytest.raises(ValueError, match="also_free"):
        refit_arms_to_club(base, moved, closed.q, also_free=("no_such_joint",))


def test_bad_input_is_refused(base) -> None:
    with pytest.raises(ValueError, match="coordinate"):
        refit_arms_to_club(base, base, np.zeros(3))
    with pytest.raises(ValueError, match="alignment_weight"):
        refit_arms_to_club(
            base, base, np.zeros(len(base["coordinate_order"])), alignment_weight=-1.0
        )


def test_a_posture_puts_the_bend_in_the_thoracic_joint_not_the_lumbar(
    base, closed
) -> None:
    """Owner request (#15277): a neutral lower back and a slight thoracic bend.

    Left free with no posture, the trunk arches at the lumbar joint to line
    the forearms up; a posture names where each trunk joint should sit.
    """
    order = base["coordinate_order"]
    moved = apply_skinned_hand_wrists(base)
    posture = {"lumbar_flexion": np.radians(2.0), "thoracic_flexion": np.radians(10.0)}
    posed = refit_arms_to_club(base, moved, closed.q, posture=posture)
    assert posed.club_offset_m <= TIGHT_M
    assert posed.closure_m <= TIGHT_M
    for name, target in posture.items():
        got = posed.q[order.index(name)]
        assert abs(np.degrees(got - target)) < 1.0, (name, np.degrees(got))
    changed = {order[i] for i in np.flatnonzero(posed.q != closed.q)}
    assert all(
        name in posture or any(key in name for key in ARM_COORDINATE_KEYS)
        for name in changed
    )


def test_a_freed_trunk_joint_stays_inside_its_soft_limits(base, closed) -> None:
    order = base["coordinate_order"]
    moved = apply_skinned_hand_wrists(base)
    far = {"thoracic_flexion": np.radians(80.0)}  # well past the 30 degree limit
    posed = refit_arms_to_club(base, moved, closed.q, posture=far)
    low, high = base["soft_limits_deg"]["thoracic_flexion"]
    got = np.degrees(posed.q[order.index("thoracic_flexion")])
    assert low - 1e-6 <= got <= high + 1e-6


def test_a_bad_posture_is_refused(base) -> None:
    q = np.zeros(len(base["coordinate_order"]))
    with pytest.raises(ValueError, match="posture"):
        refit_arms_to_club(base, base, q, posture={"no_such_joint": 0.0})
    with pytest.raises(ValueError, match="posture"):
        refit_arms_to_club(base, base, q, posture={"lumbar_flexion": float("nan")})
    with pytest.raises(ValueError, match="posture_weight"):
        refit_arms_to_club(
            base, base, q, posture={"lumbar_flexion": 0.0}, posture_weight=-1.0
        )


def test_the_address_posture_is_a_neutral_lower_back_and_a_slight_thoracic_bend() -> (
    None
):
    from src.engines.physics_engines.mujoco.python import skinned_arm_refit as refit

    deg = refit.ADDRESS_POSTURE_DEG
    assert abs(deg["lumbar_flexion"]) <= 5.0
    assert 5.0 <= deg["thoracic_flexion"] <= 15.0
    assert refit.address_posture() == {k: np.radians(v) for k, v in deg.items()}


@pytest.fixture(scope="module")
def moved(base: dict[str, Any]) -> dict[str, Any]:
    return apply_skinned_hand_wrists(base)


@pytest.fixture(scope="module")
def held(base: dict[str, Any], moved: dict[str, Any]) -> ArmRefit:
    """The address seed closed on the skinned wrists: a pose those hands hold."""
    order = base["coordinate_order"]
    seed = np.zeros(len(order))
    for name, degrees in base["address_seed_deg"].items():
        seed[order.index(name)] = np.radians(degrees)
    return refit_arms_to_club(base, moved, seed, alignment_weight=0.0, hold_club=False)


def _wrists_deg(spec: dict[str, Any], fit: ArmRefit) -> dict[str, float]:
    order = spec["coordinate_order"]
    return {n: float(np.degrees(fit.q[order.index(n)])) for n in order if "WInput" in n}


def test_the_forearm_pull_cannot_roll_an_upper_arm_over(moved, held) -> None:
    """#15297: unheld, the forearm pull rolls an upper arm over by 66 degrees."""
    fit = refit_arms_to_club(moved, moved, held.q)
    assert fit.closure_m <= TIGHT_M and fit.club_offset_m <= TIGHT_M
    twist = fit.upper_arm_twist_deg
    for side in ("lead", "trail"):
        assert abs(twist[side]) <= TWIST_DEG, twist
    assert abs(twist["lead"] - twist["trail"]) <= TWIST_APART_DEG, twist
    ranges = moved["coordinate_ranges_deg"]
    for name, value in _wrists_deg(moved, fit).items():
        low, high = ranges[name]
        assert low + WRIST_MARGIN_DEG <= value <= high - WRIST_MARGIN_DEG, name
    unheld = ArmRefitSettings(twist_stay_weight=0.0)
    rolled = refit_arms_with(unheld, moved, moved, held.q).upper_arm_twist_deg
    assert max(abs(turn) for turn in rolled.values()) > 3.0 * TWIST_DEG, rolled


def test_legacy_wrist_ranges_are_widened_to_the_evidence_ranges(moved, held) -> None:
    narrow = copy.deepcopy(moved)
    ranges = narrow["coordinate_ranges_deg"]
    wide = {name: tuple(ranges[name]) for name in ranges if "WInput" in name}
    for name in wide:
        ranges[name] = list(LEGACY_WRIST_DEG[name[-1]])
    fit = refit_arms_to_club(moved, narrow, held.q)
    assert fit.closure_m <= TIGHT_M and fit.club_offset_m <= TIGHT_M
    # Room to move does not let an arm flip, and no wrist ends on a bound.
    assert max(abs(turn) for turn in fit.upper_arm_twist_deg.values()) <= TWIST_DEG
    wrists = _wrists_deg(narrow, fit)
    for name, value in wrists.items():
        low, high = wide[name]
        assert low + WRIST_MARGIN_DEG <= value <= high - WRIST_MARGIN_DEG, name
    # The pose needs a wrist past its legacy range, and the refit may use it.
    assert any(v > LEGACY_WRIST_DEG[n[-1]][1] + 1.0 for n, v in wrists.items())
    kept = refit_arms_with(
        ArmRefitSettings(evidence_wrist_bounds=False), moved, narrow, held.q
    )
    for name, value in _wrists_deg(narrow, kept).items():
        low, high = LEGACY_WRIST_DEG[name[-1]]
        assert low - 1e-6 <= value <= high + 1e-6, name


def test_bad_settings_are_refused() -> None:
    for name in ("alignment_weight", "twist_stay_weight"):
        for bad in (-1.0, float("nan")):
            with pytest.raises(ValueError, match=name):
                ArmRefitSettings(**{name: bad})
