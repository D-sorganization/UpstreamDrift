"""Arms refitted to the moved wrist joints with the club held still (#15255).

Moving the wrist joints to the skinned hands' wrists (#15253) leaves a pose
fitted on the unmoved model with the club centimetres out of place.
``refit_arms_to_club`` solves the arm coordinates again so the club is where
the pose had it, the trail hand closes on the grip and nothing else moves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.skinned_arm_refit import (  # noqa: E402
    ARM_COORDINATE_KEYS,
    ArmRefit,
    refit_arms_to_club,
)
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (  # noqa: E402
    apply_skinned_hand_wrists,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.timeout(900)]
ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
#: Position (m) and angle (deg) the solver must close to.
TIGHT_M, TIGHT_DEG = 1e-4, 0.05


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
