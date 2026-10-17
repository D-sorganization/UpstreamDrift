"""The source back squat balances the bar load over mid-foot (OSL-15, #13093).

Before #13093 the source squat was posed without the bar: with the 100 kg bar
added on its back shelf the combined centre of mass of body and bar sat
0.15-0.19 m ahead of mid-foot (#13072), so the loaded re-track pivoted forward
on the toes.  The lift IK can now carry the bar load
(:class:`~src.shared.python.lifting.loaded_balance.LoadBalance`) and correct
the trunk angle frame by frame until the combined centre of mass lies over
mid-foot.  The tolerance below is fixed before the run.

The committed source bundle is not regenerated with the balance: tracking the
balanced IK with the bar load loses the feet inside the window (the mtp and
ankle tracking errors grow to 2.3 and 1.2 rad by the window end), so the
committed-bundle checks stay strict xfails until that tracking gap closes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.barbell_bushing import Barbell
from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE
from src.shared.python.lifting.loaded_balance import (
    LoadBalance,
    balance_lift,
    load_balance_from_record,
)
from src.shared.python.lifting.sagittal_retarget import SagittalSolution

pytestmark = pytest.mark.unit

#: Largest accepted forward distance of the body-plus-bar centre of mass from
#: mid-foot on the IK trajectory, metres (fixed before measuring).
BALANCE_TOLERANCE_M = 5e-4


def _solution(n: int = 4) -> SagittalSolution:
    return SagittalSolution(
        np.linspace(0.0, 1.0, n), np.zeros((n, 3)), np.zeros((n, 2)), {}
    )


def test_load_balance_rejects_bad_settings() -> None:
    with pytest.raises(TypeError, match="Barbell"):
        LoadBalance(barbell=100.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="segment"):
        LoadBalance(Barbell(100.0), segment="neck")
    with pytest.raises(ValueError, match="tolerance_m"):
        LoadBalance(Barbell(100.0), tolerance_m=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        LoadBalance(Barbell(100.0), max_iterations=0)


def test_load_balance_record_round_trips() -> None:
    load = LoadBalance(Barbell(100.0), tolerance_m=1e-3)
    assert load_balance_from_record(load.to_record()) == load


class _Lift:
    """Stand-in for a FullBodyLift: one row of ``q`` per solution sample."""

    def __init__(self, q: np.ndarray) -> None:
        self.q = q


def test_balance_lift_solves_each_frame_for_zero_offset() -> None:
    """A smooth monotone offset is driven below tolerance frame by frame."""
    target = np.array([0.30, -0.10, 0.05, 0.20])

    def solve(solution: SagittalSolution) -> _Lift:
        return _Lift(solution.segment_angles_rad.copy())

    def offsets(q: np.ndarray) -> np.ndarray:
        # Forward offset grows with the trunk angle (non-linear, monotone).
        return 0.4 * np.sin(q[:, 2] - target) + 0.05 * (q[:, 2] - target) ** 3

    load = LoadBalance(Barbell(100.0), tolerance_m=1e-6)
    lift, record = balance_lift(solve, _solution(), offsets, load)
    assert np.abs(offsets(lift.q)).max() <= 1e-6
    np.testing.assert_allclose(record["correction_rad"], target, atol=1e-5)
    assert record["max_offset_m"] <= 1e-6
    assert record["unbalanced_max_offset_m"] == pytest.approx(
        float(np.abs(offsets(np.zeros((4, 3)))).max())
    )


def test_balance_lift_reports_a_frame_it_cannot_balance() -> None:
    def solve(solution: SagittalSolution) -> _Lift:
        return _Lift(solution.segment_angles_rad.copy())

    def offsets(q: np.ndarray) -> np.ndarray:
        return 0.1 + 0.0 * q[:, 2]  # the trunk cannot move the centre of mass

    with pytest.raises(RuntimeError, match="centre of mass"):
        balance_lift(solve, _solution(), offsets, LoadBalance(Barbell(100.0)))


def test_balance_lift_rejects_an_offset_of_the_wrong_shape() -> None:
    def solve(solution: SagittalSolution) -> _Lift:
        return _Lift(solution.segment_angles_rad.copy())

    with pytest.raises(ValueError, match="one offset per"):
        balance_lift(
            solve, _solution(), lambda q: np.zeros(2), LoadBalance(Barbell(100.0))
        )


def test_lift_to_full_body_needs_the_back_bar_for_a_load() -> None:
    from src.shared.python.lifting.bundles import (
        LIFT_SPEC,
        lift_to_full_body,
        load_seed_pose,
    )

    with pytest.raises(ValueError, match="back-bar"):
        lift_to_full_body(
            LIFT_SPEC.read_bytes(),
            _solution(),
            load_seed_pose(),
            load=LoadBalance(Barbell(100.0)),
        )


# -- the balance on the real lift body ---------------------------------------


def _bottom_frames(bundle, step: int = 5, count: int = 3) -> SagittalSolution:  # type: ignore[no-untyped-def]
    """``count`` planar samples, ``step`` apart, from the squat bottom."""
    solution = SagittalSolution.from_record(bundle.provenance["sagittal_solution"])
    bottom = int(np.argmin(solution.bar_m[:, 1]))
    rows = bottom + step * (np.arange(count) - count // 2)
    return SagittalSolution(
        solution.time_s[rows],
        solution.segment_angles_rad[rows],
        solution.bar_m[rows],
        dict(solution.source),
    )


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_ik_balances_the_bar_load_over_mid_foot_at_the_bottom(source) -> None:  # type: ignore[no-untyped-def]
    """Three bottom frames of the committed solution, balanced from scratch.

    The record's offset is recomputed independently from the joint rows on
    the rigid-bar body (``loaded_com_offsets``), and the unbalanced offset is
    the #13072 cause B magnitude.
    """
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.back_bar import BackBarGeometry
    from src.shared.python.lifting.bundles import (
        LIFT_SPEC,
        lift_bundle_spec,
        lift_to_full_body,
        load_seed_pose,
    )
    from src.shared.python.lifting.loaded_balance import loaded_com_offsets

    geom = source.provenance["ik"]["back_bar"]["geometry"]
    geometry = BackBarGeometry.from_standard(geom["lift"], height_m=geom["height_m"])
    load = LoadBalance(Barbell(float(source.provenance["optimizer"]["bar_mass_kg"])))
    lift = lift_to_full_body(
        LIFT_SPEC.read_bytes(),
        _bottom_frames(source),
        load_seed_pose(),
        geometry=geometry,
        load=load,
    )
    assert lift.balance is not None
    assert lift.balance["max_offset_m"] <= BALANCE_TOLERANCE_M
    assert lift.balance["unbalanced_max_offset_m"] > 0.1
    offsets = loaded_com_offsets(lift.back_bar.spec_bytes, lift.q, load)
    assert float(np.abs(offsets).max()) <= BALANCE_TOLERANCE_M
    assert lift_bundle_spec(lift) != lift.back_bar.spec_bytes


# -- the committed source bundle (OSL-15 gap) ---------------------------------

LOADED_TRACKING_GAP = (
    "OSL-15 (#13093): the committed source is the unbalanced bodyweight "
    "bundle; tracking the balanced IK with the bar load loses the feet "
    "(lift_models.tex, OSL-15), so it is not regenerated yet"
)


@pytest.fixture(scope="module")
def source():  # type: ignore[no-untyped-def]
    from src.shared.python.motion_matching.same_input import InputBundle

    return InputBundle.load(BACK_SQUAT_BUNDLE)


@pytest.mark.xfail(strict=True, raises=KeyError, reason=LOADED_TRACKING_GAP)
def test_source_ik_records_the_bar_load_balance(source) -> None:  # type: ignore[no-untyped-def]
    balance = source.provenance["ik"]["balance"]
    load = load_balance_from_record(balance["load"])
    assert load.barbell.load_kg == source.provenance["optimizer"]["bar_mass_kg"]
    assert balance["max_offset_m"] <= BALANCE_TOLERANCE_M
    assert len(balance["correction_rad"]) == source.provenance["ik"]["frames"]


@pytest.mark.xfail(strict=True, raises=KeyError, reason=LOADED_TRACKING_GAP)
def test_source_dynamics_carry_the_bar_load(source) -> None:  # type: ignore[no-untyped-def]
    """A balanced source is tracked with the bar welded to the hands.

    A bodyweight tracking of the balanced posture tips backward (its body
    centre of mass alone sits behind the heels), so a balanced source's spec
    is ``rigid_bar_spec`` of the derived back-bar spec.
    """
    import json

    from src.shared.python.lifting.barbell_runs import (
        RIGID_BAR_SOLID,
        bushing_body_spec,
        rigid_bar_spec,
    )

    load = load_balance_from_record(source.provenance["ik"]["balance"]["load"])
    body = bushing_body_spec(source.spec_bytes)
    assert rigid_bar_spec(body, load.barbell) == source.spec_bytes
    solids = [
        s["name"] for b in json.loads(source.spec_bytes)["bodies"] for s in b["solids"]
    ]
    assert solids.count(RIGID_BAR_SOLID) == 1
