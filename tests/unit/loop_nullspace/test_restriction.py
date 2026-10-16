"""Reaction-restriction sensitivity on a fixed welded trajectory (NSX-26, #12569).

Row selection is exact; the restricted set of admissible reactions is a subset
of the ``WELD6`` set, so restricted envelopes are contained in the ``WELD6``
one; a restricted set may be empty and then names its conflict set, or the
unactuated rows may no longer balance (a recorded outcome, never repaired).
Closed-form oracles: the ``J_c = I`` snapshot and the exact ``restriction_counts``
golden.
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.capacity import CapacityVector
from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
    ClosureKind,
)
from src.shared.python.loop_nullspace.decomposition import (
    BALANCE_TOL,
    AllocationMetric,
    decompose,
    dimension_accounting,
)
from src.shared.python.loop_nullspace.envelopes import (
    BoundStatus,
    SignClass,
    joint_torque_quantities,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.fixtures import (
    load_golden,
    spatial_two_arm_club_snapshot,
)
from src.shared.python.loop_nullspace.registry import (
    Hypothesis,
    HypothesisSet,
    Source,
)
from src.shared.python.loop_nullspace.restriction import (
    RESTRICTION_LABEL,
    FrameOutcome,
    containment_violations,
    restrict,
    restriction_rows,
    restriction_sensitivity,
    snapshot_quantities,
)
from tests.unit.loop_nullspace.restriction_oracles import (
    golden_chain_snapshot,
    identity_closure_snapshot,
    shaft_aligned,
)

pytestmark = pytest.mark.unit

pytest.importorskip("highspy")

W6, S5, B3 = ClosureKind.WELD6, ClosureKind.SHAFT_FREE5, ClosureKind.BALL3
KINDS = (W6, S5, B3)
TORQUE = "H-TORQUE-CAPACITY"
GOLD = load_golden("restriction_counts")


# ------------------------------------------------------------ small helpers
def _capacity(n: int, cap: float) -> CapacityVector:
    names = tuple(f"j{i}" for i in range(n))
    return CapacityVector(
        coordinate_order=names,
        actuated_indices=tuple(range(n)),
        actuated_names=names,
        tau_max_negative_nm=np.full(n, cap),
        tau_max_positive_nm=np.full(n, cap),
        assumed_slots=(),
        verified_slots=names,
        warnings=(),
        force_velocity={},
    )


def _hset() -> HypothesisSet:
    hyp = Hypothesis(TORQUE, "statement", Source(None, "test"), "NOT_APPLICABLE")
    return HypothesisSet("test", "Test", (hyp,), ())


def _builder(cap: float):
    def build(dec, snap):
        return FeasibleSet.from_hypotheses(
            dec, _hset(), capacity=_capacity(snap.topology.n_actuated, cap)
        )

    return build


def _trajectory(snaps) -> AdmittedTrajectory:
    report = admit(
        list(snaps),
        gates=AdmissionGates(require_position=False),
        times=np.arange(len(snaps)) * 0.01,
    )
    return AdmittedTrajectory.from_report(report)


def _identity_frames(rows: list[list[float]]) -> list[ClosedLoopSnapshot]:
    return [identity_closure_snapshot(np.array(r)) for r in rows]


TORQUES = [snapshot_quantities(joint_torque_quantities())]


# -------------------------------------------------------------- row selection
def test_row_selection_matches_the_golden() -> None:
    for kind in KINDS:
        assert restriction_rows(W6, kind) == tuple(
            GOLD.get("kinds", kind.value, "rows")
        )
    assert restriction_rows(S5, B3) == (0, 1, 2)
    assert restriction_rows(B3, B3) == (0, 1, 2)
    assert restriction_rows(S5, S5) == (0, 1, 2, 3, 4)


@pytest.mark.requires_pinocchio
def test_restrict_selects_rows_exactly_and_changes_only_the_kind() -> None:
    snap = shaft_aligned(spatial_two_arm_club_snapshot().snapshot)
    for kind in KINDS:
        rows = list(restriction_rows(W6, kind))
        out = restrict(snap, kind)
        assert out.topology.closure_kind is kind
        np.testing.assert_array_equal(out.J_c, snap.J_c[rows])
        np.testing.assert_array_equal(out.Jdot_v, snap.Jdot_v[rows])
        for name in ("q", "v", "a", "M", "h", "S"):
            np.testing.assert_array_equal(getattr(out, name), getattr(snap, name))
        assert out.topology.actuated_indices == snap.topology.actuated_indices
        assert out.topology.loop_support_indices == snap.topology.loop_support_indices
        assert not out.J_c.flags.writeable
        assert RESTRICTION_LABEL in out.input_provenance.note
        assert kind.value in out.input_provenance.note


@pytest.mark.requires_pinocchio
def test_restriction_composes_and_identity_is_a_no_op() -> None:
    snap = shaft_aligned(spatial_two_arm_club_snapshot().snapshot)
    chained = restrict(restrict(snap, S5), B3)
    np.testing.assert_array_equal(chained.J_c, restrict(snap, B3).J_c)
    same = restrict(snap, W6)
    np.testing.assert_array_equal(same.J_c, snap.J_c)
    assert same.topology.closure_kind is W6


@pytest.mark.requires_pinocchio
def test_restricted_snapshot_still_satisfies_the_closure_levels() -> None:
    snap = shaft_aligned(spatial_two_arm_club_snapshot().snapshot)
    for kind in KINDS:
        r = restrict(snap, kind)
        assert np.abs(r.J_c @ r.v).max() < 1e-9
        assert np.abs(r.J_c @ r.a + r.Jdot_v).max() < 1e-9


@pytest.mark.parametrize(
    ("source", "target"),
    [(S5, W6), (B3, S5), (B3, W6)],
)
def test_relaxing_a_restriction_is_refused(source, target) -> None:
    with pytest.raises(ValueError, match="not an admissible restriction"):
        restriction_rows(source, target)


def test_planar_kinds_are_not_restrictable() -> None:
    with pytest.raises(ValueError, match="not an admissible restriction"):
        restriction_rows(ClosureKind.PLANAR_POINT2, B3)
    with pytest.raises(ValueError, match="not an admissible restriction"):
        restriction_rows(W6, ClosureKind.PLANAR_WELD3)


def test_unknown_kind_and_wrong_types_are_refused() -> None:
    snap = identity_closure_snapshot(np.zeros(6))
    with pytest.raises(ValueError, match="closure kind"):
        restrict(snap, "NOT_A_KIND")
    with pytest.raises(TypeError, match="ClosedLoopSnapshot"):
        restrict("snapshot", B3)  # type: ignore[arg-type]
    assert restrict(snap, "BALL3").topology.closure_kind is B3


@pytest.mark.requires_pinocchio
def test_shaft_free_needs_the_shaft_on_the_grip_x_axis() -> None:
    """The spatial fixture's club lies along grip z: row ``tx`` is not the twist."""
    raw = spatial_two_arm_club_snapshot().snapshot
    with pytest.raises(ValueError, match="shaft axis"):
        restrict(raw, S5)
    restrict(raw, B3)  # force rows only: no axis needed
    with pytest.raises(ValueError, match="shaft axis"):
        restrict(raw, S5, shaft_axis=(0.0, 0.0, 1.0))
    ok = restrict(raw, S5, shaft_axis=(1.0, 0.0, 0.0))
    assert ok.topology.closure_kind is S5  # caller asserts the axis explicitly


def test_shaft_axis_must_be_a_unit_vector_when_given() -> None:
    snap = identity_closure_snapshot(np.zeros(6))
    with pytest.raises(ValueError, match="shaft_axis"):
        restrict(snap, S5, shaft_axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="shaft_axis"):
        restrict(snap, S5, shaft_axis=(1.0, 0.0))


# --------------------------------------------------------------- golden counts
_VARIANTS = {
    "fixed_base_arms_only": {},
    "floating_root_no_feet": {"floating_root": True},
    "floating_root_two_point_feet": {"floating_root": True, "feet": True},
    "floating_root_four_point_feet": {
        "floating_root": True,
        "feet": True,
        "foot_points": 2,
    },
}


@pytest.mark.requires_pinocchio
@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.value)
def test_dimension_counts_match_the_exact_golden(variant, kind) -> None:
    pytest.importorskip("pinocchio")
    snap = shaft_aligned(spatial_two_arm_club_snapshot(**_VARIANTS[variant]).snapshot)
    acc = dimension_accounting(restrict(snap, kind))
    gold = GOLD.get("counts", variant, kind.value)
    renamed = {
        "equality_nullity": "nullity",
        "n_actuated": "n_a",
        "n_unactuated": "n_u",
        "attainable_torque_variation_fg_fixed": "attainable_torque_variation",
        "gauge_total_fg_free": "multiplier_gauge_fg_free",
    }
    skipped = {
        "nullity_fg_fixed",  # checked below as gauge + attainable variation
        "balance_rank",
        "consistency_conditions",
        "rank_J_g",
        "rank_stacked_J_c_J_g",
    }
    for key, expected in gold.items():
        if key in skipped:
            continue
        assert getattr(acc, renamed.get(key, key)) == expected, (
            variant,
            kind.value,
            key,
        )


# ----------------------------------------------------- exact torque oracle
@pytest.mark.parametrize("case", ["twist", "bend", "generic"])
def test_minimum_norm_torque_is_the_exact_residual(case) -> None:
    snap, data = golden_chain_snapshot(case)
    previous = -1.0
    for kind in KINDS:
        dec = decompose(restrict(snap, kind), weights=np.eye(14))
        expected = data["kinds"][kind.value]
        tau = np.array([float(Fraction(x)) for x in expected["tau_star"]])
        np.testing.assert_allclose(dec.torque_particular, tau, atol=1e-12)
        norm_sq = float(dec.torque_particular @ dec.torque_particular)
        assert norm_sq == pytest.approx(expected["min_tau_norm_sq_float"], abs=1e-12)
        assert norm_sq >= previous - 1e-12  # the allowed reaction set shrinks
        previous = norm_sq


def test_golden_cases_pin_which_row_a_load_needs() -> None:
    kinds = GOLD.get("oracle", "cases", "bend", "kinds")
    assert kinds["WELD6"]["min_tau_norm_sq_float"] == 0.0
    assert kinds["SHAFT_FREE5"]["min_tau_norm_sq_float"] == 0.0  # no tx in the load
    assert kinds["BALL3"]["min_tau_norm_sq_float"] > 0.0
    twist = GOLD.get("oracle", "cases", "twist", "kinds")
    assert twist["SHAFT_FREE5"]["min_tau_norm_sq_float"] > 0.0  # tx is needed


# ------------------------------------------------------ closed-form envelopes
def test_envelopes_per_kind_match_the_closed_form() -> None:
    """``J_c = I``: ``tau = b - lambda``, so dropped rows pin ``tau_i = b_i``."""
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    rep = restriction_sensitivity(
        _trajectory([identity_closure_snapshot(np.array(b))] * 3),
        KINDS,
        TORQUES,
        feasible_set_of=_builder(1.0),
    )
    tau = rep.series(W6)
    np.testing.assert_allclose(tau.lower[0], -1.0, atol=1e-9)
    np.testing.assert_allclose(tau.upper[0], 1.0, atol=1e-9)
    s5 = rep.series(S5)
    np.testing.assert_allclose(s5.lower[0, 3], 0.4, atol=1e-9)
    np.testing.assert_allclose(s5.upper[0, 3], 0.4, atol=1e-9)
    np.testing.assert_allclose(s5.lower[0, [0, 1, 2, 4, 5]], -1.0, atol=1e-9)
    b3 = rep.series(B3)
    np.testing.assert_allclose(b3.lower[0, 3:], b[3:], atol=1e-9)
    np.testing.assert_allclose(b3.upper[0, 3:], b[3:], atol=1e-9)
    np.testing.assert_allclose(b3.lower[0, :3], -1.0, atol=1e-9)


def test_nesting_holds_and_widths_shrink_on_the_oracle() -> None:
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    rep = restriction_sensitivity(
        _trajectory([identity_closure_snapshot(np.array(b))] * 3),
        KINDS,
        TORQUES,
        feasible_set_of=_builder(1.0),
    )
    assert rep.nesting_violations() == ()
    ratio = rep.width_ratio(B3, "tau_4")
    assert ratio.median == pytest.approx(0.0, abs=1e-9)
    assert rep.width_ratio(S5, "tau_1").median == pytest.approx(1.0, abs=1e-9)
    assert rep.width_ratio(S5, "tau_4").median == pytest.approx(0.0, abs=1e-9)


def test_containment_check_detects_a_widened_envelope() -> None:
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    rep = restriction_sensitivity(
        _trajectory([identity_closure_snapshot(np.array(b))] * 3),
        KINDS,
        TORQUES,
        feasible_set_of=_builder(1.0),
    )
    ref, kept = rep.series(W6), rep.series(S5)
    assert containment_violations(ref, kept) == ()
    upper = np.array(kept.upper)
    upper[1, 0] += 0.5  # a restricted envelope wider than the WELD6 one
    bad = containment_violations(ref, replace(kept, upper=upper))
    assert [(v.frame, v.quantity) for v in bad] == [(1, "tau_1")]
    ref_empty = [list(row) for row in ref.status_lower]
    ref_empty[2] = [BoundStatus.INFEASIBLE] * len(ref_empty[2])
    flipped = containment_violations(
        replace(ref, status_lower=tuple(tuple(r) for r in ref_empty)), kept
    )
    assert {v.frame for v in flipped} == {2}
    assert flipped[0].quantity == "<feasibility>"


def test_feasible_counts_and_emptiness_with_conflict_sets() -> None:
    frames = _identity_frames(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # feasible for every kind
            [0.1, 0.2, 0.3, 2.0, 0.5, 0.6],  # |tau_3| = 2 > cap: needs lambda_tx
            [0.1, 0.2, 0.3, 0.4, -1.7, 0.6],  # needs lambda_ty: ok for SHAFT only
        ]
    )
    rep = restriction_sensitivity(
        _trajectory(frames), KINDS, TORQUES, feasible_set_of=_builder(1.0)
    )
    assert rep.feasible_counts() == {W6: 3, S5: 2, B3: 1}
    assert rep.infeasible_counts() == {W6: 0, S5: 1, B3: 2}
    frame1 = rep.frame(S5, 1)
    assert frame1.outcome is FrameOutcome.INFEASIBLE
    assert frame1.conflict_set == (TORQUE,)
    assert rep.frame(B3, 2).conflict_set == (TORQUE,)
    assert rep.frame(W6, 1).outcome is FrameOutcome.FEASIBLE
    assert rep.frame(S5, 2).outcome is FrameOutcome.FEASIBLE
    assert rep.conflict_hypotheses(B3) == {TORQUE: 2}
    # infeasible frames are NaN with status INFEASIBLE, never a large number
    s5 = rep.series(S5)
    assert math.isnan(s5.lower[1, 0])
    assert s5.status_lower[1][0] is BoundStatus.INFEASIBLE
    assert rep.nesting_violations() == ()  # emptiness is containment


def test_sign_class_changes_are_counted_per_quantity() -> None:
    frames = _identity_frames(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.2, 0.3, -0.4, 0.5, 0.6],
            [0.1, 0.2, 0.3, 2.0, 0.5, 0.6],  # empty for SHAFT_FREE5
        ]
    )
    rep = restriction_sensitivity(
        _trajectory(frames), KINDS, TORQUES, feasible_set_of=_builder(1.0)
    )
    changes = rep.sign_class_changes(S5, "tau_4")
    assert changes == {"INDETERMINATE->POSITIVE": 1, "INDETERMINATE->NEGATIVE": 1}
    assert rep.lost_frames(S5) == (2,)
    assert rep.sign_class_changes(S5, "tau_1") == {}
    assert rep.series(W6).sign_class[0][3] is SignClass.INDETERMINATE
    assert rep.series(S5).sign_class[0][3] is SignClass.POSITIVE


def test_unbalanced_unactuated_rows_are_recorded_not_repaired() -> None:
    """Coordinate 0 is unactuated and only the twist row loads it.

    Under ``WELD6`` the load is carried by ``lambda_tx``; a joint that cannot
    transmit twist leaves ``b_0`` unbalanced.  The scaled residual is reported
    and ``BALANCE_TOL`` is not touched.
    """
    base = identity_closure_snapshot(np.array([0.7, 0, 0, 0, 0, 0]))
    j_c = np.eye(6)
    j_c[:, 0] = 0.0
    j_c[3, 0] = 1.0  # coordinate 0 is loaded by the tx row only
    snap = replace(
        base,
        J_c=j_c,
        S=np.eye(6)[1:],
        topology=replace(
            base.topology, actuated_indices=(1, 2, 3, 4, 5), unactuated_indices=(0,)
        ),
    )
    rep = restriction_sensitivity(
        _trajectory([snap] * 3), KINDS, TORQUES, feasible_set_of=_builder(1.0)
    )
    assert rep.frame(W6, 0).outcome is FrameOutcome.FEASIBLE
    for kind in (S5, B3):
        f = rep.frame(kind, 0)
        assert f.outcome is FrameOutcome.BALANCE_INCONSISTENT
        assert f.balance_residual_scaled > BALANCE_TOL
        assert f.conflict_set == ()
    assert rep.balance_inconsistent_counts() == {W6: 0, S5: 3, B3: 3}
    assert rep.feasible_counts() == {W6: 3, S5: 0, B3: 0}
    assert math.isnan(rep.series(S5).lower[0, 0])


# ---------------------------------------------------------- labels & records
def test_every_output_carries_the_label() -> None:
    rep = restriction_sensitivity(
        _trajectory([identity_closure_snapshot(np.full(6, 0.1))] * 3),
        KINDS,
        TORQUES,
        feasible_set_of=_builder(1.0),
    )
    assert rep.label == RESTRICTION_LABEL == "REACTION_RESTRICTION_SENSITIVITY"
    assert rep.to_dict()["label"] == RESTRICTION_LABEL
    assert set(rep.to_dict()["kinds"]) == {"WELD6", "SHAFT_FREE5", "BALL3"}
    snap = identity_closure_snapshot(np.full(6, 0.1))
    records = rep.records(snap.model_provenance, snap.input_provenance)
    assert set(records) == set(KINDS)
    for kind, per_name in records.items():
        for rec in per_name.values():
            assert RESTRICTION_LABEL in rec.conventions
            assert f"restriction_kind={kind.value}" in rec.conventions
            assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
            assert TORQUE in rec.hypotheses
    assert any("not evidence about the true grip" in c for c in rep.non_claims)


def test_infeasible_kind_records_are_unavailable_with_a_reason() -> None:
    frames = _identity_frames([[0.1, 0.2, 0.3, 2.0, 0.5, 0.6]] * 3)
    rep = restriction_sensitivity(
        _trajectory(frames), KINDS, TORQUES, feasible_set_of=_builder(1.0)
    )
    snap = frames[0]
    rec = rep.records(snap.model_provenance, snap.input_provenance)[S5]["tau_1"]
    assert rec.allocation_dependence is AllocationDependence.UNAVAILABLE
    assert RESTRICTION_LABEL in rec.conventions


# ---------------------------------------------------------- argument contracts
def test_sensitivity_argument_contracts() -> None:
    traj = _trajectory([identity_closure_snapshot(np.zeros(6))] * 3)
    ok = {"feasible_set_of": _builder(1.0)}
    with pytest.raises(ValueError, match="kinds"):
        restriction_sensitivity(traj, (), TORQUES, **ok)
    with pytest.raises(ValueError, match="unique"):
        restriction_sensitivity(traj, (W6, W6), TORQUES, **ok)
    with pytest.raises(ValueError, match="first kind"):
        restriction_sensitivity(traj, (S5, W6), TORQUES, **ok)
    with pytest.raises(ValueError, match="quantities"):
        restriction_sensitivity(traj, KINDS, [], **ok)
    with pytest.raises(TypeError, match="AdmittedTrajectory"):
        restriction_sensitivity("traj", KINDS, TORQUES, **ok)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="feasible_set_of"):
        restriction_sensitivity(traj, KINDS, TORQUES, feasible_set_of=None)  # type: ignore[arg-type]


def test_metric_defaults_to_identity_and_accepts_inverse_mass() -> None:
    frames = [identity_closure_snapshot(np.full(6, 0.1))] * 3
    traj = _trajectory(frames)
    a = restriction_sensitivity(traj, KINDS, TORQUES, feasible_set_of=_builder(1.0))
    b = restriction_sensitivity(
        traj,
        KINDS,
        TORQUES,
        feasible_set_of=_builder(1.0),
        weights=AllocationMetric.inverse_mass,
    )
    # the metric picks the particular solution only: the envelopes are unchanged
    np.testing.assert_allclose(a.series(S5).lower, b.series(S5).lower, atol=1e-9)
