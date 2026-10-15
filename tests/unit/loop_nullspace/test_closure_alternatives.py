"""True closure alternatives with kinematics re-derived per grip model (NSX-72, #12590).

Each grip model is its own closure: it keeps its own rows, so it admits a
different set of motions (NSX-26 only restricted reaction rows of the welded
kinematics).  The tests pin, per model, admission or honest refusal of the same
trajectory with the model's own position, velocity and acceleration residuals;
the exact NSX-00/NSX-26 golden counts; the independent finite-difference check
of the rows against the model's own pose residual; and that on a trajectory the
weld also admits, the rows-only models give exactly the NSX-26 restriction.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("highspy")

from src.shared.python.loop_nullspace.admission import (  # noqa: E402
    AdmissionGates,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.closure_alternatives import (  # noqa: E402
    GRIP_MODEL_SPECS,
    GripModel,
    Motion,
    SpatialGripPlant,
    admission_outcome,
    admit_model,
    motion_frames,
    position_probe,
)
from src.shared.python.loop_nullspace.closure_alternatives_report import (  # noqa: E402
    GRIP_MODEL_LABEL,
    GripModelResult,
    box_feasible_set_of,
    closure_component_quantities,
    evaluate_grip_model,
    median_width_ratio,
)
from src.shared.python.loop_nullspace.envelopes import (  # noqa: E402
    joint_torque_quantities,
)
from src.shared.python.loop_nullspace.restriction import (  # noqa: E402
    restriction_sensitivity,
    snapshot_quantities,
)
from src.shared.python.loop_nullspace.contracts import ClosureKind  # noqa: E402
from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    dimension_accounting,
)
from src.shared.python.loop_nullspace.fixtures import load_golden  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

W, S, TB, LB = (
    GripModel.WELD,
    GripModel.SHAFT_TWIST_REVOLUTE,
    GripModel.TRAIL_BALL,
    GripModel.LEAD_BALL,
)
ALL = (W, S, TB, LB)
GOLD = load_golden("restriction_counts")
GATES = AdmissionGates()  # the NSX-00 gate 1e-6, never loosened


def _plant(model: GripModel, floating_root: bool = False) -> SpatialGripPlant:
    return SpatialGripPlant(model, floating_root=floating_root)


# ------------------------------------------------------------------ specs
def test_specs_state_rows_extra_coordinates_and_limitations() -> None:
    assert GRIP_MODEL_SPECS[W].closure_kind is ClosureKind.WELD6
    assert GRIP_MODEL_SPECS[S].closure_kind is ClosureKind.SHAFT_FREE5
    assert GRIP_MODEL_SPECS[TB].closure_kind is ClosureKind.BALL3
    assert GRIP_MODEL_SPECS[LB].closure_kind is ClosureKind.WELD6
    assert [GRIP_MODEL_SPECS[m].extra_coordinates for m in ALL] == [0, 0, 0, 3]
    for spec in GRIP_MODEL_SPECS.values():
        assert spec.summary and spec.limitation
        assert "measure" in spec.limitation.lower() or spec.limitation


@pytest.mark.parametrize("model", ALL, ids=lambda m: m.value)
def test_snapshot_has_the_models_own_rows_and_shaft_first_frame(model) -> None:
    plant = _plant(model)
    spec = GRIP_MODEL_SPECS[model]
    v, a = plant.motion_state(Motion.WELDED, np.random.default_rng(1))
    snap = plant.snapshot(plant.q_star, v, a)
    assert snap.topology.closure_kind is spec.closure_kind
    assert snap.J_c.shape == (spec.closure_kind.rows, plant.n)
    assert plant.n == 14 + spec.extra_coordinates
    club = snap.bare_club.com
    assert np.linalg.norm(club[1:]) < 1e-12 and club[0] > 0  # shaft is grip x


@pytest.mark.parametrize("model", ALL, ids=lambda m: m.value)
def test_rows_match_the_models_own_pose_residual_by_finite_difference(model) -> None:
    """Independent route: central difference of the model's residual, not J_c."""
    plant = _plant(model)
    eps = 1e-6
    cols = []
    for j in range(plant.n):
        step = np.zeros(plant.n)
        step[j] = eps
        plus = plant.closure_pose_residual(plant.integrate(plant.q_star, step))
        minus = plant.closure_pose_residual(plant.integrate(plant.q_star, -step))
        cols.append((plus - minus) / (2 * eps))
    fd = np.array(cols).T
    rows = plant.closure_rate_matrix(plant.q_star)
    assert (
        fd.shape == rows.shape == (GRIP_MODEL_SPECS[model].closure_kind.rows, plant.n)
    )
    assert np.max(np.abs(fd - rows)) < 1e-6
    assert np.max(np.abs(plant.closure_pose_residual(plant.q_star))) < 1e-12


def test_rows_only_models_are_row_selections_of_the_weld_rows() -> None:
    weld = _plant(W).closure_rate_matrix(_plant(W).q_star)
    assert np.array_equal(
        _plant(S).closure_rate_matrix(_plant(S).q_star), weld[[0, 1, 2, 4, 5]]
    )
    assert np.array_equal(_plant(TB).closure_rate_matrix(_plant(TB).q_star), weld[:3])


# ---------------------------------------------------------- admission matrix
# model -> motion -> admitted?  The weld is the strictest; the rows-only models
# admit more motion; the lead-hand ball (own coordinates) is judged on the
# welded-coordinate motion embedded with a locked ball.
EXPECTED = {
    Motion.WELDED: {W: True, S: True, TB: True, LB: True},
    Motion.TWIST: {W: False, S: True, TB: True, LB: False},
    Motion.TILT: {W: False, S: False, TB: True, LB: False},
}


@pytest.mark.parametrize("motion", list(EXPECTED), ids=lambda m: m.value)
@pytest.mark.parametrize("model", ALL, ids=lambda m: m.value)
def test_same_trajectory_is_admitted_or_refused_per_model(motion, model) -> None:
    plant = _plant(model)
    snaps, times = motion_frames(plant, motion, n_frames=5, seed=7)
    out = admission_outcome(plant, snaps, times, GATES)
    assert out.admitted is EXPECTED[motion][model], (motion, model, out.reasons)
    # residuals are reported at all three levels with the model's own rows
    assert out.max_phi < GATES.phi_tol
    if out.admitted:
        assert out.max_phi_dot < GATES.phi_dot_tol
        assert out.max_phi_ddot < GATES.phi_ddot_tol
        assert out.reasons == ()
    else:
        assert max(out.max_phi_dot, out.max_phi_ddot) > 1e-3  # not a near miss
        assert out.reasons


def test_refusal_names_the_dropped_row_the_weld_still_constrains() -> None:
    plant = _plant(W)
    snaps, times = motion_frames(plant, Motion.TWIST, n_frames=4, seed=3)
    out = admission_outcome(plant, snaps, times, GATES)
    assert not out.admitted
    assert all("velocity" in r or "acceleration" in r for r in out.reasons)


@pytest.mark.parametrize("model", ALL, ids=lambda m: m.value)
def test_own_motion_is_admitted_by_its_own_model(model) -> None:
    plant = _plant(model)
    snaps, times = motion_frames(plant, Motion.OWN, n_frames=5, seed=11)
    out = admission_outcome(plant, snaps, times, GATES)
    assert out.admitted, out.reasons
    assert out.max_phi_dot < GATES.phi_dot_tol


def test_lead_ball_own_motion_moves_the_ball_which_no_other_model_can_represent() -> (
    None
):
    plant = _plant(LB)
    snaps, _ = motion_frames(plant, Motion.OWN, n_frames=3, seed=11)
    assert all(np.max(np.abs(s.v[-3:])) > 1e-3 for s in snaps)
    for other in (W, S, TB):
        with pytest.raises(ValueError, match="not representable"):
            motion_frames(
                _plant(other), Motion.OWN, n_frames=3, seed=11, source=_plant(LB)
            )


# ------------------------------------------------------------ position level
@pytest.mark.parametrize(
    ("motion", "model", "admitted"),
    [
        (Motion.TWIST, W, False),
        (Motion.TWIST, S, True),
        (Motion.TWIST, TB, True),
        (Motion.TILT, S, False),
        (Motion.TILT, TB, True),
    ],
)
def test_position_probe_uses_the_models_own_pose_residual(
    motion, model, admitted
) -> None:
    plant = _plant(model)
    snaps, times = position_probe(plant, motion, steps=(1e-4, 1.5e-4, 2e-4))
    out = admission_outcome(plant, snaps, times, GATES)
    assert out.admitted is admitted, (out.max_phi, out.reasons)
    if not admitted:
        assert out.max_phi > GATES.phi_tol
        assert any("position" in r for r in out.reasons)
    else:  # the kept rows close to second order in the step
        assert out.max_phi < GATES.phi_tol


# ------------------------------------------------------- null-space dimensions
@pytest.mark.parametrize("kind", ["WELD6", "SHAFT_FREE5", "BALL3"])
def test_rows_only_dimension_counts_match_the_exact_nsx26_golden(kind) -> None:
    model = {"WELD6": W, "SHAFT_FREE5": S, "BALL3": TB}[kind]
    variants = {"fixed_base_arms_only": False, "floating_root_no_feet": True}
    for variant, floating in variants.items():
        plant = _plant(model, floating)
        v, a = plant.motion_state(Motion.WELDED, np.random.default_rng(5))
        acc = dimension_accounting(plant.snapshot(plant.q_star, v, a))
        gold = GOLD.get("counts", variant, kind)
        assert acc.nullity == gold["equality_nullity"], (variant, kind)
        assert acc.multiplier_gauge == gold["multiplier_gauge"]
        assert acc.rank_J_c == gold["rank_J_c"]
        assert acc.rank_J_cu_T == gold["rank_J_cu_T"]


def test_kinematic_admitted_motion_dimension_grows_as_rows_drop() -> None:
    dims = {m: _plant(m).kinematic_nullity() for m in ALL}
    assert dims == {W: 14 - 6, S: 14 - 5, TB: 14 - 3, LB: 17 - 6}


def test_lead_ball_counts_follow_the_structure_of_its_unactuated_rows() -> None:
    """Structural (generic rank), not an exact golden: n=17, n_u=3, m=6."""
    plant = _plant(LB)
    v, a = plant.motion_state(Motion.OWN, np.random.default_rng(2))
    acc = dimension_accounting(plant.snapshot(plant.q_star, v, a))
    assert (acc.n, acc.n_a, acc.n_u, acc.m) == (17, 14, 3, 6)
    assert acc.rank_J_c == 6 and acc.rank_J_cu_T == 3
    assert acc.multiplier_gauge == 0 and acc.attainable_torque_variation == 3
    assert (
        acc.nullity == 3
    )  # dim_x 20 - rank_A (14 + 3): the 3 ball rows fix 3 of 6 lambdas


# ----------------------------------------------------------------- contracts
def test_contracts_refuse_bad_inputs() -> None:
    with pytest.raises(TypeError, match="GripModel"):
        SpatialGripPlant("WELD")  # type: ignore[arg-type]
    plant = _plant(W)
    with pytest.raises(ValueError, match="n_frames"):
        motion_frames(plant, Motion.WELDED, n_frames=2, seed=1)
    with pytest.raises(ValueError, match="steps"):
        position_probe(plant, Motion.TWIST, steps=(1e-4,))
    with pytest.raises(ValueError, match="positive"):
        position_probe(plant, Motion.TWIST, steps=(1e-4, -1e-4, 2e-4))
    with pytest.raises(TypeError, match="Motion"):
        motion_frames(plant, "TWIST", n_frames=3, seed=1)  # type: ignore[arg-type]


# ------------------------------------------------------- dynamics and envelopes
#: Smallest uniform torque box (N m) that holds every frame of the seed-7,
#: 20-frame WELDED motion: weld 5.75, shaft-twist 6.10, trail ball 7.64,
#: lead ball 7.46 (a min-max LP per frame, run once to choose the caps below).
TIGHT, LOOSE = 6.0, 12.0


def _result(model, motion, cap, *, n_frames=20) -> GripModelResult:
    plant = _plant(model)
    snaps, times = motion_frames(plant, motion, n_frames=n_frames, seed=7)
    adm = admission_outcome(plant, snaps, times, GATES)
    builders = [
        snapshot_quantities(joint_torque_quantities(None)),
        closure_component_quantities(GRIP_MODEL_SPECS[model].components),
    ]
    return evaluate_grip_model(
        adm, quantities=builders, feasible_set_of=box_feasible_set_of(cap)
    )


def test_welded_trajectory_dynamics_equal_the_nsx26_restriction() -> None:
    """On a motion the weld admits, a rows-only model is exactly NSX-26's restriction."""
    plant = _plant(W)
    snaps, times = motion_frames(plant, Motion.WELDED, n_frames=20, seed=7)
    traj = AdmittedTrajectory.from_report(
        admit(snaps, gates=GATES, times=times, plant=plant)
    )
    kinds = (ClosureKind.WELD6, ClosureKind.SHAFT_FREE5, ClosureKind.BALL3)
    ref = restriction_sensitivity(
        traj,
        kinds,
        [snapshot_quantities(joint_torque_quantities(None))],
        feasible_set_of=box_feasible_set_of(TIGHT),
    )
    for model, kind in ((W, kinds[0]), (S, kinds[1]), (TB, kinds[2])):
        own = _result(model, Motion.WELDED, TIGHT)
        assert own.counts() == {
            "feasible": ref.feasible_counts()[kind],
            "infeasible": ref.infeasible_counts()[kind],
            "balance_inconsistent": ref.balance_inconsistent_counts()[kind],
        }
        mine = own.widths("tau_")
        theirs = ref.series(kind).width()
        assert np.allclose(mine, theirs, equal_nan=True, rtol=1e-9, atol=1e-9)


def test_a_tight_torque_box_separates_the_models_on_the_same_trajectory() -> None:
    counts = {m: _result(m, Motion.WELDED, TIGHT).counts() for m in ALL}
    assert counts[W]["infeasible"] == 0 and counts[W]["feasible"] == 20
    assert counts[S]["infeasible"] >= 1  # needs 6.10 N m somewhere, box is 6.0
    assert counts[TB]["infeasible"] >= counts[S]["infeasible"]
    assert counts[LB]["infeasible"] >= 1  # a ball at the lead hand needs more
    for m in ALL:
        assert counts[m]["balance_inconsistent"] == 0
        assert sum(counts[m].values()) == 20


def test_a_loose_box_makes_every_model_feasible_and_envelopes_nest_for_rows_models() -> (
    None
):
    res = {m: _result(m, Motion.WELDED, LOOSE) for m in ALL}
    assert all(r.counts()["feasible"] == 20 for r in res.values())
    for model in (S, TB):  # restricting rows shrinks the torque envelope
        ratio = median_width_ratio(res[W], res[model], "tau_")
        assert ratio is not None and ratio <= 1.0 + 1e-9
    assert res[W].allocation_nullity == (6, 6)
    assert res[S].allocation_nullity == (5, 5)
    assert res[TB].allocation_nullity == (3, 3)
    assert res[LB].allocation_nullity == (3, 3)


def test_refused_trajectory_has_no_dynamics_and_says_why() -> None:
    res = _result(W, Motion.TWIST, LOOSE)
    assert not res.admission.admitted and res.report is None
    assert res.counts() == {"feasible": 0, "infeasible": 0, "balance_inconsistent": 0}
    assert res.allocation_nullity is None and res.median_width("tau_") is None
    assert res.to_dict()["admitted"] is False and res.to_dict()["reasons"]


def test_result_record_is_labelled_and_json_serialisable() -> None:
    record = _result(S, Motion.TWIST, LOOSE).to_dict()
    assert record["label"] == GRIP_MODEL_LABEL
    assert record["closure_kind"] == "SHAFT_FREE5" and record["admitted"] is True
    assert record["kinematic_nullity"] == [9, 9]
    assert json.loads(json.dumps(record))["model"] == "SHAFT_TWIST_REVOLUTE"


def test_floating_root_without_feet_is_balance_inconsistent() -> None:
    """A floating root with no ground contact cannot balance (recorded, not repaired)."""
    plant = SpatialGripPlant(W, floating_root=True)
    snaps, times = motion_frames(plant, Motion.WELDED, n_frames=3, seed=7)
    adm = admit_model(W, snaps, times, GATES, plant=plant)
    assert adm.admitted
    res = evaluate_grip_model(
        adm,
        quantities=[snapshot_quantities(joint_torque_quantities(None))],
        feasible_set_of=box_feasible_set_of(LOOSE),
    )
    assert res.counts()["balance_inconsistent"] == 3


def test_report_contracts() -> None:
    with pytest.raises(ValueError, match="cap_nm"):
        box_feasible_set_of(0.0)
    plant = _plant(W)
    snaps, times = motion_frames(plant, Motion.WELDED, n_frames=3, seed=7)
    adm = admission_outcome(plant, snaps, times, GATES)
    with pytest.raises(ValueError, match="at least one"):
        evaluate_grip_model(
            adm, quantities=[], feasible_set_of=box_feasible_set_of(8.0)
        )
    with pytest.raises(ValueError, match="component names"):
        evaluate_grip_model(
            adm,
            quantities=[closure_component_quantities(("fx", "fy"))],
            feasible_set_of=box_feasible_set_of(LOOSE),
        )
