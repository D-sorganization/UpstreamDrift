"""Deadlift, clean-and-jerk and snatch same-input bundles (LIFT-5, #11745).

All three are pulls built by the back-squat pipeline (``scripts/generate_lift_bundle
.py --lift ...``): a Movement Optimizer solve, full-body marker IK with the bar
held in the hands at the standard's grip width, and a 0.2 s MuJoCo tracking
window centred on the knee pass.  The IK thresholds are the residuals measured
when the committed bundles were generated, rounded up to 0.1 mm; they are
recorded evidence, not tuning targets.

The snatch and the clean and jerk solve the whole lift, so their thresholds
are whole-lift maxima set by phases after the window (the catches, the rack
and the jerk).  Whether the full-body IK can follow those phases is a
separate strict xfail against the unchanged deadlift bounds for each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.lifting.bundles import (
    LIFT_SPEC,
    lift_hands_to_full_body,
    load_seed_pose,
)
from src.shared.python.lifting.lift_recipes import knee_pass_time, recipe
from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SagittalSolution,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

PULLS = ("deadlift", "clean_and_jerk", "snatch")
MAX_BUNDLE_BYTES = 300_000
# Measured maxima over all 151 frames (metres); see the module docstring.
# #13037 (whole-row de Leva lift body plus the 10 deg tibial torsion of
# #12814): the ankle, knee and shoulder snapshots are re-derived from the
# regenerated bundles with the same rounding; evidence on #13041
# (issuecomment-6243170509).
IK_RESIDUAL_THRESHOLDS_M: dict[str, dict[str, float]] = {
    "deadlift": {
        "ankle_l": 0.0038,
        "ankle_r": 0.0032,
        "knee_l": 0.0035,
        "knee_r": 0.0110,
        "hip_l": 0.0084,
        "hip_r": 0.0060,
        "shoulder_l": 0.0042,
        "shoulder_r": 0.0046,
        "bar": 0.0029,
    },
    # Whole-lift maxima of the full clean and jerk (Tools #5662), set by the
    # front-squat catch, the jerk and the overhead phases (after the window):
    # see _CLEAN_AND_JERK_RACK_AND_JERK below.  The first-pull-only bundle's
    # snapshot (ankle_l 3.8 ... bar 1.2 mm) is replaced because the solution
    # now covers the whole lift; measurements on #11745.
    "clean_and_jerk": {
        "ankle_l": 0.0716,
        "ankle_r": 0.0744,
        "knee_l": 0.0698,
        "knee_r": 0.0664,
        "hip_l": 0.0566,
        "hip_r": 0.0577,
        "shoulder_l": 0.0273,
        "shoulder_r": 0.0297,
        "bar": 0.0565,
    },
    # Whole-lift maxima, set by the turnover and the catch (0.50-0.77 of the
    # lift, after the window): see _SNATCH_CATCH below.  #13037: shoulder_r
    # 11.16 -> 15.09 mm at the turnover, re-derived with the same rounding; the
    # other maxima fell (#13041, issuecomment-6243170509).
    "snatch": {
        "ankle_l": 0.0473,
        "ankle_r": 0.0502,
        "knee_l": 0.0455,
        "knee_r": 0.0431,
        "hip_l": 0.0471,
        "hip_r": 0.0473,
        "shoulder_l": 0.0198,
        "shoulder_r": 0.0151,
        "bar": 0.0402,
    },
}
# OSL-14 (#13057): the bounded arm solve ends with the hanging arm at its
# cited elbow-extension limit, so the grip weld closes to tens of microns
# (the unbounded golf-body solve closed to 1e-7 m with out-of-range joints).
# Before/after numbers: #13057 (issuecomment-6229479012); re-derived on the
# #13037 lift body (8.94e-5 and 3.29e-5 m): #13041 (issuecomment-6243170509).
CLOSURE_THRESHOLD_M: dict[str, float] = {
    "deadlift": 9.0e-5,
    "clean_and_jerk": 3.9e-4,  # 3.80e-4 m, whole lift (#11745)
    "snatch": 2.4e-4,  # 1.75e-4 -> 2.35e-4 m on the #13037 lift body
}
FOOT_PENETRATION_THRESHOLD_M: dict[str, float] = {
    "deadlift": 7.0e-4,
    "clean_and_jerk": 2.6e-3,  # 2.55 mm, whole lift (#11745)
    "snatch": 2.1e-3,
}
_CLEAN_AND_JERK_RACK_AND_JERK = pytest.mark.xfail(
    strict=True,
    reason=(
        "Measured clean-and-jerk IK (#11745): the setup, the first pull "
        "(0-0.20 of the lift) and the 0.693-0.893 s window are deadlift class "
        "(worst knee_r 11.1 mm, bar 1.15 mm, feet 0.69 mm) and so is the "
        "overhead recovery (0.85-1.0: hip_l 8.0 mm, bar 0.78 mm), but the "
        "full-body model cannot follow the turnover (legs 31-40 mm, bar "
        "39.1 mm at 0.20-0.30), the planar front-squat catch (legs 49-74 mm, "
        "feet 2.55 mm into the ground at 0.30-0.55), the front rack (bar "
        "56 mm, shoulders 27-30 mm at 0.30-0.70: the planar rack bar sits at "
        "the shoulder joints, which the hands cannot reach at the 0.6 m grip) "
        "or the power-jerk catch (legs 19-29 mm at 0.70-0.85).  The deadlift "
        "bounds are not loosened."
    ),
)
_SNATCH_CATCH = pytest.mark.xfail(
    strict=True,
    reason=(
        "Measured snatch IK on the #13037 lift body (#11745, #13041): through "
        "the pulls (0-0.45 of the lift) and the 0.467-0.667 s window the "
        "residuals are deadlift class (worst knee_r 11.3 mm, bar 0.23 mm, feet "
        "0.69 mm), but the full-body model cannot follow the turnover (bar "
        "19.5 mm, shoulder_r 15.1 mm at 0.50) or the planar overhead-squat "
        "catch (legs 34-45 mm, feet 1.88 mm into the ground at 0.53-0.77).  "
        "The deadlift bounds are not loosened."
    ),
)


@pytest.fixture(scope="module", params=PULLS)
def lift(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def bundle(lift: str) -> InputBundle:
    return InputBundle.load(recipe(lift).bundle_path)


def test_committed_bundle_is_small(lift: str, bundle: InputBundle) -> None:
    assert recipe(lift).bundle_path.stat().st_size <= MAX_BUNDLE_BYTES
    assert bundle.reference_engine == "mujoco"
    assert bundle.steps == 200


def test_bundle_round_trips(bundle: InputBundle, tmp_path: Path) -> None:
    path = tmp_path / "again.npz"
    bundle.save(path)
    again = InputBundle.load(path)
    assert again.manifest() == bundle.manifest()
    assert again.spec_bytes == bundle.spec_bytes
    for name in ("q0", "v0", "efforts", "reference_q", "reference_v"):
        np.testing.assert_array_equal(getattr(again, name), getattr(bundle, name))


def test_bundle_carries_provenance(lift: str, bundle: InputBundle) -> None:
    from src.shared.python.lifting.back_bar import apply_grip_width
    from src.shared.python.lifting.lift_spec import lift_grip

    pull = recipe(lift)
    prov = bundle.provenance
    assert prov["lift"] == pull.bundle_name
    assert len(prov["source_commit"]) == 40
    assert len(prov["tools_pin"]) == 40
    optimizer = prov["optimizer"]
    assert optimizer["package"] == "movement_optimizer"
    assert optimizer["exercise"] == pull.exercise
    assert optimizer["bar_convention"] == pull.bar_convention
    assert optimizer["duration_s"] == pull.duration_s
    assert optimizer["success"] is True
    assert prov["recipe"]["phase"] == pull.phase
    assert prov["bar_frame"] == "Grip"
    hand_bar = prov["ik"]["hand_bar"]
    assert hand_bar["lift"] == pull.lift
    width = lift_grip(pull.lift).width_m
    assert hand_bar["grip_width_m"] == width
    assert bundle.spec_bytes == apply_grip_width(LIFT_SPEC.read_bytes(), width)
    assert prov["spec_sha256"] == bundle.spec_sha256
    assert "Hand-held bar" in prov["limitations"]


def test_window_is_centred_on_the_knee_pass(bundle: InputBundle) -> None:
    prov = bundle.provenance
    solution = SagittalSolution.from_record(prov["sagittal_solution"])
    centre = knee_pass_time(solution)
    start = prov["window"]["start_s"]
    assert start <= centre <= start + prov["window"]["duration_s"]
    step = float(np.diff(solution.time_s).max())
    assert abs(start + prov["window"]["duration_s"] / 2 - centre) <= step


def test_recorded_ik_residuals_are_below_thresholds(
    lift: str, bundle: InputBundle
) -> None:
    ik = bundle.provenance["ik"]
    assert set(ik["max_residual_m"]) == set(LANDMARK_LABELS)
    for label, limit in IK_RESIDUAL_THRESHOLDS_M[lift].items():
        assert ik["max_residual_m"][label] <= limit, label
    assert set(IK_RESIDUAL_THRESHOLDS_M[lift]) == set(LANDMARK_LABELS)
    assert ik["max_closure_m"] <= CLOSURE_THRESHOLD_M[lift]
    assert ik["min_sphere_height_m"] >= -FOOT_PENETRATION_THRESHOLD_M[lift]


def test_window_bar_hangs_below_the_shoulders(bundle: InputBundle) -> None:
    """The recorded bar path stays under the shoulder girdle (a pull)."""
    bar = np.asarray(bundle.provenance["window_bar_path_m"], dtype=float)
    assert bar.shape == (bundle.steps + 1, 3)
    from src.shared.python.motion_matching.same_input import VectorPlant

    pytest.importorskip("mujoco")
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    hub = plant.kinematic_frames(bundle.reference_q[0])["Hub"][:3, 3]
    assert bar[0, 2] < hub[2] - 0.4


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_ik_from_the_recorded_solution_meets_thresholds(
    lift: str, bundle: InputBundle
) -> None:
    pytest.importorskip("mujoco")
    solution = SagittalSolution.from_record(bundle.provenance["sagittal_solution"])
    full = lift_hands_to_full_body(
        LIFT_SPEC.read_bytes(), solution, load_seed_pose(), recipe(lift).lift
    )
    assert full.back_bar is not None
    assert full.back_bar.spec_bytes == bundle.spec_bytes
    residual = full.max_residual_by_landmark()
    for label, limit in IK_RESIDUAL_THRESHOLDS_M[lift].items():
        assert residual[label] <= limit, (label, residual[label])
    assert float(full.closure_m.max()) <= CLOSURE_THRESHOLD_M[lift]
    assert float(full.lowest_sphere_m.min()) >= -FOOT_PENETRATION_THRESHOLD_M[lift]


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_mujoco_replay_reproduces_the_bar_path(bundle: InputBundle) -> None:
    """Open-loop MuJoCo replay of the first 10 steps is bit exact (smoke)."""
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.bundles import bar_path
    from src.shared.python.motion_matching.same_input import VectorPlant, open_loop

    steps = 10
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    rollout = open_loop(
        plant,
        bundle.q0,
        bundle.v0,
        bundle.replay_efforts(steps),
        dt_s=bundle.dt_s,
        policy=bundle.step_policy(),
    )
    np.testing.assert_array_equal(rollout.q, bundle.reference_q[: steps + 1])
    recorded = np.asarray(bundle.provenance["window_bar_path_m"], dtype=float)
    np.testing.assert_allclose(
        bar_path(plant, rollout.q, frame=bundle.provenance["bar_frame"]),
        recorded[: steps + 1],
        rtol=0,
        atol=1e-9,
    )


def test_lift_hands_rejects_an_unknown_lift() -> None:
    solution = SagittalSolution(
        np.linspace(0.0, 1.0, 5),
        np.zeros((5, 3)),
        np.zeros((5, 2)),
        {"segment_lengths_m": [0.44, 0.44, 0.51]},
    )
    with pytest.raises(ValueError, match="lift"):
        lift_hands_to_full_body(
            LIFT_SPEC.read_bytes(), solution, load_seed_pose(), "curl"
        )


@_SNATCH_CATCH
def test_snatch_whole_lift_ik_meets_the_deadlift_bounds() -> None:
    ik = InputBundle.load(recipe("snatch").bundle_path).provenance["ik"]
    for label, limit in IK_RESIDUAL_THRESHOLDS_M["deadlift"].items():
        assert ik["max_residual_m"][label] <= limit, label
    assert ik["min_sphere_height_m"] >= -FOOT_PENETRATION_THRESHOLD_M["deadlift"]


def test_clean_and_jerk_solution_racks_then_locks_the_bar_overhead() -> None:
    """The recorded solution is the whole lift with the phase-correct bar.

    Heights are the optimizer's own (ankle origin, ``up``): the bar hangs one
    arm length below the shoulder at the setup, sits on it (front rack) at the
    standard's rack catch and is one arm length above it at the lockout.
    """
    from src.shared.python.lifting.lift_spec import lift_phases

    prov = InputBundle.load(recipe("clean_and_jerk").bundle_path).provenance
    solution = SagittalSolution.from_record(prov["sagittal_solution"])
    lengths = np.asarray(solution.source["segment_lengths_m"], dtype=float)
    arm = float(solution.source["arm_length_m"])
    shoulder = np.cos(solution.segment_angles_rad) @ lengths
    frac = solution.time_s / solution.time_s[-1]
    phases = {p.name: p.t_frac for p in lift_phases("clean_and_jerk")}
    catch = int(np.argmin(np.abs(frac - phases["rack_catch"])))
    bar = solution.bar_m[:, 1]
    assert bar[0] == pytest.approx(shoulder[0] - arm)
    assert bar[catch] == pytest.approx(shoulder[catch])
    assert bar[-1] == pytest.approx(shoulder[-1] + arm)
    overhead = frac >= phases["overhead_catch"]
    np.testing.assert_allclose(bar[overhead], shoulder[overhead] + arm, atol=1e-12)


@_CLEAN_AND_JERK_RACK_AND_JERK
def test_clean_and_jerk_whole_lift_ik_meets_the_deadlift_bounds() -> None:
    ik = InputBundle.load(recipe("clean_and_jerk").bundle_path).provenance["ik"]
    for label, limit in IK_RESIDUAL_THRESHOLDS_M["deadlift"].items():
        assert ik["max_residual_m"][label] <= limit, label
    assert ik["min_sphere_height_m"] >= -FOOT_PENETRATION_THRESHOLD_M["deadlift"]
