"""Back-squat same-input bundle: round trip, IK residuals, MuJoCo replay (#11745).

The IK thresholds below are the residuals measured when the committed bundle
was generated (``scripts/generate_lift_bundle.py``), rounded up to 0.1 mm;
they are recorded evidence, not tuning targets.  Since OSL-14 (#13057) the
bundle replays the AN-1 lift body under its cited ranges: the hip stops at its
120 deg limit near the bottom, where the optimizer's planar trunk-thigh fold
asks for more, so the knee, hip and shoulder landmarks trail their sagittal
targets by up to 2.2 cm there (the Rajagopal legs are also mildly asymmetric).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.lifting.bundles import (
    BACK_SQUAT_BUNDLE,
    LIFT_SPEC,
    build_lift_bundle,
    lift_to_full_body,
    load_seed_pose,
)
from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SagittalSolution,
)
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = pytest.mark.unit

# Measured maxima over all 151 frames (metres); see module docstring.  The
# widened entries (golf body: knee_r 0.0110, hip_l 0.0149, shoulders 0.0098,
# penetration 5.4e-4) and their physical reasons are recorded on #13057
# (issuecomment-6229479012); the #13072 re-track fix does not change them.
# OSL-16 (#13104): the committed source is now the bar-balanced IK (#13093),
# whose more upright trunk moves the right ankle to 3.59 mm and the right hip
# to 9.13 mm (were 3.08 and 6.32 mm), so those two bounds are widened with the
# evidence on #13104; every other landmark fell (ankle_l 4.37, knee_l 3.15,
# knee_r 11.68, hip_l 8.07, shoulder_l 3.65, shoulder_r 3.04 mm, bar 0.01 mm)
# and keeps its earlier bound.
IK_RESIDUAL_THRESHOLDS_M = {
    "ankle_l": 0.0057,
    "ankle_r": 0.0036,
    "knee_l": 0.0152,
    "knee_r": 0.0225,
    "hip_l": 0.0178,
    "hip_r": 0.0092,
    "shoulder_l": 0.0196,
    "shoulder_r": 0.0151,
    "bar": 0.0001,
}
CLOSURE_THRESHOLD_M = 6.4e-8
FOOT_PENETRATION_THRESHOLD_M = 7.0e-4
#: Committed bar-loaded bundle size bound (bytes).  OSL-16 (#13104): a loaded
#: bundle records its 0.25 ms hold efforts (``hold_efforts``, 4 per step) so
#: a replay reproduces it exactly; at 200 steps that table is 281,728 raw
#: bytes (about 233 KB compressed, full-precision torques).  The earlier
#: 300,000 byte bound plus that raw table, rounded up; evidence on #13104.
LOADED_BUNDLE_MAX_BYTES = 600_000


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return InputBundle.load(BACK_SQUAT_BUNDLE)


def test_committed_bundle_is_small(bundle: InputBundle) -> None:
    assert BACK_SQUAT_BUNDLE.stat().st_size <= LOADED_BUNDLE_MAX_BYTES
    assert bundle.reference_engine == "mujoco"


def test_bundle_round_trips(bundle: InputBundle, tmp_path: Path) -> None:
    path = tmp_path / "again.npz"
    bundle.save(path)
    again = InputBundle.load(path)
    assert again.manifest() == bundle.manifest()
    assert again.spec_bytes == bundle.spec_bytes
    for name in ("q0", "v0", "efforts", "reference_q", "reference_v"):
        np.testing.assert_array_equal(getattr(again, name), getattr(bundle, name))


def test_bundle_carries_provenance(bundle: InputBundle) -> None:
    prov = bundle.provenance
    assert prov["lift"] == "back_squat"
    assert len(prov["source_commit"]) == 40
    optimizer = prov["optimizer"]
    assert optimizer["package"] == "movement_optimizer"
    assert optimizer["exercise"] == "full_squat"
    assert optimizer["success"] is True
    for key in ("version", "body_mass_kg", "height_m", "bar_mass_kg", "n_eval"):
        assert key in optimizer
    ik = prov["ik"]
    assert set(ik["max_residual_m"]) == set(LANDMARK_LABELS)
    assert prov["spec_sha256"] == bundle.spec_sha256
    if "back_bar" in ik:
        from src.shared.python.lifting.back_bar import apply_grip_width

        width = ik["back_bar"]["geometry"]["grip_width_m"]
        derived = apply_grip_width(LIFT_SPEC.read_bytes(), width)
        if "balance" in ik:
            # #13093: the balanced source's dynamics carry the bar load.
            from src.shared.python.lifting.barbell_runs import rigid_bar_spec
            from src.shared.python.lifting.loaded_balance import (
                load_balance_from_record,
            )

            load = load_balance_from_record(ik["balance"]["load"])
            derived = rigid_bar_spec(derived, load.barbell)
        assert bundle.spec_bytes == derived
    else:
        assert bundle.spec_bytes == LIFT_SPEC.read_bytes()


def test_recorded_ik_residuals_are_below_thresholds(bundle: InputBundle) -> None:
    ik = bundle.provenance["ik"]
    for label, limit in IK_RESIDUAL_THRESHOLDS_M.items():
        assert ik["max_residual_m"][label] <= limit, label
    assert ik["max_closure_m"] <= CLOSURE_THRESHOLD_M
    assert ik["min_sphere_height_m"] >= -FOOT_PENETRATION_THRESHOLD_M


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_ik_from_the_recorded_solution_meets_thresholds(
    bundle: InputBundle,
) -> None:
    pytest.importorskip("mujoco")
    solution = SagittalSolution.from_record(bundle.provenance["sagittal_solution"])
    geom_rec = bundle.provenance["ik"].get("back_bar", {}).get("geometry")
    geometry = None
    if geom_rec is not None:
        from src.shared.python.lifting.back_bar import BackBarGeometry
        from src.shared.python.lifting.back_rack import RACKS

        rack_rec = geom_rec.get("rack")
        geometry = BackBarGeometry.from_standard(
            geom_rec["lift"],
            height_m=geom_rec["height_m"],
            rack=None if rack_rec is None else RACKS[rack_rec["name"]],
        )
    balance = bundle.provenance["ik"].get("balance")
    load = None
    if balance is not None:
        from src.shared.python.lifting.loaded_balance import load_balance_from_record

        load = load_balance_from_record(balance["load"])
    # The generator solves the back bar from the lift-body spec (it derives
    # the grip-width spec itself); a legacy bundle carries that spec.
    body = bundle.spec_bytes if geometry is None else LIFT_SPEC.read_bytes()
    lift = lift_to_full_body(
        body, solution, load_seed_pose(), geometry=geometry, load=load
    )
    if balance is not None:
        assert lift.balance is not None
        np.testing.assert_allclose(
            lift.balance["correction_rad"], balance["correction_rad"], atol=1e-9
        )
    residual = lift.max_residual_by_landmark()
    for label, limit in IK_RESIDUAL_THRESHOLDS_M.items():
        assert residual[label] <= limit, (label, residual[label])
    assert float(lift.closure_m.max()) <= CLOSURE_THRESHOLD_M
    assert float(lift.lowest_sphere_m.min()) >= -FOOT_PENETRATION_THRESHOLD_M


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_mujoco_replay_reproduces_the_bar_path(bundle: InputBundle) -> None:
    """Open-loop MuJoCo replay of the first 10 steps is bit exact (smoke).

    Ten steps keep the test inside the 60 s budget on a loaded runner; the
    full-window cross-engine bar-path parity is the next LIFT-5 slice.
    """
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.bundles import BAR_FRAME, bar_path
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
    frame = bundle.provenance.get("bar_frame", BAR_FRAME)
    np.testing.assert_allclose(
        bar_path(plant, rollout.q, frame=frame),
        recorded[: steps + 1],
        rtol=0,
        atol=1e-9,
    )


def test_build_rejects_a_window_outside_the_lift() -> None:
    solution = SagittalSolution(
        np.linspace(0.0, 1.0, 5), np.zeros((5, 3)), np.zeros((5, 2)), {}
    )
    with pytest.raises(ValueError, match="window"):
        build_lift_bundle(
            b"{}",
            solution,
            None,
            lift_name="back_squat",
            window_start_s=0.9,
            window_s=0.5,
        )
    with pytest.raises(ValueError, match="window_s"):
        build_lift_bundle(
            b"{}",
            solution,
            None,
            lift_name="back_squat",
            window_start_s=0.0,
            window_s=0.0,
        )


def test_lift_to_full_body_rejects_a_bad_seed() -> None:
    solution = SagittalSolution(
        np.linspace(0.0, 1.0, 5), np.zeros((5, 3)), np.zeros((5, 2)), {}
    )
    with pytest.raises(ValueError, match="seed_q"):
        lift_to_full_body(LIFT_SPEC.read_bytes(), solution, np.zeros(3))
    with pytest.raises(TypeError, match="spec_bytes"):
        lift_to_full_body("not bytes", solution, np.zeros(41))  # type: ignore[arg-type]
