"""The bar-loaded squat keeps its feet on the IK (OSL-16, #13104).

Tracking the bar-balanced squat IK with the 100 kg bar under the original
contract (eight 0.125 ms RK4 substeps, 1 ms effort hold) lost the feet: the
mtp, ankle and subtalar ran away from the IK.  The stiffest open-loop
foot-contact mode gave |lambda h| above 2.785 from the first steps, outside
RK4's real stability limit 2.785, and the mode reaches 8.95e4 1/s at the
squat bottom.  Bar-loaded lifts now track at 64 substeps with a 0.25 ms hold
(``barbell_runs.LOADED_TRACKING_POLICY``), |lambda h| = 1.40, inside RK4's
maximum-damping point 1.596.

The bounds are fixed, not tuned: each side's foot points, in its tibia frame
(``loaded_foot.foot_point_deviation_m``), stay within that side's LIFT-5 ankle
IK residual threshold for the whole window, and the result converges as the
step shrinks (RK4 substeps at a fixed hold; the hold at fixed substeps).
The record is ``scripts/measure_loaded_foot_tracking.py``'s sweep.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.lifting.barbell_runs import LOADED_TRACKING_POLICY
from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE
from src.shared.python.lifting.loaded_foot import LOADED_FOOT_RECORD
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.motion_matching.same_input.stiffness import (
    RK4_MAX_DAMPING_Z,
    RK4_REAL_STABILITY_LIMIT,
)

pytestmark = pytest.mark.unit

#: |lambda h| target of the substep rule (stiffness.DEFAULT_TARGET).
STABILITY_TARGET = RK4_MAX_DAMPING_Z
#: Substep convergence: 64 -> 128 substeps at the committed hold moves no
#: foot coordinate by more than this (rad) -- the integration error is
#: resolved.
SUBSTEP_CONVERGENCE_RAD = 1e-3


def _ankle_bounds() -> dict[str, float]:
    from test_lift_bundles import IK_RESIDUAL_THRESHOLDS_M

    return {
        "r": IK_RESIDUAL_THRESHOLDS_M["ankle_r"],
        "l": IK_RESIDUAL_THRESHOLDS_M["ankle_l"],
    }


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return InputBundle.load(BACK_SQUAT_BUNDLE)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(LOADED_FOOT_RECORD.read_text(encoding="utf-8"))


def _run(record: dict, substeps: int, hold_s: float) -> dict:
    for run in record["runs"]:
        if run["substeps"] == substeps and run["effort_hold_s"] == pytest.approx(
            hold_s, rel=1e-9
        ):
            return run
    raise AssertionError(f"no run at {substeps} substeps, {hold_s} s hold")


def _committed(record: dict) -> dict:
    hold = 1e-3 / LOADED_TRACKING_POLICY.effort_holds
    return _run(record, LOADED_TRACKING_POLICY.substeps, hold)


def test_record_is_for_the_committed_bundle(record, bundle) -> None:
    assert record["bundle_spec_sha256"] == bundle.spec_sha256
    assert record["bundle_policy"] == bundle.manifest()["policy"]
    assert record["window"] == bundle.provenance["window"]
    assert len(record["source_commit"]) == 40


def test_committed_bundle_tracks_with_the_loaded_policy(bundle) -> None:
    assert "balance" in bundle.provenance["ik"]
    policy = bundle.step_policy()
    assert policy.substeps == LOADED_TRACKING_POLICY.substeps
    assert policy.effort_holds == LOADED_TRACKING_POLICY.effort_holds
    stability = bundle.provenance["rk4_stability"]
    assert stability["substep_s"] == pytest.approx(bundle.dt_s / policy.substeps)
    assert stability["max_abs_z"] <= STABILITY_TARGET


def test_original_contract_is_outside_the_rk4_region(record) -> None:
    """The red evidence: eight substeps lose the loaded feet."""
    run = _run(record, 8, 1e-3)
    assert run["rk4_stability"]["max_abs_z"] > RK4_REAL_STABILITY_LIMIT
    bounds = _ankle_bounds()
    assert any(run["foot_point_deviation_m"][s] > bounds[s] for s in bounds)


def test_loaded_feet_stay_within_the_lift5_ankle_bounds(record) -> None:
    run = _committed(record)
    assert run["failure"] is None
    assert run["reached_steps"] == round(record["window"]["duration_s"] / 1e-3)
    assert run["rk4_stability"]["max_abs_z"] <= STABILITY_TARGET
    for side, limit in _ankle_bounds().items():
        assert run["foot_point_deviation_m"][side] <= limit, side


def test_foot_tracking_converges_in_the_rk4_substep(record) -> None:
    coarse, fine = _run(record, 64, 2.5e-4), _run(record, 128, 2.5e-4)
    for joint, error in coarse["foot_joint_error_rad"].items():
        assert abs(error - fine["foot_joint_error_rad"][joint]) <= (
            SUBSTEP_CONVERGENCE_RAD
        ), joint


def test_foot_tracking_converges_in_the_effort_hold(record) -> None:
    """At 64 substeps the foot error falls monotonically as the hold halves."""
    holds = (1e-3, 5e-4, 2.5e-4)
    deviation = [
        max(_run(record, 64, hold)["foot_point_deviation_m"].values()) for hold in holds
    ]
    assert deviation == sorted(deviation, reverse=True)
    mtp = [
        max(
            v
            for k, v in _run(record, 64, hold)["foot_joint_error_rad"].items()
            if k.startswith("mtp")
        )
        for hold in holds
    ]
    assert mtp == sorted(mtp, reverse=True)


@pytest.mark.integration
@pytest.mark.requires_mujoco
@pytest.mark.slow
def test_committed_reference_foot_deviation_matches_the_record(record, bundle) -> None:
    """The record's committed-policy run is the committed bundle's tracking."""
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.loaded_foot import (
        balanced_ik_window,
        foot_point_deviation_m,
        window_reference,
    )

    times, track, _ = balanced_ik_window(bundle)
    run_time = np.arange(bundle.steps + 1) * bundle.dt_s
    ref = window_reference(times, track, run_time)
    measured = foot_point_deviation_m(
        bundle.spec_bytes, bundle.coordinate_order, bundle.reference_q, ref
    )
    expected = _committed(record)["foot_point_deviation_m"]
    assert measured == pytest.approx(expected, rel=0, abs=1e-9)
