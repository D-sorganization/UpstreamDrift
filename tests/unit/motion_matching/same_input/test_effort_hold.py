"""Effort-hold rate of same-input bundles (#12279, blocks GCV-20 #11767).

The reference export evaluates the computed-torque controller once per
1 ms step and holds it over the step's eight RK4 substeps.  For a stiff
tracking law that zero-order hold, not the integrator, lets the export drift
from the swing it tracks.  These engine-free tests pin the opt-in finer hold:
it is a validated policy parameter, it is recorded in the bundle manifest,
and every replay reads it from there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import (
    ROOT_COORDINATES,
    InputBundle,
    StepPolicy,
    integrate,
    open_loop,
    segmented_replay,
)
from src.shared.python.motion_matching.same_input.integrator import (
    DEFAULT_EFFORT_HOLDS,
    effort_holds_for,
)
from src.shared.python.motion_matching.same_input.scoring import (
    COORDINATE_BOUND_RAD,
)

pytestmark = pytest.mark.unit

ORDER = (*ROOT_COORDINATES, "Joint")
SPEC = json.dumps({"coordinate_order": list(ORDER)}).encode()
DT_S = 1e-3  # the bundle step (pipeline DT_S)
STEPS = 200
# Computed-torque tracking of q_ref = A sin(W t) on a = -q + tau.  Under a
# continuously evaluated law the tracking error obeys e'' + KD e' + KP e = 0
# from rest, so the tracked swing is exactly q_ref; any departure is the hold.
AMPLITUDE, W = 0.5, 20.0
# Closed-loop bandwidth (critically damped), rad/s.  The 7-iron export's
# stiff modes are the foot contacts (integrator: eigenvalues near -1e4 1/s);
# 1500 rad/s puts this toy loop past what a 1 ms hold can follow while a
# 0.125 ms hold (|omega h| ~ 0.19) tracks it.
OMEGA_C = 1500.0
KP, KD = OMEGA_C**2, 2.0 * OMEGA_C
# The existing epic #11605 L2 coordinate bound, unchanged.
TRACK_TOLERANCE_RAD = COORDINATE_BOUND_RAD


class _Oscillator:
    """Unit oscillators a = -q + tau; one trivially satisfied closure (the
    unforced first root coordinate stays at zero), so the per-step closure
    projection is well posed; the tip frame follows the joint."""

    coordinate_order = ORDER

    def acceleration(self, q, v, tau):
        return -q + tau

    def closure_pose_residual(self, q):
        return q[:1].copy()

    def closure_rate_matrix(self, q):
        rate = np.zeros((1, q.size))
        rate[0, 0] = 1.0
        return rate

    def kinematic_frames(self, q):
        pose = np.eye(4)
        pose[0, 3] = q[-1]
        return {"tip": pose}


def _reference(t: float) -> tuple[float, float, float]:
    s, c = np.sin(W * t), np.cos(W * t)
    return AMPLITUDE * s, AMPLITUDE * W * c, -AMPLITUDE * W * W * s


def _controller(_j: int, t: float, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_ref, v_ref, a_ref = _reference(t)
    tau = np.zeros_like(q)
    tau[-1] = q[-1] + a_ref + KP * (q_ref - q[-1]) + KD * (v_ref - v[-1])
    return tau


def _track(policy: StepPolicy, controller=_controller):
    q0, v0 = np.zeros(7), np.zeros(7)
    v0[-1] = AMPLITUDE * W
    return integrate(
        _Oscillator(), q0, v0, controller, steps=STEPS, dt_s=DT_S, policy=policy
    )


def _tracking_error(rollout) -> float:
    q_ref = np.array([_reference(t)[0] for t in rollout.time_s])
    return float(np.abs(rollout.q[:, -1] - q_ref).max())


def _bundle(holds: int) -> InputBundle:
    rollout = _track(StepPolicy(project=False, effort_holds=holds))
    return InputBundle(
        spec_bytes=SPEC,
        coordinate_order=ORDER,
        dt_s=DT_S,
        q0=rollout.q[0],
        v0=rollout.v[0],
        efforts=rollout.efforts[::holds],
        reference_q=rollout.q,
        reference_v=rollout.v,
        reference_engine="toy",
        effort_holds=holds,
        hold_efforts=None if holds == 1 else rollout.efforts,
    )


# -- the defect and the fix ---------------------------------------------------
def test_fine_hold_tracks_the_swing_and_the_1ms_hold_does_not() -> None:
    fine = _track(StepPolicy(project=False, effort_holds=8))
    coarse = _track(StepPolicy(project=False))
    assert _tracking_error(fine) <= TRACK_TOLERANCE_RAD
    assert _tracking_error(coarse) > TRACK_TOLERANCE_RAD


def test_hold_error_shrinks_with_the_hold_period() -> None:
    """Below the stability limit the hold error is first order in the hold."""
    soft = 300.0  # rad/s

    def controller(_j, t, q, v):
        q_ref, v_ref, a_ref = _reference(t)
        tau = np.zeros_like(q)
        tau[-1] = q[-1] + a_ref + soft**2 * (q_ref - q[-1]) + 2 * soft * (v_ref - v[-1])
        return tau

    errors = [
        _tracking_error(_track(StepPolicy(project=False, effort_holds=h), controller))
        for h in (1, 2, 4, 8)
    ]
    assert all(b < a for a, b in zip(errors, errors[1:], strict=False))
    assert errors[0] / errors[-1] > 5.0


def test_held_efforts_are_recorded_at_the_hold_rate() -> None:
    rollout = _track(StepPolicy(project=False, effort_holds=8))
    assert rollout.q.shape == (STEPS + 1, 7)
    assert rollout.efforts.shape == (STEPS * 8, 7)
    # each step's first sample is the law at the step's start state and time
    for k in (0, 3, STEPS - 1):
        start = _controller(0, k * DT_S, rollout.q[k], rollout.v[k])
        np.testing.assert_array_equal(rollout.efforts[k * 8], start)
    # the later samples are re-evaluated inside the step, not copies
    assert not np.array_equal(rollout.efforts[24], rollout.efforts[31])


def test_open_loop_replay_of_fine_hold_efforts_is_bit_exact() -> None:
    policy = StepPolicy(project=False, effort_holds=8)
    tracked = _track(policy)
    replayed = open_loop(
        _Oscillator(),
        tracked.q[0],
        tracked.v[0],
        tracked.efforts,
        dt_s=DT_S,
        policy=policy,
    )
    np.testing.assert_array_equal(replayed.q, tracked.q)
    np.testing.assert_array_equal(replayed.v, tracked.v)


def test_open_loop_rejects_efforts_not_at_the_hold_rate() -> None:
    tracked = _track(StepPolicy(project=False))
    with pytest.raises(ValueError, match="effort_holds"):
        open_loop(
            _Oscillator(),
            tracked.q[0],
            tracked.v[0],
            tracked.efforts[:-1],
            dt_s=DT_S,
            policy=StepPolicy(project=False, effort_holds=8),
        )


# -- the default contract is unchanged ---------------------------------------
def test_default_hold_is_one_per_step() -> None:
    assert DEFAULT_EFFORT_HOLDS == 1
    assert StepPolicy().effort_holds == 1
    assert StepPolicy().substeps_per_hold == 8


def test_one_hold_per_step_reproduces_the_original_integrator() -> None:
    explicit = _track(StepPolicy(project=False, effort_holds=1))
    default = _track(StepPolicy(project=False))
    np.testing.assert_array_equal(explicit.q, default.q)
    np.testing.assert_array_equal(explicit.efforts, default.efforts)


# -- DbC of the parameter -----------------------------------------------------
@pytest.mark.parametrize("holds", [0, -1])
def test_step_policy_rejects_nonpositive_holds(holds: int) -> None:
    with pytest.raises(ValueError, match="effort_holds"):
        StepPolicy(effort_holds=holds)


@pytest.mark.parametrize("holds", [1.0, True, "8"])
def test_step_policy_rejects_non_integer_holds(holds: object) -> None:
    with pytest.raises(TypeError, match="effort_holds"):
        StepPolicy(effort_holds=holds)  # type: ignore[arg-type]


def test_step_policy_requires_holds_to_divide_substeps() -> None:
    with pytest.raises(ValueError, match="divide"):
        StepPolicy(substeps=8, effort_holds=3)


def test_effort_holds_for_converts_a_hold_period() -> None:
    assert effort_holds_for(1e-3, 1e-3) == 1
    assert effort_holds_for(0.125e-3, 1e-3) == 8
    assert effort_holds_for(0.25e-3, 1e-3) == 4


@pytest.mark.parametrize("hold_s", [0.3e-3, 2e-3, 0.0, -1e-3, float("nan")])
def test_effort_holds_for_rejects_a_hold_that_does_not_split_the_step(
    hold_s: float,
) -> None:
    with pytest.raises(ValueError):
        effort_holds_for(hold_s, 1e-3)


def test_effort_holds_for_rejects_a_hold_finer_than_a_substep() -> None:
    with pytest.raises(ValueError, match="divide"):
        effort_holds_for(1e-3 / 16, 1e-3)


# -- the manifest carries the hold and every replay reads it -----------------
def test_manifest_records_the_hold(tmp_path: Path) -> None:
    bundle = _bundle(8)
    policy = bundle.manifest()["policy"]
    assert policy["effort_holds_per_step"] == 8
    assert policy["effort_hold_s"] == pytest.approx(DT_S / 8)
    assert policy["substeps"] == 8
    path = tmp_path / "fine.npz"
    bundle.save(path)
    loaded = InputBundle.load(path)
    assert loaded.effort_holds == 8
    np.testing.assert_array_equal(loaded.replay_efforts(), bundle.replay_efforts())
    assert loaded.step_policy().effort_holds == 8


def test_default_manifest_records_one_hold_and_legacy_bundles_load(
    tmp_path: Path,
) -> None:
    bundle = _bundle(1)
    assert bundle.manifest()["policy"]["effort_holds_per_step"] == 1
    assert bundle.manifest()["policy"]["effort_hold_s"] == pytest.approx(DT_S)
    path = tmp_path / "legacy.npz"
    bundle.save(path)
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    manifest = json.loads(str(arrays["manifest"]))
    for key in ("effort_holds_per_step", "effort_hold_s"):
        del manifest["policy"][key]
    arrays["manifest"] = np.array(json.dumps(manifest, sort_keys=True))
    np.savez_compressed(path, **arrays)
    loaded = InputBundle.load(path)
    assert loaded.effort_holds == 1
    np.testing.assert_array_equal(loaded.replay_efforts(), bundle.efforts)


def test_bundle_step_efforts_stay_one_row_per_step() -> None:
    bundle = _bundle(8)
    assert bundle.efforts.shape == (STEPS, 7)
    assert bundle.replay_efforts().shape == (STEPS * 8, 7)
    assert bundle.replay_efforts(steps=5).shape == (5 * 8, 7)


def test_bundle_rejects_inconsistent_hold_efforts() -> None:
    bundle = _bundle(8)
    fields = {
        "spec_bytes": SPEC,
        "coordinate_order": ORDER,
        "dt_s": DT_S,
        "q0": bundle.q0,
        "v0": bundle.v0,
        "efforts": bundle.efforts,
        "reference_q": bundle.reference_q,
        "reference_v": bundle.reference_v,
        "reference_engine": "toy",
    }
    with pytest.raises(ValueError, match="hold_efforts"):
        InputBundle(**fields, effort_holds=8)  # missing hold table
    with pytest.raises(ValueError, match="hold_efforts"):
        InputBundle(**fields, effort_holds=8, hold_efforts=bundle.replay_efforts()[1:])
    shifted = bundle.replay_efforts().copy()
    shifted[0, -1] += 1.0
    with pytest.raises(ValueError, match="hold_efforts"):
        InputBundle(**fields, effort_holds=8, hold_efforts=shifted)
    with pytest.raises(ValueError, match="divide"):
        InputBundle(**fields, effort_holds=3, hold_efforts=bundle.replay_efforts())


def test_segmented_replay_reads_the_hold_from_the_bundle() -> None:
    bundle = _bundle(8)
    segments = segmented_replay(_Oscillator(), bundle, segment_steps=50)
    assert max(s["coordinate_error_rad"] for s in segments) == 0.0
