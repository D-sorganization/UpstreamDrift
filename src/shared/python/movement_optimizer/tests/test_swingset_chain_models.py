# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for swingset and segmented-chain analysis models."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from movement_optimizer.models.chain_dynamics import (
    ChainConfig,
    ChainState,
    initial_catenary_angles,
    initial_tip_kick_velocities,
    link_midpoints,
    random_wadded_chain_state,
    simulate_chain,
    simulate_chain_for_duration,
    step_chain,
    steps_for_duration,
    total_energy,
)
from movement_optimizer.models.swingset import (
    MAX_OPTIMIZER_BUDGET,
    SWING_TORSO_LEAN_LIMITS_RAD,
    CyclicPolicyBounds,
    CyclicPolicyParameters,
    CyclicPolicySearchSpace,
    HumanSegmentSpec,
    SwingControlAction,
    SwingPose,
    SwingSetConfig,
    SwingSetState,
    build_swingset_snapshot,
    constrain_swing_pose,
    cyclic_policy_controls,
    cyclic_pumping_policy,
    estimate_swingset_joint_torques,
    heuristic_pumping_policy,
    optimize_cyclic_policy,
    optimize_cyclic_policy_iterative,
    simulate_swingset,
    step_swingset,
)


def test_chain_positions_include_anchor_and_tip() -> None:
    config = ChainConfig(segment_count=4, segment_length_m=0.25)
    state = ChainState.stationary(config)

    positions = state.node_positions(config)

    assert positions.shape == (5, 2)
    assert positions[-1, 1] == pytest.approx(1.0)
    assert state.metrics(config).tip_speed_m_s == pytest.approx(0.0)


def test_chain_config_validates_physical_inputs() -> None:
    with pytest.raises(ValueError, match="segment_count"):
        ChainConfig(segment_count=0)
    with pytest.raises(ValueError, match="segment_length_m"):
        ChainConfig(segment_length_m=0.0)
    with pytest.raises(ValueError, match="damping"):
        ChainConfig(damping=-0.1)
    with pytest.raises(ValueError, match="coupling"):
        ChainConfig(coupling=-0.1)
    with pytest.raises(ValueError, match="bend_damping"):
        ChainConfig(bend_damping=-0.1)


def test_chain_state_validation_rejects_bad_arrays() -> None:
    config = ChainConfig(segment_count=2)
    with pytest.raises(ValueError, match="angles_rad"):
        ChainState(np.zeros((2, 1)), np.zeros(2)).validated(config)
    with pytest.raises(ValueError, match="angular_velocities"):
        ChainState(np.zeros(2), np.zeros(3)).validated(config)
    with pytest.raises(ValueError, match="finite"):
        ChainState(np.asarray([0.0, np.nan]), np.zeros(2)).validated(config)


def test_chain_midpoints_validate_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        link_midpoints(np.zeros((1, 2)))


def test_chain_catenary_and_anchor_validate_inputs() -> None:
    config = ChainConfig(segment_count=2)
    state = ChainState.stationary(config)
    with pytest.raises(ValueError, match="at least 2"):
        initial_catenary_angles(1, 0.2)
    with pytest.raises(ValueError, match="non-negative"):
        initial_catenary_angles(2, -0.2)
    with pytest.raises(ValueError, match="two finite"):
        state.node_positions(config, anchor_xy_m=(0.0, float("nan")))


def test_chain_simulation_damps_energy() -> None:
    config = ChainConfig(segment_count=6, damping=0.25)
    initial = ChainState(
        initial_catenary_angles(config.segment_count, 0.25),
        np.full(config.segment_count, 0.15),
    )

    rollout = simulate_chain(config, initial, steps=24, dt_s=0.01)

    assert len(rollout.states) == 25
    assert rollout.frame_count() == 25
    assert rollout.positions.shape == (25, 7, 2)
    assert np.all(np.isfinite(rollout.energy_j))
    assert total_energy(config, rollout.states[-1]) == pytest.approx(
        rollout.energy_j[-1]
    )
    diffs = np.diff(rollout.positions, axis=1)
    link_lengths = np.sqrt(
        np.einsum("...i,...i->...", diffs, diffs)
    )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is faster than np.linalg.norm(..., axis=2)
    np.testing.assert_allclose(link_lengths, config.segment_length_m)


def test_chain_bend_damping_reduces_wadded_chain_tip_speed() -> None:
    base = ChainConfig(segment_count=8, damping=0.0015, bend_damping=0.0)
    damped = ChainConfig(segment_count=8, damping=0.0015, bend_damping=0.005)
    initial = ChainState(
        np.asarray([0.9, -1.1, 1.4, -0.8, 1.2, -1.3, 0.7, -0.4], dtype=np.float64),
        np.full(8, 1.8, dtype=np.float64),
    )

    base_rollout = simulate_chain(base, initial, steps=80, dt_s=0.01)
    damped_rollout = simulate_chain(damped, initial, steps=80, dt_s=0.01)

    assert damped_rollout.tip_speed_m_s[-1] < base_rollout.tip_speed_m_s[-1]
    assert damped_rollout.energy_j[-1] < base_rollout.energy_j[-1]


def test_chain_single_segment_gravity_matches_slender_rod_pendulum() -> None:
    """One chain link behaves like a physical rod pivoted at the anchor."""
    config = ChainConfig(
        segment_count=1,
        segment_length_m=0.5,
        damping=0.0,
        coupling=0.0,
        bend_damping=0.0,
    )
    angle = 0.2
    dt_s = 1e-4
    state = ChainState(
        np.asarray([angle], dtype=np.float64), np.zeros(1, dtype=np.float64)
    )

    stepped = step_chain(config, state, dt_s=dt_s)

    expected_acceleration = -(
        3.0 * config.gravity_m_s2 / (2.0 * config.segment_length_m)
    ) * np.sin(angle)
    observed_acceleration = stepped.angular_velocities_rad_s[0] / dt_s
    assert observed_acceleration == pytest.approx(expected_acceleration, rel=0.02)


def test_chain_tip_kick_velocities_increase_toward_tip() -> None:
    velocities = initial_tip_kick_velocities(5, 2.0)

    assert velocities[0] == pytest.approx(0.0)
    assert velocities[-1] == pytest.approx(2.0)
    assert np.all(np.diff(velocities) > 0.0)


def test_chain_random_wadded_start_is_deterministic_and_validated() -> None:
    config = ChainConfig(segment_count=5)

    first = random_wadded_chain_state(
        config, angle_span_rad=np.pi, velocity_span_rad_s=0.4, seed=7
    )
    second = random_wadded_chain_state(
        config, angle_span_rad=np.pi, velocity_span_rad_s=0.4, seed=7
    )

    np.testing.assert_allclose(first.angles_rad, second.angles_rad)
    np.testing.assert_allclose(
        first.angular_velocities_rad_s, second.angular_velocities_rad_s
    )
    assert first.angles_rad.shape == (5,)
    assert np.max(np.abs(first.angles_rad)) <= np.pi
    with pytest.raises(ValueError, match="angle_span_rad"):
        random_wadded_chain_state(config, angle_span_rad=-0.1)
    with pytest.raises(ValueError, match="velocity_span_rad_s"):
        random_wadded_chain_state(config, velocity_span_rad_s=-0.1)


def test_chain_duration_simulation_uses_requested_time() -> None:
    config = ChainConfig(segment_count=3)
    initial = ChainState.stationary(config, angle_rad=0.25)

    assert steps_for_duration(1.0, 0.25) == 4
    rollout = simulate_chain_for_duration(config, initial, duration_s=1.0, dt_s=0.25)

    assert len(rollout.states) == 5
    with pytest.raises(ValueError, match="duration_s"):
        steps_for_duration(0.0, 0.01)


def test_chain_simulation_validates_rollout_inputs() -> None:
    config = ChainConfig(segment_count=3)
    initial = ChainState.stationary(config)
    with pytest.raises(ValueError, match="steps"):
        simulate_chain(config, initial, steps=0, dt_s=0.01)
    with pytest.raises(ValueError, match="dt_s"):
        step_chain(config, initial, dt_s=0.0)
    with pytest.raises(ValueError, match="incompatible"):
        simulate_chain(
            config, initial, steps=2, dt_s=0.01, torque_history_nm=np.zeros((2, 2))
        )


def test_swingset_snapshot_models_body_chain_and_mass() -> None:
    config = SwingSetConfig(chain_segments=8, seat_placement_thigh_fraction=0.40)
    snapshot = build_swingset_snapshot(
        config,
        SwingPose(swing_angle_rad=0.12, hip_angle_rad=0.2, elbow_angle_rad=0.3),
    )

    assert snapshot.chain_nodes.shape == (9, 2)
    assert {"seat", "hand", "elbow", "shoulder", "waist", "hip", "knee", "foot"} <= set(
        snapshot.points
    )
    assert snapshot.center_of_mass_m.shape == (2,)
    assert snapshot.hand_chain_error_m == pytest.approx(0.0)
    assert snapshot.seat_chain_error_m == pytest.approx(0.0)
    thigh_attachment = snapshot.points["hip"] + 0.40 * (
        snapshot.points["knee"] - snapshot.points["hip"]
    )
    np.testing.assert_allclose(thigh_attachment, snapshot.chain_nodes[-1])
    assert np.max(np.abs(np.diff(snapshot.chain_nodes[:, 0]))) > 0.0


def test_swingset_elbow_branch_does_not_mirror_when_control_crosses_zero() -> None:
    config = SwingSetConfig(chain_segments=8)
    branch_signs: list[float] = []
    elbow_points: list[np.ndarray] = []

    for elbow_angle in np.linspace(-0.5, 0.5, 11):
        snapshot = build_swingset_snapshot(
            config,
            SwingPose(
                swing_angle_rad=0.12,
                torso_lean_rad=0.1,
                hip_angle_rad=0.2,
                elbow_angle_rad=float(elbow_angle),
            ),
        )
        shoulder = snapshot.points["shoulder"]
        hand = snapshot.points["hand"]
        elbow = snapshot.points["elbow"]
        hand_delta = hand - shoulder
        elbow_delta = elbow - shoulder
        branch_signs.append(
            float(hand_delta[0] * elbow_delta[1] - hand_delta[1] * elbow_delta[0])
        )
        elbow_points.append(elbow)

    # The elbow must never mirror to the far branch as the requested flexion
    # sweeps through zero (the IK stays on the +normal side).
    assert min(branch_signs) > 0.0
    # No discontinuous jump (a mirror flip would be a large step); the elbow
    # moves smoothly across the swept range.
    max_step = max(
        float(np.linalg.norm(end - start)) for start, end in pairwise(elbow_points)
    )
    assert max_step < 0.1


def test_swingset_elbow_bias_changes_geometry() -> None:
    """elbow_bias_rad must actually affect the resolved elbow position (#489).

    Previously the bias was discarded and hard-coded to 1.0, so every requested
    flexion produced the identical elbow point. A neutral (0 rad) elbow should
    sit at the full +normal offset; increasing flexion tucks the elbow toward
    the shoulder->hand line, producing a distinct, monotonically closer point.
    """
    config = SwingSetConfig(chain_segments=8)
    pose = SwingPose(swing_angle_rad=0.12, torso_lean_rad=0.1, hip_angle_rad=0.2)

    def elbow_for(angle: float) -> np.ndarray:
        snap = build_swingset_snapshot(config, replace(pose, elbow_angle_rad=angle))
        return snap.points["elbow"] - snap.points["shoulder"]

    neutral = elbow_for(0.0)
    mid = elbow_for(0.175)
    flexed = elbow_for(0.35)

    # Distinct poses must produce distinct elbow geometry.
    assert float(np.linalg.norm(flexed - neutral)) > 1e-3
    assert float(np.linalg.norm(mid - neutral)) > 1e-3
    # Perpendicular offset shrinks monotonically as the elbow flexes: the
    # neutral elbow is farthest off the shoulder->hand line.
    assert np.linalg.norm(neutral) > np.linalg.norm(mid) > np.linalg.norm(flexed)
    # A negative request clamps to the neutral (0 rad) limit, not the bug's
    # silent identical pose.
    np.testing.assert_allclose(elbow_for(-0.2), neutral, atol=1e-12)


def test_swingset_pose_constraints_keep_torso_upright() -> None:
    constrained = constrain_swing_pose(
        SwingPose(
            swing_angle_rad=0.0,
            torso_lean_rad=2.0,
            hip_angle_rad=-3.0,
            knee_angle_rad=2.0,
            shoulder_angle_rad=2.0,
            elbow_angle_rad=2.0,
        )
    )
    snapshot = build_swingset_snapshot(SwingSetConfig(chain_segments=10), constrained)

    lower, upper = SWING_TORSO_LEAN_LIMITS_RAD
    assert lower <= constrained.torso_lean_rad <= upper
    assert constrained.elbow_angle_rad <= 0.35

    shoulder = snapshot.points["shoulder"]
    hip = snapshot.points["hip"]
    # The rider sits upright: the shoulder rides above the hip (screen-up is
    # smaller world-y) and the trunk is closer to vertical than horizontal.
    assert shoulder[1] < hip[1]
    assert abs(shoulder[0] - hip[0]) < abs(shoulder[1] - hip[1])


def test_swingset_joint_end_range_damping_prevents_limit_overshoot() -> None:
    config = SwingSetConfig()
    torso_upper = SWING_TORSO_LEAN_LIMITS_RAD[1]
    near_limit = SwingSetState(
        pose=SwingPose(torso_lean_rad=torso_upper - 0.001, elbow_angle_rad=0.349),
        swing_angular_velocity_rad_s=0.0,
    )
    full_rate = SwingControlAction(torso_lean_rate_rad_s=2.0, elbow_rate_rad_s=2.0)

    stepped = step_swingset(config, near_limit, full_rate, dt_s=0.1)

    # End-range damping keeps the joints from overshooting their hard limits.
    assert stepped.pose.torso_lean_rad <= torso_upper
    assert stepped.pose.torso_lean_rad - near_limit.pose.torso_lean_rad < 0.01
    assert stepped.pose.elbow_angle_rad <= 0.35
    assert stepped.pose.elbow_angle_rad - near_limit.pose.elbow_angle_rad < 0.01


def test_swingset_pose_constraints_reject_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        constrain_swing_pose(SwingPose(torso_lean_rad=float("nan")))


def test_swingset_config_validates_inputs() -> None:
    with pytest.raises(ValueError, match="chain_segments"):
        SwingSetConfig(chain_segments=0)
    with pytest.raises(ValueError, match="chain_length_m"):
        SwingSetConfig(chain_length_m=-1.0)
    with pytest.raises(ValueError, match="damping"):
        SwingSetConfig(damping=-0.1)
    with pytest.raises(ValueError, match="pump_gain"):
        SwingSetConfig(pump_gain=-0.1)
    with pytest.raises(ValueError, match="seat_placement"):
        SwingSetConfig(seat_placement_thigh_fraction=0.0)
    with pytest.raises(ValueError, match="seat_placement"):
        SwingSetConfig(seat_placement_thigh_fraction=1.1)
    with pytest.raises(ValueError, match="length_m"):
        HumanSegmentSpec(0.0, 1.0)
    with pytest.raises(ValueError, match="mass_kg"):
        HumanSegmentSpec(1.0, 0.0)


def test_swingset_static_position_remains_static_without_control() -> None:
    config = SwingSetConfig()
    state = SwingSetState.rest()

    next_state = step_swingset(config, state, SwingControlAction(), dt_s=0.02)

    assert next_state.pose.swing_angle_rad == pytest.approx(0.0)
    assert next_state.swing_angular_velocity_rad_s == pytest.approx(0.0)


def test_swingset_single_segment_chain_snapshot_is_supported() -> None:
    snapshot = build_swingset_snapshot(SwingSetConfig(chain_segments=1), SwingPose())

    assert snapshot.chain_nodes.shape == (2, 2)
    assert snapshot.hand_chain_error_m == pytest.approx(0.0)


def test_swingset_control_limits_clip_rates_and_torque() -> None:
    action = SwingControlAction(
        torso_lean_rate_rad_s=99.0,
        hip_rate_rad_s=-99.0,
        knee_rate_rad_s=99.0,
        shoulder_rate_rad_s=-99.0,
        elbow_rate_rad_s=99.0,
        external_torque_nm=999.0,
    ).clipped()

    assert np.max(np.abs(action.vector())) == pytest.approx(2.4)
    assert action.external_torque_nm == pytest.approx(70.0)


def test_swingset_heuristic_policy_builds_amplitude() -> None:
    config = SwingSetConfig()

    rollout = simulate_swingset(
        config,
        SwingSetState(pose=SwingPose(swing_angle_rad=0.08)),
        steps=80,
        dt_s=0.02,
        policy=heuristic_pumping_policy,
    )

    assert rollout.controls.shape == (80, 5)
    assert rollout.max_abs_swing_angle_rad() == pytest.approx(
        rollout.metrics.max_abs_swing_angle_rad
    )
    assert rollout.metrics.max_abs_swing_angle_rad > 0.08
    assert rollout.metrics.max_height_gain_m > 0.0
    assert rollout.metrics.final_energy_proxy_j >= 0.0
    assert rollout.metrics.mean_hand_chain_error_m == pytest.approx(0.0)
    assert rollout.metrics.mean_seat_chain_error_m == pytest.approx(0.0)
    for snapshot in rollout.snapshots:
        np.testing.assert_allclose(snapshot.chain_nodes[0], 0.0)


def test_cyclic_policy_controls_match_callback_policy() -> None:
    params = CyclicPolicyParameters(
        frequency_hz=0.7,
        hip_rate_amplitude_rad_s=1.1,
        torso_rate_amplitude_rad_s=0.6,
        knee_rate_ratio=0.4,
        phase_rad=0.25,
    )
    dt_s = 0.02
    controls = cyclic_policy_controls(params, steps=8, dt_s=dt_s)
    policy = cyclic_pumping_policy(params)
    callback_controls = np.vstack(
        [policy(SwingSetState.rest(), step * dt_s).vector() for step in range(8)]
    )

    np.testing.assert_allclose(controls, callback_controls)


def test_swingset_cyclic_policy_search_selects_height_objective() -> None:
    result = optimize_cyclic_policy(SwingSetConfig(), steps=40, dt_s=0.02)

    assert result.objective_height_m == pytest.approx(
        result.rollout.metrics.max_height_gain_m
    )
    assert result.objective_height_m > 0.0
    assert result.parameters.frequency_hz > 0.0


def test_swingset_policy_search_reports_progress_and_uses_cycles() -> None:
    progress: list[tuple[int, int, float]] = []
    search_space = CyclicPolicySearchSpace(
        frequency_hz_min=0.5,
        frequency_hz_max=1.0,
        frequency_samples=2,
        hip_rate_min_rad_s=0.8,
        hip_rate_max_rad_s=0.8,
        hip_rate_samples=1,
        torso_rate_min_rad_s=0.4,
        torso_rate_max_rad_s=0.4,
        torso_rate_samples=1,
        knee_ratio_min=0.35,
        knee_ratio_max=0.35,
        knee_ratio_samples=1,
        phase_samples=2,
    )

    result = optimize_cyclic_policy(
        SwingSetConfig(),
        cycles=2.0,
        dt_s=0.02,
        search_space=search_space,
        progress_callback=lambda done, total, score, _params: progress.append(
            (done, total, score)
        ),
    )

    assert result.evaluated_candidates == 4
    assert result.optimized_cycles == pytest.approx(2.0)
    assert progress[-1][0] == progress[-1][1] == 4
    assert progress[-1][2] == pytest.approx(result.objective_height_m)
    assert len(result.trace) == result.evaluated_candidates
    assert result.trace[-1].best_score_m == pytest.approx(result.objective_height_m)
    best_scores = np.asarray([sample.best_score_m for sample in result.trace])
    assert np.all(np.diff(best_scores) >= 0.0)


def test_swingset_joint_torque_estimates_are_reported_per_policy_joint() -> None:
    config = SwingSetConfig()
    rollout = simulate_swingset(
        config,
        SwingSetState(pose=SwingPose(swing_angle_rad=0.08)),
        steps=8,
        dt_s=0.02,
        policy=heuristic_pumping_policy,
    )

    torques = estimate_swingset_joint_torques(config, rollout, dt_s=0.02)

    assert torques.shape == (8, 5)
    assert np.all(np.isfinite(torques))
    assert np.max(np.abs(torques)) > 0.0


def test_swingset_joint_torque_estimator_validates_control_history() -> None:
    config = SwingSetConfig()
    rollout = simulate_swingset(
        config,
        SwingSetState(pose=SwingPose(swing_angle_rad=0.08)),
        steps=2,
        dt_s=0.02,
        policy=heuristic_pumping_policy,
    )

    empty = replace(rollout, controls=np.zeros((0, 5), dtype=np.float64))
    bad_shape = replace(rollout, controls=np.zeros((2, 4), dtype=np.float64))

    assert estimate_swingset_joint_torques(config, empty, dt_s=0.02).shape == (0, 5)
    with pytest.raises(ValueError, match="shape"):
        estimate_swingset_joint_torques(config, bad_shape, dt_s=0.02)


def test_swingset_rollout_validates_inputs() -> None:
    config = SwingSetConfig()
    with pytest.raises(ValueError, match="steps"):
        simulate_swingset(
            config, SwingSetState.rest(), 0, 0.02, heuristic_pumping_policy
        )
    with pytest.raises(ValueError, match="dt_s"):
        step_swingset(config, SwingSetState.rest(), SwingControlAction(), dt_s=0.0)
    with pytest.raises(ValueError, match="steps"):
        optimize_cyclic_policy(config, steps=0)
    with pytest.raises(ValueError, match="frequency_hz"):
        CyclicPolicyParameters(0.0, 1.0, 1.0, 0.5, 0.0)
    with pytest.raises(ValueError, match="hip_rate"):
        CyclicPolicyParameters(1.0, -1.0, 1.0, 0.5, 0.0)
    with pytest.raises(ValueError, match="torso_rate"):
        CyclicPolicyParameters(1.0, 1.0, -1.0, 0.5, 0.0)
    with pytest.raises(ValueError, match="knee_rate"):
        CyclicPolicyParameters(1.0, 1.0, 1.0, -0.5, 0.0)
    with pytest.raises(ValueError, match="frequency_samples"):
        CyclicPolicySearchSpace(frequency_samples=0)
    with pytest.raises(ValueError, match="frequency_hz"):
        CyclicPolicySearchSpace(frequency_hz_min=1.0, frequency_hz_max=0.5)
    with pytest.raises(ValueError, match="cycles"):
        optimize_cyclic_policy(config, cycles=0.0)


# ---------------------------------------------------------------------------
# Iterative optimizer (optimize_cyclic_policy_iterative)
# ---------------------------------------------------------------------------


def test_iterative_optimizer_is_deterministic() -> None:
    config = SwingSetConfig()
    first = optimize_cyclic_policy_iterative(config, steps=40, budget=60, seed=7)
    second = optimize_cyclic_policy_iterative(config, steps=40, budget=60, seed=7)
    assert first.objective_height_m == pytest.approx(second.objective_height_m)
    assert first.parameters.frequency_hz == pytest.approx(
        second.parameters.frequency_hz
    )
    assert first.parameters.phase_rad == pytest.approx(second.parameters.phase_rad)
    assert len(first.trace) == len(second.trace)


@pytest.mark.parametrize("budget", [50, 120, 300])
def test_iterative_optimizer_honors_budget(budget: int) -> None:
    config = SwingSetConfig()
    result = optimize_cyclic_policy_iterative(config, steps=40, budget=budget, seed=1)
    assert result.evaluated_candidates <= budget
    assert len(result.trace) == result.evaluated_candidates


def test_iterative_optimizer_matches_or_beats_grid() -> None:
    config = SwingSetConfig()
    grid = optimize_cyclic_policy(
        config, steps=80, search_space=CyclicPolicySearchSpace()
    )
    iterative = optimize_cyclic_policy_iterative(config, steps=80, budget=400, seed=0)
    assert iterative.objective_height_m >= grid.objective_height_m - 0.05


def test_iterative_optimizer_trace_best_is_monotonic() -> None:
    config = SwingSetConfig()
    result = optimize_cyclic_policy_iterative(config, steps=40, budget=120, seed=3)
    best = [sample.best_score_m for sample in result.trace]
    assert all(later >= earlier for earlier, later in pairwise(best))
    assert result.trace[-1].best_parameters.frequency_hz == pytest.approx(
        result.parameters.frequency_hz
    )


def test_iterative_optimizer_progress_callback_contract() -> None:
    config = SwingSetConfig()
    calls: list[tuple[int, int, float]] = []

    def _record(
        completed: int, total: int, best: float, params: CyclicPolicyParameters
    ) -> None:
        calls.append((completed, total, best))
        assert isinstance(params, CyclicPolicyParameters)

    result = optimize_cyclic_policy_iterative(
        config, steps=40, budget=80, seed=2, progress_callback=_record
    )
    assert calls
    assert all(total == 80 for _completed, total, _best in calls)
    best_values = [best for _c, _t, best in calls]
    assert all(later >= earlier for earlier, later in pairwise(best_values))
    assert calls[-1][0] <= 80
    assert result.evaluated_candidates <= 80


def test_iterative_optimizer_without_refine_still_valid() -> None:
    config = SwingSetConfig()
    result = optimize_cyclic_policy_iterative(
        config, steps=40, budget=90, seed=0, local_refine=False
    )
    assert result.evaluated_candidates <= 90
    assert np.isfinite(result.objective_height_m)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": 0},
        {"budget": MAX_OPTIMIZER_BUDGET + 1},
        {"steps": 0},
        {"dt_s": 0.0},
        {"cycles": -1.0},
    ],
)
def test_iterative_optimizer_rejects_bad_inputs(kwargs: dict) -> None:
    config = SwingSetConfig()
    call = {"steps": 40, "budget": 40, "seed": 0}
    call.update(kwargs)
    with pytest.raises(ValueError):
        optimize_cyclic_policy_iterative(config, **call)


def test_cyclic_policy_bounds_validation() -> None:
    with pytest.raises(ValueError, match="frequency_hz"):
        CyclicPolicyBounds(frequency_hz=(0.8, 0.4))
    with pytest.raises(ValueError, match="phase_rad_min"):
        CyclicPolicyBounds(phase_rad=(-0.1, 1.0))
    with pytest.raises(ValueError, match="phase_rad_max"):
        CyclicPolicyBounds(phase_rad=(1.0, 0.5))
