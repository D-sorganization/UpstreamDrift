"""Contracts for model-independent drift/control joint-transfer attribution."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.drift_control_transfer import (
    JointTransferTrajectory,
    PathWeightedMeanForce,
    PhaseTransferSummary,
    SwingPhase,
    attribution_shares,
    build_phase_masks,
    compute_impulses,
    compute_path_frame,
    compute_path_weighted_mean_force,
    compute_power_and_work,
    project_forces_onto_path,
    summarize_phases,
)

pytestmark = pytest.mark.unit


def _trajectory() -> JointTransferTrajectory:
    time = np.array([0.0, 0.5, 1.0])
    velocity = np.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 2.0]],
        ]
    )
    drift_force = np.array(
        [
            [[2.0, 1.0], [1.0, 3.0]],
            [[2.0, 1.0], [1.0, 3.0]],
            [[2.0, 1.0], [1.0, 3.0]],
        ]
    )
    control_force = np.array(
        [
            [[-1.0, 2.0], [2.0, -1.0]],
            [[-1.0, 2.0], [2.0, -1.0]],
            [[-1.0, 2.0], [2.0, -1.0]],
        ]
    )
    drift_couple = np.full((3, 2), 2.0)
    control_couple = np.full((3, 2), -0.5)
    return JointTransferTrajectory(
        time=time,
        joint_names=("shoulder", "wrist"),
        position=np.zeros((3, 2, 2)),
        velocity=velocity,
        force_total=drift_force + control_force,
        force_drift=drift_force,
        force_control=control_force,
        couple_total=drift_couple + control_couple,
        couple_drift=drift_couple,
        couple_control=control_couple,
        angular_velocity=np.full((3, 2), 4.0),
        model_tier="fixture",
        force_direction="proximal_on_distal",
    )


def test_trajectory_validates_time_shapes_finiteness_and_split_closure() -> None:
    trajectory = _trajectory()
    assert trajectory.sample_count == 3
    assert trajectory.joint_count == 2

    with pytest.raises(ValueError, match="strictly increasing"):
        JointTransferTrajectory(
            **{**trajectory.as_init_dict(), "time": np.array([0.0, 0.5, 0.5])}
        )
    with pytest.raises(ValueError, match="force_total"):
        JointTransferTrajectory(
            **{
                **trajectory.as_init_dict(),
                "force_total": trajectory.force_total + 1.0,
            }
        )
    with pytest.raises(ValueError, match="finite"):
        bad = trajectory.velocity.copy()
        bad[0, 0, 0] = np.nan
        JointTransferTrajectory(**{**trajectory.as_init_dict(), "velocity": bad})


def test_path_frame_and_force_projection_reconstruct_valid_vectors() -> None:
    trajectory = _trajectory()
    frame = compute_path_frame(trajectory.velocity, speed_epsilon=1e-9)
    projection = project_forces_onto_path(trajectory, frame)

    np.testing.assert_allclose(frame.speed[:, 0], 1.0)
    np.testing.assert_allclose(frame.speed[:, 1], 2.0)
    np.testing.assert_allclose(frame.tangent[:, 0], np.tile([1.0, 0.0], (3, 1)))
    np.testing.assert_allclose(frame.normal[:, 1], np.tile([-1.0, 0.0], (3, 1)))
    reconstructed = (
        projection.total_along[..., None] * frame.tangent
        + projection.total_normal[..., None] * frame.normal
    )
    np.testing.assert_allclose(
        reconstructed[frame.valid], trajectory.force_total[frame.valid]
    )
    np.testing.assert_allclose(
        projection.total_along,
        projection.drift_along + projection.control_along,
    )


def test_path_frame_marks_stationary_samples_undefined() -> None:
    velocity = np.zeros((3, 1, 2))
    velocity[1, 0] = [1e-3, 0.0]
    frame = compute_path_frame(velocity, speed_epsilon=1e-4)
    assert frame.valid[:, 0].tolist() == [False, True, False]
    np.testing.assert_array_equal(frame.tangent[[0, 2], 0], 0.0)


def test_impulses_include_vector_and_signed_positive_negative_absolute_path_terms() -> (
    None
):
    trajectory = _trajectory()
    projection = project_forces_onto_path(
        trajectory, compute_path_frame(trajectory.velocity)
    )
    impulses = compute_impulses(trajectory, projection)

    np.testing.assert_allclose(impulses.vector_total[-1], trajectory.force_total[0])
    np.testing.assert_allclose(
        impulses.vector_total,
        impulses.vector_drift + impulses.vector_control,
    )
    np.testing.assert_allclose(impulses.tangent_total_signed[-1], [1.0, 2.0])
    np.testing.assert_allclose(impulses.tangent_total_positive[-1], [1.0, 2.0])
    np.testing.assert_allclose(impulses.tangent_total_negative[-1], 0.0)
    np.testing.assert_allclose(impulses.tangent_total_absolute[-1], [1.0, 2.0])
    np.testing.assert_allclose(
        impulses.tangent_total_signed,
        impulses.tangent_drift_signed + impulses.tangent_control_signed,
    )


def test_force_couple_total_power_and_cumulative_work_close() -> None:
    trajectory = _trajectory()
    result = compute_power_and_work(trajectory)

    np.testing.assert_allclose(
        result.force_power_total,
        result.force_power_drift + result.force_power_control,
    )
    np.testing.assert_allclose(
        result.couple_power_total,
        result.couple_power_drift + result.couple_power_control,
    )
    np.testing.assert_allclose(
        result.total_power_total,
        result.force_power_total + result.couple_power_total,
    )
    np.testing.assert_allclose(
        result.total_work_total,
        result.total_work_drift + result.total_work_control,
    )
    np.testing.assert_allclose(result.total_work_total[-1], result.total_power_total[0])


def test_phase_masks_are_deterministic_nonoverlapping_and_exhaustive() -> None:
    time = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    phases = (
        SwingPhase("Early Downswing", 0.0, 0.5),
        SwingPhase("Late Downswing", 0.5, 1.0),
    )
    masks = build_phase_masks(time, phases)
    assert masks["Early Downswing"].tolist() == [True, True, False, False, False]
    assert masks["Late Downswing"].tolist() == [False, False, True, True, True]
    np.testing.assert_array_equal(sum(masks.values()), np.ones(time.size))

    with pytest.raises(ValueError, match="adjacent"):
        build_phase_masks(
            time,
            (
                SwingPhase("Early", 0.0, 0.4),
                SwingPhase("Late", 0.5, 1.0),
            ),
        )


def test_attribution_shares_handle_zero_and_cancellation_without_infinity() -> None:
    drift = np.array([8.0, 1.0, 0.0, -2.0])
    control = np.array([2.0, -1.0, 0.0, 1.0])
    total = drift + control
    shares = attribution_shares(total, drift, control, epsilon=1e-12)

    np.testing.assert_allclose(shares.signed_drift_share[[0, 3]], [0.8, 2.0])
    assert not shares.signed_valid[1]
    assert not shares.magnitude_valid[2]
    assert np.isnan(shares.signed_drift_share[1])
    assert np.isnan(shares.magnitude_drift_share[2])
    np.testing.assert_allclose(
        shares.magnitude_drift_share[shares.magnitude_valid]
        + shares.magnitude_control_share[shares.magnitude_valid],
        1.0,
    )
    assert shares.cancellation_index[1] == pytest.approx(1.0)
    assert np.all(np.isfinite(shares.cancellation_index[shares.magnitude_valid]))

    with pytest.raises(ValueError, match="total"):
        attribution_shares(total + 1.0, drift, control)


def test_path_weighted_mean_force_is_signed_linear_work_over_path_length() -> None:
    velocity = np.array([[[1.0, 0.0]], [[3.0, 0.0]], [[3.0, 0.0]]])
    drift_force = np.array([[[10.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
    control_force = np.zeros_like(drift_force)
    zeros = np.zeros((3, 1))
    trajectory = JointTransferTrajectory(
        time=np.array([0.0, 0.5, 1.0]),
        joint_names=("hand",),
        position=np.zeros((3, 1, 2)),
        velocity=velocity,
        force_total=drift_force,
        force_drift=drift_force,
        force_control=control_force,
        couple_total=zeros,
        couple_drift=zeros,
        couple_control=zeros,
        angular_velocity=zeros,
        model_tier="fixture",
    )
    estimand = compute_path_weighted_mean_force(
        trajectory, compute_path_frame(trajectory.velocity)
    )

    assert isinstance(estimand, PathWeightedMeanForce)
    assert estimand.valid.tolist() == [True]
    np.testing.assert_allclose(estimand.path_length, [2.5])
    np.testing.assert_allclose(estimand.force_work_total, [2.5])
    np.testing.assert_allclose(estimand.force_work_drift, [2.5])
    np.testing.assert_allclose(estimand.force_work_control, [0.0])
    np.testing.assert_allclose(estimand.mean_force_total, [1.0])
    np.testing.assert_allclose(estimand.mean_force_drift, [1.0])
    np.testing.assert_allclose(estimand.mean_force_control, [0.0])
    assert estimand.mean_force_total[0] != pytest.approx(2.5)


def test_path_weighted_mean_force_is_undefined_without_valid_path_intervals() -> None:
    trajectory = _trajectory()
    velocity = np.zeros_like(trajectory.velocity)
    velocity[1] = trajectory.velocity[1]
    stationary = JointTransferTrajectory(
        **{**trajectory.as_init_dict(), "velocity": velocity}
    )
    estimand = compute_path_weighted_mean_force(
        stationary, compute_path_frame(stationary.velocity)
    )

    assert not np.any(estimand.valid)
    np.testing.assert_allclose(estimand.path_length, 0.0)
    np.testing.assert_allclose(estimand.force_work_total, 0.0)
    assert np.all(np.isnan(estimand.mean_force_total))


def test_phase_summaries_use_disjoint_intervals_and_end_minus_start_values() -> None:
    trajectory = _trajectory()
    impulses = compute_impulses(
        trajectory,
        project_forces_onto_path(trajectory, compute_path_frame(trajectory.velocity)),
    )
    power_work = compute_power_and_work(trajectory)
    phases = (
        SwingPhase("Early Downswing", 0.0, 0.5),
        SwingPhase("Late Downswing", 0.5, 1.0),
    )
    summaries = summarize_phases(trajectory, impulses, power_work, phases)

    assert all(isinstance(item, PhaseTransferSummary) for item in summaries)
    assert [(item.start_index, item.end_index) for item in summaries] == [
        (0, 1),
        (1, 2),
    ]
    assert [item.interval_count for item in summaries] == [1, 1]
    assert [(item.start_time_s, item.end_time_s) for item in summaries] == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]
    np.testing.assert_allclose(
        summaries[0].vector_impulse_total + summaries[1].vector_impulse_total,
        impulses.vector_total[-1],
    )
    np.testing.assert_allclose(
        summaries[0].total_work_total + summaries[1].total_work_total,
        power_work.total_work_total[-1],
    )

    with pytest.raises(ValueError, match="sample time"):
        summarize_phases(
            trajectory,
            impulses,
            power_work,
            (
                SwingPhase("Early", 0.0, 0.4),
                SwingPhase("Late", 0.4, 1.0),
            ),
        )
