"""Contracts for proximal-to-distal spatial evidence and predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.interaction_evidence import (
    EvidenceManifest,
    InterfaceDescriptor,
    PredictionRecord,
    SpatialWrenchTrajectory,
    calibrate_convergence_tolerance,
    load_evidence_manifest,
    spatial_from_planar,
)
from src.shared.python.biomechanics.drift_control_transfer import (
    JointTransferTrajectory,
)

pytestmark = pytest.mark.unit


def _trajectory() -> SpatialWrenchTrajectory:
    time = np.array([0.0, 0.5, 1.0])
    force_drift = np.array(
        [
            [[2.0, -1.0, 0.5]],
            [[2.5, -0.5, 0.4]],
            [[3.0, 0.0, 0.3]],
        ]
    )
    force_control = np.array(
        [
            [[-0.5, 0.4, 0.0]],
            [[-0.4, 0.3, 0.1]],
            [[-0.3, 0.2, 0.0]],
        ]
    )
    moment_drift = np.array([[[0.1, 0.2, -1.0]]] * 3)
    moment_control = np.array([[[0.0, -0.1, 0.3]]] * 3)
    drift = np.concatenate((force_drift, moment_drift), axis=2)
    control = np.concatenate((force_control, moment_control), axis=2)
    twist = np.array([[[1.0, 0.2, -0.1, 0.1, -0.2, 2.0]]] * 3)
    return SpatialWrenchTrajectory(
        time=time,
        interfaces=(
            InterfaceDescriptor(
                name="mid_grip",
                proximal_body="hands",
                distal_body="club",
                frame="world",
                reference_point="mid_grip",
                action_direction="proximal_on_distal",
            ),
        ),
        reference_position_m=np.zeros((3, 1, 3)),
        wrench_total=drift + control,
        wrench_drift=drift,
        wrench_control=control,
        twist=twist,
        model_tier="fixture",
    )


def _prediction(prediction_id: str = "PD-H1-P1") -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        hypothesis_id="H1",
        statement="Late distal work contains a state-matched drift contribution.",
        estimand="Late-window distal interface drift work in joules.",
        intervention="Set commanded distal torque to zero at the achieved state.",
        expected_result="Drift work remains positive in the declared late window.",
        falsifier="Drift work is non-positive or fails component closure.",
        competing_explanations=("shaft recoil", "prescribed-base work"),
        negative_controls=("remove velocity-dependent terms",),
        required_model_tiers=("double_pendulum", "forward_two_hand"),
        tolerance_id="convergence-primary",
    )


def test_spatial_trajectory_closes_and_computes_wrench_power() -> None:
    trajectory = _trajectory()

    np.testing.assert_allclose(
        trajectory.wrench_total,
        trajectory.wrench_drift + trajectory.wrench_control,
    )
    np.testing.assert_allclose(
        trajectory.power("total"),
        trajectory.power("drift") + trajectory.power("control"),
    )
    assert trajectory.sample_count == 3
    assert trajectory.interface_count == 1


def test_spatial_trajectory_rejects_nonclosing_or_ambiguous_evidence() -> None:
    trajectory = _trajectory()
    with pytest.raises(ValueError, match="wrench_total"):
        SpatialWrenchTrajectory(
            **{
                **trajectory.as_init_dict(),
                "wrench_total": trajectory.wrench_total + 1.0,
            }
        )
    with pytest.raises(ValueError, match="unique"):
        SpatialWrenchTrajectory(
            **{
                **trajectory.as_init_dict(),
                "interfaces": trajectory.interfaces * 2,
                "reference_position_m": np.repeat(
                    trajectory.reference_position_m, 2, axis=1
                ),
                "wrench_total": np.repeat(trajectory.wrench_total, 2, axis=1),
                "wrench_drift": np.repeat(trajectory.wrench_drift, 2, axis=1),
                "wrench_control": np.repeat(trajectory.wrench_control, 2, axis=1),
                "twist": np.repeat(trajectory.twist, 2, axis=1),
            }
        )


def test_reference_transport_preserves_total_power_and_round_trips() -> None:
    trajectory = _trajectory()
    new_points = np.array(
        [
            [[0.2, -0.1, 0.0]],
            [[0.3, -0.1, 0.1]],
            [[0.4, 0.0, 0.1]],
        ]
    )

    transported = trajectory.transport(new_points, reference_point="club_center")
    restored = transported.transport(
        trajectory.reference_position_m,
        reference_point="mid_grip",
    )

    np.testing.assert_allclose(transported.power("total"), trajectory.power("total"))
    np.testing.assert_allclose(restored.wrench_total, trajectory.wrench_total)
    np.testing.assert_allclose(restored.twist, trajectory.twist)


def test_proper_rotation_preserves_power_and_rejects_reflections() -> None:
    trajectory = _trajectory()
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = trajectory.rotate(rotation, frame="laboratory")

    np.testing.assert_allclose(rotated.power("total"), trajectory.power("total"))
    with pytest.raises(ValueError, match="proper rotation"):
        trajectory.rotate(np.diag([1.0, 1.0, -1.0]), frame="reflected")


def test_prediction_contract_requires_predeclared_falsifier_and_alternatives() -> None:
    assert _prediction().status == "untested"
    with pytest.raises(ValueError, match="falsifier"):
        PredictionRecord(**{**_prediction().as_record(), "falsifier": ""})
    with pytest.raises(ValueError, match="competing_explanations"):
        PredictionRecord(**{**_prediction().as_record(), "competing_explanations": ()})


def test_convergence_tolerance_uses_finest_pair_before_outcome_evaluation() -> None:
    tolerance = calibrate_convergence_tolerance(
        tolerance_id="convergence-primary",
        step_sizes=np.array([0.004, 0.002, 0.001]),
        observed_values=np.array([10.20, 10.05, 10.01]),
        safety_factor=2.0,
        source="three-level manufactured fixture",
    )

    assert tolerance.absolute == pytest.approx(0.08)
    assert tolerance.relative == pytest.approx(0.08 / 10.01)
    assert tolerance.calibration_method == "finest-pair-difference-times-safety-factor"
    with pytest.raises(ValueError, match="strictly decreasing"):
        calibrate_convergence_tolerance(
            tolerance_id="bad",
            step_sizes=np.array([0.001, 0.002]),
            observed_values=np.array([1.0, 1.1]),
            safety_factor=2.0,
            source="fixture",
        )


def test_manifest_round_trip_and_repository_prediction_registry() -> None:
    tolerance = calibrate_convergence_tolerance(
        tolerance_id="convergence-primary",
        step_sizes=np.array([0.004, 0.002, 0.001]),
        observed_values=np.array([10.20, 10.05, 10.01]),
        safety_factor=2.0,
        source="fixture",
    )
    manifest = EvidenceManifest(
        study_id="proximal-distal-model-completion",
        predictions=(_prediction(),),
        tolerances=(tolerance,),
    )
    rebuilt = EvidenceManifest.from_record(json.loads(json.dumps(manifest.as_record())))
    assert rebuilt == manifest

    root = Path(__file__).resolve().parents[3]
    registered = load_evidence_manifest(
        root
        / "docs/research/proximal_distal_energy_transfer/data/model_completion_predictions.json"
    )
    assert registered.schema_version == "proximal-distal-evidence-v2"
    assert {prediction.hypothesis_id for prediction in registered.predictions} == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
    }
    statuses = {
        prediction.hypothesis_id: prediction.status
        for prediction in registered.predictions
    }
    assert statuses == {
        "H1": "supported",
        "H2": "supported",
        "H3": "supported",
        "H4": "supported",
        "H5": "inconclusive",
    }
    assert all(prediction.status_scope for prediction in registered.predictions)
    assert all(prediction.remaining_gate for prediction in registered.predictions)
    assert all(prediction.falsifier for prediction in registered.predictions)


def test_adjudicated_prediction_requires_scope_and_remaining_gate() -> None:
    with pytest.raises(ValueError, match="status_scope"):
        PredictionRecord(**{**_prediction().as_record(), "status": "supported"})


def test_manifest_rejects_duplicate_ids_and_missing_tolerances() -> None:
    prediction = _prediction()
    with pytest.raises(ValueError, match="unique"):
        EvidenceManifest(
            study_id="fixture",
            predictions=(prediction, prediction),
            tolerances=(),
        )
    with pytest.raises(ValueError, match="unknown tolerance"):
        EvidenceManifest(
            study_id="fixture",
            predictions=(prediction,),
            tolerances=(),
        )


def test_planar_migration_is_lossless_in_declared_axes() -> None:
    time = np.array([0.0, 1.0])
    force_drift = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    force_control = np.array([[[0.5, -0.5]], [[0.2, -0.2]]])
    couple_drift = np.array([[2.0], [3.0]])
    couple_control = np.array([[-1.0], [-1.5]])
    planar = JointTransferTrajectory(
        time=time,
        joint_names=("wrist",),
        position=np.array([[[0.1, 0.2]], [[0.3, 0.4]]]),
        velocity=np.array([[[1.0, 0.0]], [[0.0, 2.0]]]),
        force_total=force_drift + force_control,
        force_drift=force_drift,
        force_control=force_control,
        couple_total=couple_drift + couple_control,
        couple_drift=couple_drift,
        couple_control=couple_control,
        angular_velocity=np.array([[4.0], [5.0]]),
        model_tier="planar_fixture",
        frame="swing_plane_cartesian",
        reference_point="joint_origin",
        force_direction="proximal_on_distal",
    )

    spatial = spatial_from_planar(
        planar,
        body_pairs=(("forearm", "club"),),
    )

    np.testing.assert_array_equal(
        spatial.reference_position_m[..., :2], planar.position
    )
    np.testing.assert_array_equal(spatial.wrench_total[..., :2], planar.force_total)
    np.testing.assert_array_equal(spatial.wrench_total[..., 5], planar.couple_total)
    np.testing.assert_array_equal(spatial.twist[..., :2], planar.velocity)
    np.testing.assert_array_equal(spatial.twist[..., 5], planar.angular_velocity)
    np.testing.assert_array_equal(spatial.reference_position_m[..., 2], 0.0)
    assert spatial.interfaces[0].proximal_body == "forearm"
    assert spatial.interfaces[0].distal_body == "club"
