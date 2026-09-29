"""Analytic contracts for model-independent golf trajectories."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.biomechanics.golf_trajectory import (
    GolfTrajectory,
    SegmentTrajectory,
    compute_golf_metrics,
)

pytestmark = pytest.mark.unit


def segment(name, angles, mass=1.0, membership="body", positions=None):
    n = len(angles)
    return SegmentTrajectory(
        name,
        np.zeros((n, 3)) if positions is None else positions,
        Rotation.from_euler("z", np.asarray(angles)[:, None]).as_matrix(),
        mass,
        np.zeros(3),
        membership,
    )


def trajectory(segments, **kwargs):
    return GolfTrajectory(
        np.arange(len(next(iter(segments.values())).positions), dtype=float),
        segments,
        source="analytic",
        world_frame="right-handed z-up",
        **kwargs,
    )


def test_separation_stretch_is_event_angle_not_rate():
    t = trajectory(
        {
            "pelvis": segment("pelvis", [0, 0, 0, 0]),
            "thorax": segment("thorax", np.deg2rad([10, 30, 45, 5])),
        },
        transition_index=1,
        impact_index=3,
        event_name="hip_transition",
    )
    result = compute_golf_metrics(t)
    np.testing.assert_allclose(
        result.channels["x_factor_projected"].values, np.deg2rad([10, 30, 45, 5])
    )
    assert result.summaries["x_factor_stretch"] == pytest.approx(np.deg2rad(15))
    assert result.channels["x_factor_rate"].unit == "rad/s"


def test_mass_weighted_body_and_club_com_use_rotated_local_offset():
    body = segment("pelvis", [np.pi / 2] * 3, mass=3)
    body = SegmentTrajectory(
        body.name, body.positions, body.rotations, 3, np.array([1.0, 0, 0]), "body"
    )
    club = segment(
        "club",
        [0] * 3,
        mass=1,
        membership="club",
        positions=np.tile([4.0, 0, 0], (3, 1)),
    )
    result = compute_golf_metrics(
        trajectory({"pelvis": body, "club": club}, expected_body_segments=("pelvis",))
    )
    np.testing.assert_allclose(
        result.channels["body_com"].values, np.tile([0, 1.0, 0], (3, 1)), atol=1e-15
    )
    np.testing.assert_allclose(
        result.channels["body_club_com"].values,
        np.tile([1.0, 0.75, 0], (3, 1)),
        atol=1e-15,
    )


def test_shaft_twist_projects_physical_omega_not_euler_derivative():
    club = segment("club", [0, 0.2, 0.4, 0.6], membership="club")
    result = compute_golf_metrics(
        trajectory({"club": club}, shaft_axis_local=np.array([0.0, 0, 1]))
    )
    np.testing.assert_allclose(
        result.channels["shaft_twist_velocity"].values, 0.2, atol=1e-14
    )
    np.testing.assert_allclose(
        result.channels["segment.club.angular_velocity_world"].values,
        np.tile([0, 0, 0.2], (4, 1)),
        atol=1e-14,
    )


def test_missing_mass_and_membership_never_silently_renormalize():
    result = compute_golf_metrics(
        trajectory(
            {"pelvis": segment("pelvis", [0] * 3)},
            expected_body_segments=("pelvis", "head"),
        )
    )
    assert "body_com" not in result.channels
    assert "body_com" in result.unavailable


def test_gaps_remain_gaps_and_serialization_is_strict_json():
    positions = np.array([[0.0, 0, 0], [1, 0, 0], [np.nan] * 3, [50, 0, 0], [51, 0, 0]])
    club = segment("club", [0] * 5, membership="club", positions=positions)
    result = compute_golf_metrics(
        trajectory({"club": club}, clubhead_local=np.zeros(3))
    )
    np.testing.assert_allclose(
        result.channels["clubhead_speed"].values, [1, 1, np.nan, 1, 1], equal_nan=True
    )
    assert (
        json.loads(json.dumps(result.to_dict(), allow_nan=False))["channels"][
            "clubhead_speed"
        ]["values"][2]
        is None
    )


@pytest.mark.parametrize("times", [[0, 0, 1], [0, 2, 1], [0, np.nan, 2]])
def test_rejects_invalid_times(times):
    with pytest.raises(ValueError, match="time"):
        GolfTrajectory(
            np.array(times), {"pelvis": segment("pelvis", [0] * 3)}, "analytic", "z-up"
        )


def test_rejects_reflected_rotation():
    with pytest.raises(ValueError, match="rotation"):
        SegmentTrajectory(
            "a", np.zeros((3, 3)), np.tile(np.diag([-1.0, 1, 1]), (3, 1, 1))
        )


def test_parser_preserves_provenance_and_rejects_unknown_fields():
    from src.shared.python.biomechanics.golf_trajectory import golf_trajectory_from_dict

    payload = {
        "times": [0, 1],
        "source": "kinematic",
        "world_frame": "lab",
        "calibration_id": "subject-7",
        "segments": {
            "club": {
                "positions": [[0, 0, 0], [None, None, None]],
                "rotations": np.tile(np.eye(3), (2, 1, 1)).tolist(),
                "membership": "club",
            }
        },
    }
    parsed = golf_trajectory_from_dict(payload)
    assert parsed.calibration_id == "subject-7"
    assert np.isnan(parsed.segments["club"].positions[1]).all()
    assert (
        compute_golf_metrics(parsed).to_dict()["provenance"]["calibration_id"]
        == "subject-7"
    )
    with pytest.raises(ValueError, match="Unknown"):
        golf_trajectory_from_dict({**payload, "units": "degrees"})


def test_3d_separation_distinct_from_projected_and_vertical_is_undefined():
    pelvis = segment("pelvis", [0] * 3)
    matrices = Rotation.from_euler(
        "y", np.array([0.0, 0.5, np.pi / 2])[:, None]
    ).as_matrix()
    thorax = SegmentTrajectory("thorax", np.zeros((3, 3)), matrices)
    result = compute_golf_metrics(trajectory({"pelvis": pelvis, "thorax": thorax}))
    np.testing.assert_allclose(
        result.channels["x_factor_3d"].values, [0.0, 0.5, np.pi / 2], atol=1e-14
    )
    np.testing.assert_allclose(
        result.channels["x_factor_projected"].values[:2], 0.0, atol=1e-14
    )
    assert np.isnan(result.channels["x_factor_projected"].values[2])


def test_rotational_gap_and_unresolved_half_turn_are_unavailable():
    from src.shared.python.biomechanics.golf_trajectory import angular_velocity

    matrices = Rotation.from_euler(
        "z", np.array([0.0, 0.2, 0.4, 0.6, 0.8])[:, None]
    ).as_matrix()
    matrices[2] = np.nan
    omega = angular_velocity(np.arange(5.0), matrices)
    np.testing.assert_allclose(omega[[0, 1, 3, 4], 2], 0.2)
    assert np.isnan(omega[2]).all()
    matrices = Rotation.from_euler("z", np.array([0.0, np.pi])[:, None]).as_matrix()
    assert np.isnan(angular_velocity(np.arange(2.0), matrices)).all()


def test_event_stretch_with_gap_and_unknown_mass_is_unavailable():
    p = segment("pelvis", [0] * 3, mass=None)
    r = np.tile(np.eye(3), (3, 1, 1))
    r[1] = np.nan
    q = SegmentTrajectory("thorax", np.zeros((3, 3)), r)
    result = compute_golf_metrics(
        trajectory(
            {"pelvis": p, "thorax": q},
            expected_body_segments=("pelvis",),
            transition_index=0,
            impact_index=2,
            event_name="club_top",
        )
    )
    assert "x_factor_stretch" in result.unavailable
    assert "body_com" in result.unavailable


def test_world_frame_covariance_nonuniform_time_and_directed_shaft_axis():
    times = np.array([0.0, 0.1, 0.4, 1.0])
    base = Rotation.from_euler("x", [0.7]).as_matrix()
    rotations = base @ Rotation.from_euler("z", (0.3 * times)[:, None]).as_matrix()
    club = SegmentTrajectory("club", np.zeros((4, 3)), rotations, membership="club")
    t = GolfTrajectory(
        times,
        {"club": club},
        "synthetic",
        "tilted",
        shaft_axis_local=np.array([0.0, 0, -1]),
    )
    result = compute_golf_metrics(t)
    np.testing.assert_allclose(
        result.channels["shaft_twist_velocity"].values, -0.3, atol=1e-14
    )
    np.testing.assert_allclose(
        result.channels["segment.club.angular_velocity_world"].values,
        np.tile(base @ np.array([0.0, 0, 0.3]), (4, 1)),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transition_index": 1},
        {"transition_index": 2, "impact_index": 1, "event_name": "club_top"},
        {"transition_index": True, "event_name": "club_top"},
        {"shaft_axis_local": [0, 0, 2]},
        {"expected_body_segments": ["pelvis", "pelvis"]},
    ],
)
def test_calibration_and_event_contracts(kwargs):
    with pytest.raises(ValueError):
        trajectory({"pelvis": segment("pelvis", [0] * 3)}, **kwargs)


def test_input_arrays_are_copied_and_body_membership_does_not_drift():
    positions = np.zeros((3, 3))
    s = segment("pelvis", [0] * 3, positions=positions)
    segments = {"pelvis": s}
    t = trajectory(segments, expected_body_segments=("pelvis",))
    positions[:] = 99
    segments.clear()
    np.testing.assert_array_equal(t.segments["pelvis"].positions, 0)
    with pytest.raises(TypeError):
        t.segments["head"] = s
