"""The strike trace: the artifact view the metrics are computed from (#8614).

These tests cover the input contract itself -- validation, the quaternion
algebra, the scene geometry, and the round trip through the versioned result
artifact that makes the metrics fidelity-tier agnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.io.schema import BunkerShotResultWriter
from bunkershot3d.metrics import (
    HeadModel,
    StrikeScene,
    StrikeTrace,
    angular_velocity_world_radps,
    rotate_body_to_world,
    rotate_world_to_body,
)

from .conftest import build_decelerating_trace, reference_head

pytestmark = pytest.mark.unit


def _spin_quaternions(time_s: np.ndarray, rate_radps: float) -> np.ndarray:
    """Return quaternions for a constant-rate spin about world +z."""
    angle = 0.5 * rate_radps * time_s
    return np.column_stack(
        [np.cos(angle), np.zeros_like(angle), np.zeros_like(angle), np.sin(angle)]
    )


class TestRotation:
    """Quaternion helpers, scalar-first, body -> world."""

    def test_a_quarter_turn_about_z_takes_x_to_y(self) -> None:
        """q = (cos 45, 0, 0, sin 45) rotates by +90 deg about +z."""
        quat = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]])

        rotated = rotate_body_to_world(quat, np.array([1.0, 0.0, 0.0]))

        np.testing.assert_allclose(rotated[0], [0.0, 1.0, 0.0], atol=1e-15)

    def test_world_to_body_inverts_body_to_world(self) -> None:
        """Round-tripping a vector through both directions returns it."""
        quat = np.array([[0.5, 0.5, 0.5, 0.5]])
        vector = np.array([0.3, -0.2, 0.7])

        world = rotate_body_to_world(quat, vector)
        back = rotate_world_to_body(quat, world)

        np.testing.assert_allclose(back[0], vector, atol=1e-15)

    def test_angular_velocity_of_a_constant_spin_is_the_spin_rate(self) -> None:
        """A 10 rad/s spin about +z reports omega = (0, 0, 10)."""
        time_s = np.linspace(0.0, 0.01, 51)

        omega = angular_velocity_world_radps(time_s, _spin_quaternions(time_s, 10.0))

        np.testing.assert_allclose(omega, np.tile([0.0, 0.0, 10.0], (51, 1)), atol=1e-5)

    def test_a_sign_flipped_quaternion_sequence_does_not_spike(self) -> None:
        """q and -q are the same rotation; a raw derivative across a flip is not.

        Negating the second half of a constant-rate spin must leave the reported
        angular velocity unchanged.
        """
        time_s = np.linspace(0.0, 0.01, 51)
        quaternions = _spin_quaternions(time_s, 10.0)
        flipped = quaternions.copy()
        flipped[25:] *= -1.0

        omega = angular_velocity_world_radps(time_s, flipped)

        np.testing.assert_allclose(omega, np.tile([0.0, 0.0, 10.0], (51, 1)), atol=1e-5)


class TestStrikeTraceValidation:
    """A trace that cannot be trusted is refused at construction."""

    @pytest.fixture
    def parts(self):
        """A minimal valid set of trace arrays."""
        time_s = np.linspace(0.0, 0.01, 5)
        return {
            "time_s": time_s,
            "head_position_m": np.zeros((5, 3)),
            "head_orientation_quat": np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
            "sand_force_N": np.zeros((5, 3)),
            "sand_moment_Nm": np.zeros((5, 3)),
        }

    def test_a_two_sample_trace_is_refused(self, parts) -> None:
        """Second-order edge differences need three samples."""
        parts["time_s"] = np.array([0.0, 1.0])
        for name in ("head_position_m", "sand_force_N", "sand_moment_Nm"):
            parts[name] = np.zeros((2, 3))
        parts["head_orientation_quat"] = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))

        with pytest.raises(ValueError, match="at least 3 samples"):
            StrikeTrace(**parts)

    def test_time_must_increase(self, parts) -> None:
        """A repeated or reversed sample time makes every rate meaningless."""
        parts["time_s"] = np.array([0.0, 0.002, 0.001, 0.003, 0.004])

        with pytest.raises(ValueError, match="strictly increasing"):
            StrikeTrace(**parts)

    def test_a_nan_is_refused_by_a_raise_not_an_assert(self, parts) -> None:
        """python -O strips asserts, so the NaN check is an explicit raise."""
        parts["sand_force_N"] = parts["sand_force_N"].copy()
        parts["sand_force_N"][2, 0] = np.nan

        with pytest.raises(ValueError, match="must be finite"):
            StrikeTrace(**parts)

    def test_a_non_unit_quaternion_is_refused(self, parts) -> None:
        """A non-unit quaternion silently scales every rotated vector."""
        parts["head_orientation_quat"] = parts["head_orientation_quat"] * 2.0

        with pytest.raises(ValueError, match="unit quaternions"):
            StrikeTrace(**parts)

    def test_a_wrong_width_is_refused(self, parts) -> None:
        """A (T, 2) position array is not a position array."""
        parts["head_position_m"] = np.zeros((5, 2))

        with pytest.raises(ValueError, match="head_position_m must have shape"):
            StrikeTrace(**parts)


class TestHeadModelValidation:
    """The head description, which the artifact does not carry."""

    def test_a_non_positive_definite_inertia_is_refused(self) -> None:
        """An inertia tensor with a non-positive eigenvalue is not a solid body."""
        with pytest.raises(ValueError, match="positive definite"):
            HeadModel(
                mass_kg=0.3,
                centre_of_mass_body_m=np.zeros(3),
                sole_reference_body_m=np.zeros(3),
                shaft_axis_body=np.array([0.0, 0.0, 1.0]),
                inertia_body_kg_m2=np.diag([1e-4, 1e-4, -1e-4]),
            )

    def test_an_asymmetric_inertia_is_refused(self) -> None:
        """An inertia tensor is symmetric by construction."""
        inertia = np.diag([1e-4, 2e-4, 3e-4])
        inertia[0, 1] = 1e-5

        with pytest.raises(ValueError, match="symmetric"):
            HeadModel(
                mass_kg=0.3,
                centre_of_mass_body_m=np.zeros(3),
                sole_reference_body_m=np.zeros(3),
                shaft_axis_body=np.array([0.0, 0.0, 1.0]),
                inertia_body_kg_m2=inertia,
            )

    def test_the_shaft_axis_is_normalised(self) -> None:
        """A caller may pass any non-zero length; the stored axis is a unit vector."""
        head = HeadModel(
            mass_kg=0.3,
            centre_of_mass_body_m=np.zeros(3),
            sole_reference_body_m=np.zeros(3),
            shaft_axis_body=np.array([0.0, 0.0, 5.0]),
        )

        np.testing.assert_allclose(head.shaft_axis_body, [0.0, 0.0, 1.0])

    def test_a_zero_shaft_axis_is_refused(self) -> None:
        """A zero vector has no direction to resolve a moment onto."""
        with pytest.raises(ValueError, match="must have a direction"):
            HeadModel(
                mass_kg=0.3,
                centre_of_mass_body_m=np.zeros(3),
                sole_reference_body_m=np.zeros(3),
                shaft_axis_body=np.zeros(3),
            )

    def test_shaft_axis_inertia_is_not_guessed(self) -> None:
        """Without a tensor the moment of inertia is unknown, and it says so."""
        head = HeadModel(
            mass_kg=0.3,
            centre_of_mass_body_m=np.zeros(3),
            sole_reference_body_m=np.zeros(3),
            shaft_axis_body=np.array([0.0, 0.0, 1.0]),
        )

        with pytest.raises(ValueError, match="was not supplied"):
            head.shaft_axis_moment_of_inertia()

    def test_shaft_axis_inertia_is_the_quadratic_form(self) -> None:
        """a . I . a for a = +z and I = diag(2, 3, 4) e-4 is 4e-4."""
        head = reference_head()

        assert head.shaft_axis_moment_of_inertia() == pytest.approx(4.0e-4)
        assert head.shaft_axis_moment_of_inertia(
            np.array([1.0, 0.0, 0.0])
        ) == pytest.approx(2.0e-4)


class TestStrikeScene:
    """Where the strike happened."""

    def test_depth_is_positive_below_the_surface(self) -> None:
        """A surface at z = 0.5 m puts a point at z = 0.4 m at 0.1 m depth."""
        scene = StrikeScene(
            sand_surface_height_m=0.5,
            ball_position_m=np.zeros(3),
            travel_axis=np.array([1.0, 0.0, 0.0]),
        )

        assert scene.depth_m(np.array([0.0, 0.0, 0.4])) == pytest.approx(0.1)
        assert scene.depth_m(np.array([0.0, 0.0, 0.6])) == pytest.approx(-0.1)

    def test_travel_is_measured_from_the_ball(self) -> None:
        """Behind the ball is negative, past it positive."""
        scene = StrikeScene(
            sand_surface_height_m=0.0,
            ball_position_m=np.array([1.0, 2.0, 0.0]),
            travel_axis=np.array([1.0, 0.0, 0.0]),
        )

        assert scene.along_travel_m(np.array([0.9, 2.0, 0.0])) == pytest.approx(-0.1)
        assert scene.along_travel_m(np.array([1.3, 2.0, 0.0])) == pytest.approx(0.3)

    def test_a_tilted_travel_axis_is_refused(self) -> None:
        """The vertical component is the attack angle, which belongs to the trace."""
        with pytest.raises(ValueError, match="must be horizontal"):
            StrikeScene(
                sand_surface_height_m=0.0,
                ball_position_m=np.zeros(3),
                travel_axis=np.array([1.0, 0.0, 0.1]),
            )


class TestResultArtifactRoundTrip:
    """The metrics read the same artifact every fidelity tier writes."""

    def test_a_written_result_reads_back_as_a_strike_trace(self, tmp_path) -> None:
        """Schema v2 in, strike trace out, arrays unchanged."""
        source = build_decelerating_trace()
        path = tmp_path / "strike.h5"
        with BunkerShotResultWriter(path) as writer:
            for index, time_s in enumerate(source.time_s):
                writer.write_clubhead_state(
                    float(time_s),
                    source.head_position_m[index],
                    source.head_orientation_quat[index],
                )
                writer.write_contact_wrench(
                    float(time_s),
                    source.sand_force_N[index],
                    source.sand_moment_Nm[index],
                )

        trace = StrikeTrace.from_result_file(path)

        np.testing.assert_allclose(trace.time_s, source.time_s)
        np.testing.assert_allclose(trace.head_position_m, source.head_position_m)
        np.testing.assert_allclose(trace.sand_force_N, source.sand_force_N)

    def test_streams_on_different_time_bases_are_refused(self, tmp_path) -> None:
        """A force cannot be attributed to a pose recorded at another instant."""
        source = build_decelerating_trace(n_samples=5)
        path = tmp_path / "mismatched.h5"
        with BunkerShotResultWriter(path) as writer:
            for index, time_s in enumerate(source.time_s):
                writer.write_clubhead_state(
                    float(time_s),
                    source.head_position_m[index],
                    source.head_orientation_quat[index],
                )
                writer.write_contact_wrench(
                    float(time_s) + 1.0e-5,
                    source.sand_force_N[index],
                    source.sand_moment_Nm[index],
                )

        with pytest.raises(ValueError, match="do not share a time base"):
            StrikeTrace.from_result_file(path)
