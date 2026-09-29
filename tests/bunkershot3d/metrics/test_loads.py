"""Head deceleration, peak loads and head twist, against hand arithmetic.

Issue #8614 (W7). The decelerating trace slows a 0.300 kg head 25 -> 15 m/s in
10 ms, so the deceleration is exactly ``10 / 0.010 = 1000 m/s^2`` at every
sample and the sand force is exactly ``0.3 * 1000 = 300 N``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import (
    STANDARD_GRAVITY_MPS2,
    HeadModel,
    StrikeTrace,
    WrenchReference,
    centre_of_mass_moment_Nm,
    head_load_metrics,
    head_twist_metrics,
    shaft_travel_loft_axes,
)

from .conftest import build_decelerating_trace, reference_head

pytestmark = pytest.mark.unit

#: Constant sand moment used for the twist tests: 2 N.m about world +z.
TWIST_MOMENT_NM = np.array([0.0, 0.0, 2.0])


class TestHeadLoadMetrics:
    """Deceleration and peak loads on the uniformly decelerating trace."""

    @pytest.fixture
    def metrics(self, decelerating_trace, head):
        """Load metrics for the reference decelerating trace."""
        return head_load_metrics(decelerating_trace, head)

    def test_deceleration_is_uniform_and_matches_the_speed_drop(self, metrics) -> None:
        """(25 - 15) m/s over 0.010 s is 1000 m/s^2, peak and mean alike."""
        assert metrics.peak_deceleration_mps2 == pytest.approx(1000.0, rel=1e-9)
        assert metrics.mean_deceleration_mps2 == pytest.approx(1000.0, rel=1e-9)
        assert metrics.entry_speed_mps == pytest.approx(25.0, rel=1e-9)
        assert metrics.exit_speed_mps == pytest.approx(15.0, rel=1e-9)

    def test_deceleration_in_g(self, metrics) -> None:
        """1000 / 9.80665 = 101.97 g -- the unit the smoke test is quoted in."""
        assert metrics.peak_deceleration_g == pytest.approx(
            1000.0 / STANDARD_GRAVITY_MPS2, rel=1e-9
        )
        assert metrics.peak_deceleration_g == pytest.approx(101.972, rel=1e-4)

    def test_force_peak_mean_and_impulse(self, metrics) -> None:
        """A constant 300 N for 10 ms is a 3.0 N.s impulse, backwards."""
        assert metrics.peak_resultant_force_N == pytest.approx(300.0, rel=1e-12)
        assert metrics.mean_resultant_force_N == pytest.approx(300.0, rel=1e-12)
        np.testing.assert_allclose(
            metrics.linear_impulse_Ns, [-3.0, 0.0, 0.0], atol=1e-12
        )

    def test_moment_peak_and_angular_impulse(self, head) -> None:
        """A constant 2 N.m for 10 ms is a 0.02 N.m.s angular impulse."""
        trace = build_decelerating_trace(moment_Nm=TWIST_MOMENT_NM)

        metrics = head_load_metrics(trace, head)

        assert metrics.peak_resultant_moment_Nm == pytest.approx(2.0, rel=1e-12)
        np.testing.assert_allclose(
            metrics.angular_impulse_Nms, [0.0, 0.0, 0.02], atol=1e-12
        )

    def test_a_two_sample_window_is_refused(self, decelerating_trace, head) -> None:
        """A second-order rate needs three samples; two would be quietly wrong."""
        with pytest.raises(ValueError, match="at least 3 samples"):
            head_load_metrics(decelerating_trace, head, window=slice(0, 2))


class TestMomentReference:
    """The transport term that finding B5b showed was missing."""

    def test_a_moment_about_the_head_origin_is_transported_to_the_cg(
        self, head
    ) -> None:
        """r_origin - r_cg = (0, 0, -0.010) m and F = (-300, 0, 0) N.

        The cross product is (0, 0, -0.010) x (-300, 0, 0) = (0, +3.0, 0) N.m,
        so the moment about the CG is 3 N.m larger about +y than the recorded
        moment about the origin. That term is the entire reason a sole digs.
        """
        origin_referenced = build_decelerating_trace(moment_Nm=TWIST_MOMENT_NM)
        trace = StrikeTrace(
            time_s=origin_referenced.time_s,
            head_position_m=origin_referenced.head_position_m,
            head_orientation_quat=origin_referenced.head_orientation_quat,
            sand_force_N=origin_referenced.sand_force_N,
            sand_moment_Nm=origin_referenced.sand_moment_Nm,
            moment_reference=WrenchReference.HEAD_ORIGIN,
        )

        moment = centre_of_mass_moment_Nm(trace, head)

        np.testing.assert_allclose(moment[0], [0.0, 3.0, 2.0], atol=1e-9)

    def test_a_cg_referenced_moment_is_passed_through(self, head) -> None:
        """The default reference needs no transport, so nothing is added."""
        trace = build_decelerating_trace(moment_Nm=TWIST_MOMENT_NM)

        moment = centre_of_mass_moment_Nm(trace, head)

        np.testing.assert_allclose(moment[0], TWIST_MOMENT_NM, atol=1e-12)


class TestTwistAxes:
    """The orthonormal triad the twist is resolved onto."""

    def test_the_triad_is_orthonormal_and_correctly_oriented(
        self, decelerating_trace, head, scene
    ) -> None:
        """With an unrotated head the triad is world (+z, +x, +y)."""
        shaft, travel, loft = shaft_travel_loft_axes(decelerating_trace, head, scene)

        np.testing.assert_allclose(shaft[0], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(travel[0], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(loft[0], [0.0, 1.0, 0.0], atol=1e-12)

    def test_a_shaft_axis_along_travel_is_refused(
        self, decelerating_trace, scene
    ) -> None:
        """Then the perpendicular component vanishes and the triad is undefined."""
        along_travel = HeadModel(
            mass_kg=0.300,
            centre_of_mass_body_m=np.zeros(3),
            sole_reference_body_m=np.array([0.0, 0.0, -0.020]),
            shaft_axis_body=np.array([1.0, 0.0, 0.0]),
        )

        with pytest.raises(ValueError, match="parallel to the travel axis"):
            shaft_travel_loft_axes(decelerating_trace, along_travel, scene)


class TestHeadTwistMetrics:
    """Head rotation under sand load -- the metric nobody has published."""

    @pytest.fixture
    def twist(self, head, scene):
        """Twist metrics for a constant 2 N.m about the (world +z) shaft axis."""
        trace = build_decelerating_trace(moment_Nm=TWIST_MOMENT_NM)
        return head_twist_metrics(trace, head, scene)

    def test_the_moment_resolves_entirely_onto_the_shaft_axis(self, twist) -> None:
        """The applied couple is about +z, which is exactly the shaft axis."""
        assert twist.peak_shaft_axis_moment_Nm == pytest.approx(2.0, rel=1e-12)
        assert twist.mean_shaft_axis_moment_Nm == pytest.approx(2.0, rel=1e-9)
        assert twist.peak_travel_axis_moment_Nm == pytest.approx(0.0, abs=1e-12)
        assert twist.peak_loft_axis_moment_Nm == pytest.approx(0.0, abs=1e-12)

    def test_angular_impulse_about_the_shaft(self, twist) -> None:
        """2 N.m held for 10 ms is 0.020 N.m.s."""
        assert twist.shaft_axis_angular_impulse_Nms == pytest.approx(0.020, rel=1e-9)

    def test_free_face_rotation_is_the_double_integral(self, twist) -> None:
        """Constant M/I gives theta = (M/I) T^2 / 2.

        I_shaft = 4.0e-4 kg.m^2, so M/I = 5000 rad/s^2, the rate reached is
        5000 * 0.010 = 50 rad/s and the rotation is 5000 * 1e-4 / 2 = 0.25 rad
        = 14.3239 deg. It is an upper bound: a gripped club is restrained.
        """
        assert twist.shaft_axis_inertia_kg_m2 == pytest.approx(4.0e-4, rel=1e-12)
        assert twist.free_face_rate_radps == pytest.approx(50.0, rel=1e-9)
        assert twist.free_face_rotation_rad == pytest.approx(0.25, rel=1e-9)
        assert twist.free_face_rotation_deg == pytest.approx(14.32394487, rel=1e-8)

    def test_rotation_is_not_invented_without_an_inertia_tensor(self, scene) -> None:
        """No inertia, no rotation figure -- the fields are None, not a guess."""
        reference = reference_head()
        head_without_inertia = HeadModel(
            mass_kg=reference.mass_kg,
            centre_of_mass_body_m=reference.centre_of_mass_body_m,
            sole_reference_body_m=reference.sole_reference_body_m,
            shaft_axis_body=reference.shaft_axis_body,
        )
        trace = build_decelerating_trace(
            head=head_without_inertia, moment_Nm=TWIST_MOMENT_NM
        )

        twist = head_twist_metrics(trace, head_without_inertia, scene)

        assert twist.shaft_axis_inertia_kg_m2 is None
        assert twist.free_face_rotation_rad is None
        assert twist.free_face_rotation_deg is None
        assert twist.shaft_axis_angular_impulse_Nms == pytest.approx(0.020, rel=1e-9)

    def test_the_sign_convention_matches_the_documented_worked_example(
        self, head, scene
    ) -> None:
        """Sand retarding the sole on the toe side of the shaft opens the face.

        Force (-300, 0, 0) N applied 0.020 m toward +y of the CG gives
        r x F = (0, 0.020, 0) x (-300, 0, 0) = (0, 0, +6.0) N.m: a positive
        shaft-axis moment, which the module documents as face-opening for a
        right-handed player travelling along +x.
        """
        lever_m = np.array([0.0, 0.020, 0.0])
        force_N = np.array([-300.0, 0.0, 0.0])
        trace = build_decelerating_trace(moment_Nm=np.cross(lever_m, force_N))

        twist = head_twist_metrics(trace, head, scene)

        assert twist.peak_shaft_axis_moment_Nm == pytest.approx(6.0, rel=1e-12)
        assert twist.peak_shaft_axis_moment_Nm > 0.0
