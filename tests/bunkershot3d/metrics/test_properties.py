"""Property-based tests for the designer metrics (issue #8614, W7).

Each metric here has a natural invariant or bound that must hold for *any*
input, not just the hand-built ones: a fraction that lives in [0, 1], an area
that cannot exceed its domain, a correlation that cannot exceed one in
magnitude, a triangular divot whose section area is exactly half base times
height. Hypothesis explores the space; the assertions are the physics.

``deadline=None`` is mandatory for numeric work (the digest says so), and
``allow_subnormal=False`` keeps subnormals -- which diverge across platforms and
flake -- out of the generated floats.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bunkershot3d.metrics import (
    PlayabilityAxis,
    SoleLoadTrace,
    StrikeTrace,
    angular_velocity_world_radps,
    bounce_utilisation,
    divot_metrics,
    energy_partition,
    forgiveness_sensitivity,
    playability_window,
    rotate_body_to_world,
)

from .conftest import (
    DT_S,
    SAND_FRICTION_ANGLE_DEG,
    VEE_DX_M,
    build_decelerating_trace,
    build_trace,
    reference_head,
    reference_scene,
)

pytestmark = pytest.mark.unit

SETTINGS = settings(deadline=None, max_examples=50)


def _finite_floats(low: float, high: float):
    """Return a strategy for well-behaved floats in ``[low, high]``."""
    return st.floats(
        min_value=low,
        max_value=high,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        width=64,
    )


def _unit_vectors():
    """Return a strategy for unit 3-vectors."""
    return (
        st.lists(_finite_floats(-1.0, 1.0), min_size=3, max_size=3)
        .map(np.array)
        .filter(lambda vector: float(np.linalg.norm(vector)) > 1e-3)
        .map(lambda vector: vector / np.linalg.norm(vector))
    )


class TestRotationProperties:
    """Quaternion rotation is an isometry, whatever the quaternion."""

    @given(
        axis=_unit_vectors(),
        angle_rad=_finite_floats(-np.pi, np.pi),
        vector=_unit_vectors(),
    )
    @SETTINGS
    def test_rotation_preserves_length(
        self, axis: np.ndarray, angle_rad: float, vector: np.ndarray
    ) -> None:
        """A rotation cannot lengthen or shorten what it rotates."""
        quat = np.concatenate(
            [[np.cos(0.5 * angle_rad)], np.sin(0.5 * angle_rad) * axis]
        ).reshape(1, 4)

        rotated = rotate_body_to_world(quat, vector)

        assert float(np.linalg.norm(rotated[0])) == pytest.approx(
            float(np.linalg.norm(vector)), rel=1e-12
        )

    @given(axis=_unit_vectors(), rate_radps=_finite_floats(-50.0, 50.0))
    @SETTINGS
    def test_a_constant_spin_reports_its_own_rate(
        self, axis: np.ndarray, rate_radps: float
    ) -> None:
        """omega = rate * axis, for any axis.

        The centred difference is exact for a quadratic but a spin is
        trigonometric, so the residual truncation error is O((omega h)^2) -- at
        50 rad/s and h = 0.25 ms that is 2.6e-5 relative, which sets the
        tolerance here.
        """
        time_s = np.linspace(0.0, 0.01, 41)
        half = 0.5 * rate_radps * time_s
        quaternions = np.column_stack([np.cos(half), np.outer(np.sin(half), axis)])

        omega = angular_velocity_world_radps(time_s, quaternions)

        np.testing.assert_allclose(
            omega[len(omega) // 2], rate_radps * axis, atol=1e-6, rtol=1e-3
        )


class TestDivotProperties:
    """A triangular divot has an exactly known area, at any scale."""

    @given(
        n_descend=st.integers(min_value=5, max_value=60),
        n_ascend=st.integers(min_value=5, max_value=60),
        apex_depth_m=_finite_floats(0.005, 0.060),
        width_m=_finite_floats(0.005, 0.030),
        density_kg_m3=_finite_floats(1400.0, 1750.0),
    )
    @SETTINGS
    def test_a_triangular_divot_has_half_base_times_height(
        self,
        n_descend: int,
        n_ascend: int,
        apex_depth_m: float,
        width_m: float,
        density_kg_m3: float,
    ) -> None:
        """Section area = 0.5 * length * max depth, and mass follows from it."""
        head = reference_head()
        scene = reference_scene()
        n_pre, n_post = 4, 4
        count = n_pre + n_descend + n_ascend + n_post + 1
        entry_station_m = -0.120
        station = entry_station_m + VEE_DX_M * (np.arange(count) - n_pre)
        apex_station_m = entry_station_m + VEE_DX_M * n_descend
        descent = apex_depth_m / (n_descend * VEE_DX_M)
        ascent = apex_depth_m / (n_ascend * VEE_DX_M)
        depth = np.where(
            station <= apex_station_m,
            descent * (station - entry_station_m),
            apex_depth_m - ascent * (station - apex_station_m),
        )
        trace = build_trace(
            sole_path_m=np.column_stack([station, np.zeros(count), -depth]),
            time_s=DT_S * np.arange(count),
            head=head,
        )

        metrics = divot_metrics(
            trace,
            head,
            scene,
            width_m=width_m,
            bulk_density_kg_m3=density_kg_m3,
            friction_angle_deg=SAND_FRICTION_ANGLE_DEG,
        )

        expected_length_m = (n_descend + n_ascend) * VEE_DX_M
        assert metrics.length_m == pytest.approx(expected_length_m, rel=1e-9)
        assert metrics.max_depth_m == pytest.approx(apex_depth_m, rel=1e-9)
        assert metrics.section_area_m2 == pytest.approx(
            0.5 * expected_length_m * apex_depth_m, rel=1e-9
        )
        # The section can never exceed its bounding rectangle.
        assert metrics.section_area_m2 <= expected_length_m * apex_depth_m + 1e-15
        assert metrics.mass_kg == pytest.approx(
            metrics.volume_m3 * density_kg_m3, rel=1e-12
        )


class TestEnergyProperties:
    """The partition closes for any consistent deceleration."""

    @given(
        entry_speed_mps=_finite_floats(16.0, 30.0),
        speed_drop_mps=_finite_floats(1.0, 12.0),
        duration_s=_finite_floats(0.004, 0.020),
    )
    @SETTINGS
    def test_fractions_sum_to_one_and_the_residual_vanishes(
        self, entry_speed_mps: float, speed_drop_mps: float, duration_s: float
    ) -> None:
        """Force and motion are consistent, so nothing is unaccounted for."""
        head = reference_head()
        trace = build_decelerating_trace(
            head=head,
            entry_speed_mps=entry_speed_mps,
            exit_speed_mps=entry_speed_mps - speed_drop_mps,
            duration_s=duration_s,
        )

        partition = energy_partition(trace, head)

        assert partition.fraction_sum == pytest.approx(1.0, abs=1e-12)
        assert abs(partition.residual_fraction) < 1e-9


class TestPlayabilityProperties:
    """Area is bounded by its domain and monotone in the tolerance."""

    @given(
        carry_values=st.lists(_finite_floats(10.0, 50.0), min_size=15, max_size=15),
        tolerance=_finite_floats(0.02, 0.30),
    )
    @SETTINGS
    def test_area_is_bounded_and_connectivity_never_exceeds_it(
        self, carry_values: list[float], tolerance: float
    ) -> None:
        """0 <= largest connected <= area <= domain, always."""
        axis_a = PlayabilityAxis(name="a", unit="m", values=np.linspace(0.0, 4.0, 5))
        axis_b = PlayabilityAxis(name="b", unit="rad", values=np.linspace(0.0, 1.0, 3))
        carry = np.array(carry_values).reshape(5, 3)

        window = playability_window(
            axis_a,
            axis_b,
            carry,
            target_carry_m=30.0,
            tolerance_fraction=tolerance,
        )

        assert 0.0 <= window.largest_connected_area <= window.area + 1e-12
        assert window.area <= window.domain_area + 1e-12
        assert 0.0 <= window.fraction <= 1.0 + 1e-12

    @given(
        carry_values=st.lists(_finite_floats(10.0, 50.0), min_size=15, max_size=15),
        narrow=_finite_floats(0.02, 0.20),
        widening=_finite_floats(0.01, 0.30),
    )
    @SETTINGS
    def test_a_wider_tolerance_never_shrinks_the_window(
        self, carry_values: list[float], narrow: float, widening: float
    ) -> None:
        """Acceptance is a band, so widening it can only add points."""
        axis_a = PlayabilityAxis(name="a", unit="m", values=np.linspace(0.0, 4.0, 5))
        axis_b = PlayabilityAxis(name="b", unit="rad", values=np.linspace(0.0, 1.0, 3))
        carry = np.array(carry_values).reshape(5, 3)

        tight = playability_window(
            axis_a, axis_b, carry, target_carry_m=30.0, tolerance_fraction=narrow
        )
        loose = playability_window(
            axis_a,
            axis_b,
            carry,
            target_carry_m=30.0,
            tolerance_fraction=min(narrow + widening, 1.0),
        )

        assert loose.area >= tight.area - 1e-12


class TestBounceUtilisationProperties:
    """Areas partition, and the map is invariant to overall load scale."""

    @given(
        forces_N=st.lists(_finite_floats(0.0, 500.0), min_size=3, max_size=8).filter(
            lambda values: max(values) > 1.0
        ),
        areas_cm2=st.lists(_finite_floats(0.5, 4.0), min_size=3, max_size=8),
        scale=_finite_floats(0.1, 50.0),
    )
    @SETTINGS
    def test_areas_partition_and_the_map_is_scale_free(
        self, forces_N: list[float], areas_cm2: list[float], scale: float
    ) -> None:
        """Utilised + removable = total, and scaling every load changes nothing."""
        count = min(len(forces_N), len(areas_cm2))
        forces = np.array(forces_N[:count])
        areas = np.array(areas_cm2[:count]) * 1.0e-4
        centroids = np.column_stack(
            [np.linspace(-0.01, 0.01, count), np.zeros(count), np.zeros(count)]
        )
        time_s = np.linspace(0.0, 0.01, 3)
        load = SoleLoadTrace(
            time_s=time_s,
            element_centroid_body_m=centroids,
            element_area_m2=areas,
            element_normal_force_N=np.tile(forces, (3, 1)),
        )
        scaled = SoleLoadTrace(
            time_s=time_s,
            element_centroid_body_m=centroids,
            element_area_m2=areas,
            element_normal_force_N=np.tile(forces * scale, (3, 1)),
        )

        utilisation = bounce_utilisation(load)
        rescaled = bounce_utilisation(scaled)

        assert utilisation.utilised_area_m2 + utilisation.removable_area_m2 == (
            pytest.approx(utilisation.total_area_m2, rel=1e-12)
        )
        assert 0.0 < utilisation.utilisation_fraction <= 1.0
        np.testing.assert_array_equal(rescaled.loaded_mask, utilisation.loaded_mask)
        assert rescaled.total_impulse_Ns == pytest.approx(
            scale * utilisation.total_impulse_Ns, rel=1e-9
        )
        # The centre of pressure is a weighted mean, so it stays inside the hull
        # of the centroids, up to the rounding of the weighted sum.
        assert (
            centroids[:, 0].min() - 1e-12
            <= utilisation.centre_of_pressure_body_m[0]
            <= centroids[:, 0].max() + 1e-12
        )


class TestForgivenessProperties:
    """Correlation is bounded, and it does not care about the factor's units."""

    @given(
        carry_values=st.lists(_finite_floats(10.0, 50.0), min_size=4, max_size=12),
        scale=_finite_floats(0.1, 1000.0),
        shift=_finite_floats(-10.0, 10.0),
    )
    @SETTINGS
    def test_correlation_is_bounded_and_unit_agnostic(
        self, carry_values: list[float], scale: float, shift: float
    ) -> None:
        """|r| <= 1, and rescaling the factor scales the slope by 1/scale.

        This is the metamorphic statement of "r is dimensionless": measuring
        entry distance in millimetres rather than metres must not change how
        forgiving the design is judged to be.
        """
        carry_m = np.array(carry_values)
        # A constant carry has no correlation at all; that case is covered by an
        # explicit unit test rather than being silently accepted here.
        assume(float(np.ptp(carry_m)) > 1e-9)
        factor = np.linspace(0.025, 0.150, carry_m.size)

        base = forgiveness_sensitivity(
            factor, carry_m, factor="entry_distance_behind_ball_m", target_carry_m=30.0
        )
        rescaled = forgiveness_sensitivity(
            scale * factor + shift,
            carry_m,
            factor="entry_distance_behind_ball_m",
            target_carry_m=30.0,
        )

        assert abs(base.correlation_r) <= 1.0 + 1e-12
        assert rescaled.correlation_r == pytest.approx(base.correlation_r, abs=1e-9)
        assert rescaled.slope_m_per_unit == pytest.approx(
            base.slope_m_per_unit / scale, rel=1e-6
        )
        assert rescaled.carry_change_over_span_m == pytest.approx(
            base.carry_change_over_span_m, rel=1e-6
        )


@given(offset_m=st.lists(_finite_floats(-5.0, 5.0), min_size=3, max_size=3))
@SETTINGS
def test_translating_a_trace_leaves_the_wrench_untouched(offset_m: list[float]) -> None:
    """Translation moves the head and nothing else -- the force is the same force."""
    trace = build_decelerating_trace(moment_Nm=np.array([0.0, 0.0, 2.0]))

    moved = trace.translated(np.array(offset_m))

    assert isinstance(moved, StrikeTrace)
    np.testing.assert_array_equal(moved.sand_force_N, trace.sand_force_N)
    np.testing.assert_array_equal(moved.sand_moment_Nm, trace.sand_moment_Nm)
    np.testing.assert_allclose(
        moved.head_position_m - trace.head_position_m,
        np.tile(offset_m, (trace.n_samples, 1)),
        atol=1e-12,
    )
