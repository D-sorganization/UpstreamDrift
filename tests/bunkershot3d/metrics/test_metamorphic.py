"""Metamorphic relations on the designer metrics (issue #8614, W7).

A full bunker shot has no analytic oracle, so the research digest's answer is
metamorphic testing: transform the input in a way whose effect on the output is
known exactly, and assert it. Two relations are exercised here.

**Translation.** Move the trace *and* the scene by the same offset and every
metric must be unchanged. Depth is measured from the sand surface and travel
from the ball, both of which move with the strike, so a metric that leaked an
absolute world coordinate fails immediately.

**Rotation about the vertical.** Rotate the whole configuration a quarter turn
about world +z -- positions, wrench, orientations, travel axis and ball -- and
every *scalar* metric must be unchanged. This is the relation that catches
axis-swapped indices, which a translation cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import (
    StrikeScene,
    StrikeTrace,
    bounce_utilisation,
    dig_vs_skid,
    divot_metrics,
    energy_partition,
    head_load_metrics,
    head_twist_metrics,
    playability_window,
)

from .conftest import (
    DIVOT_WIDTH_M,
    SAND_BULK_DENSITY_KG_M3,
    build_decelerating_trace,
    build_sole_load_trace,
    build_vee_trace,
)
from .test_playability import BAD_CARRY_M, GOOD_CARRY_M, TARGET_CARRY_M

pytestmark = pytest.mark.unit

#: Offsets to translate by: a small power-of-two-friendly one, a metre-scale one,
#: and one far enough away to expose any absolute-coordinate leak.
OFFSETS_M = [
    np.array([0.25, -0.5, 0.125]),
    np.array([2.0, 4.0, -1.0]),
    np.array([100.0, -50.0, 25.0]),
]

#: Quarter turn about world +z, as an exact matrix and its quaternion.
QUARTER_TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
QUARTER_TURN_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``left (x) right`` for scalar-first rows."""
    scalar_l, vector_l = left[..., 0:1], left[..., 1:4]
    scalar_r, vector_r = right[..., 0:1], right[..., 1:4]
    scalar = scalar_l * scalar_r - np.sum(vector_l * vector_r, axis=-1, keepdims=True)
    vector = scalar_l * vector_r + scalar_r * vector_l + np.cross(vector_l, vector_r)
    return np.concatenate([scalar, vector], axis=-1)


def _rotated_configuration(
    trace: StrikeTrace, scene: StrikeScene
) -> tuple[StrikeTrace, StrikeScene]:
    """Return the configuration rotated a quarter turn about world +z."""
    rotated_trace = StrikeTrace(
        time_s=trace.time_s,
        head_position_m=trace.head_position_m @ QUARTER_TURN.T,
        head_orientation_quat=_quaternion_product(
            QUARTER_TURN_QUAT, trace.head_orientation_quat
        ),
        sand_force_N=trace.sand_force_N @ QUARTER_TURN.T,
        sand_moment_Nm=trace.sand_moment_Nm @ QUARTER_TURN.T,
        moment_reference=trace.moment_reference,
    )
    rotated_scene = StrikeScene(
        sand_surface_height_m=scene.sand_surface_height_m,
        ball_position_m=QUARTER_TURN @ scene.ball_position_m,
        travel_axis=QUARTER_TURN @ scene.travel_axis,
    )
    return rotated_trace, rotated_scene


def _divot_scalars(trace: StrikeTrace, head, scene: StrikeScene) -> dict[str, float]:
    """Return the scalar divot and dig/skid metrics as a flat mapping."""
    divot = divot_metrics(
        trace,
        head,
        scene,
        width_m=DIVOT_WIDTH_M,
        bulk_density_kg_m3=SAND_BULK_DENSITY_KG_M3,
    )
    dig = dig_vs_skid(trace, head, scene)
    return {
        "entry_distance": divot.entry_distance_behind_ball_m,
        "max_depth": divot.max_depth_m,
        "max_depth_behind_ball": divot.max_depth_behind_ball_m,
        "exit_distance": divot.exit_distance_past_ball_m,
        "length": divot.length_m,
        "section_area": divot.section_area_m2,
        "volume": divot.volume_m3,
        "mass": divot.mass_kg,
        "duration": divot.submerged_duration_s,
        "entry_slope": dig.entry_penetration_slope,
        "incoming_slope": dig.incoming_path_slope,
        "slope_ratio": dig.slope_ratio,
        "sand_impulse": dig.vertical_sand_impulse_Ns,
        "momentum_change": dig.measured_vertical_momentum_change_Ns,
        "constraint_impulse": dig.constraint_vertical_impulse_Ns,
    }


def _load_scalars(trace: StrikeTrace, head, scene: StrikeScene) -> dict[str, float]:
    """Return the scalar load, twist and energy metrics as a flat mapping."""
    loads = head_load_metrics(trace, head)
    twist = head_twist_metrics(trace, head, scene)
    energy = energy_partition(trace, head)
    return {
        "peak_deceleration": loads.peak_deceleration_mps2,
        "mean_deceleration": loads.mean_deceleration_mps2,
        "entry_speed": loads.entry_speed_mps,
        "exit_speed": loads.exit_speed_mps,
        "peak_force": loads.peak_resultant_force_N,
        "mean_force": loads.mean_resultant_force_N,
        "peak_moment": loads.peak_resultant_moment_Nm,
        "shaft_moment": twist.peak_shaft_axis_moment_Nm,
        "travel_moment": twist.peak_travel_axis_moment_Nm,
        "loft_moment": twist.peak_loft_axis_moment_Nm,
        "angular_impulse": twist.shaft_axis_angular_impulse_Nms,
        "free_rotation": twist.free_face_rotation_rad,
        "ke_loss": energy.club_kinetic_energy_loss_J,
        "work_on_sand": energy.work_on_sand_J,
        "residual": energy.residual_J,
    }


def _assert_same(
    expected: dict[str, float],
    actual: dict[str, float],
    *,
    rel: float,
    absolute: float = 1.0e-10,
) -> None:
    """Assert two metric mappings agree key by key.

    Args:
        expected: Metrics of the untransformed configuration.
        actual: Metrics of the transformed one.
        rel: Relative tolerance.
        absolute: Absolute floor, for metrics that are legitimately near zero.
    """
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=rel, abs=absolute), key


def _translation_floor(offset_m: np.ndarray) -> float:
    """Return an absolute tolerance that admits differencing far-away coordinates.

    Velocity is differenced from positions, so translating a 0.1 m strike to
    100 m from the origin costs about ``eps * |offset| / dt`` of velocity
    resolution -- roughly 1e-10 m/s here. That propagates into the near-zero
    energy residual. It is float cancellation, not a leak of absolute
    coordinates: every metric with a magnitude of its own still matches to
    ``rel``.

    Args:
        offset_m: The translation applied.

    Returns:
        The absolute tolerance to allow.
    """
    return 1.0e-10 * (1.0 + float(np.linalg.norm(offset_m)))


class TestTranslationInvariance:
    """Moving the whole strike leaves every metric alone."""

    @pytest.mark.parametrize("offset_m", OFFSETS_M)
    def test_divot_metrics_are_translation_invariant(
        self, head, scene, offset_m
    ) -> None:
        """Depth is measured from the surface and travel from the ball."""
        trace = build_vee_trace(force_N=np.array([0.0, 0.0, 500.0]))

        _assert_same(
            _divot_scalars(trace, head, scene),
            _divot_scalars(
                trace.translated(offset_m), head, scene.translated(offset_m)
            ),
            rel=1e-6,
            absolute=_translation_floor(offset_m),
        )

    @pytest.mark.parametrize("offset_m", OFFSETS_M)
    def test_load_and_energy_metrics_are_translation_invariant(
        self, head, scene, offset_m
    ) -> None:
        """Forces, rates and energies do not know where the strike happened."""
        trace = build_decelerating_trace(moment_Nm=np.array([0.0, 0.0, 2.0]))

        _assert_same(
            _load_scalars(trace, head, scene),
            _load_scalars(trace.translated(offset_m), head, scene.translated(offset_m)),
            rel=1e-6,
            absolute=_translation_floor(offset_m),
        )

    def test_translating_only_the_trace_does_change_the_divot(
        self, head, scene
    ) -> None:
        """The relation is about the *configuration*, and the test proves it bites.

        Lifting the trace without lifting the sand surface must change the
        divot -- otherwise the invariance above would be vacuous.
        """
        trace = build_vee_trace()
        lifted = trace.translated(np.array([0.0, 0.0, 0.005]))

        original = _divot_scalars(trace, head, scene)
        shallower = _divot_scalars(lifted, head, scene)

        assert shallower["max_depth"] < original["max_depth"]


class TestRotationInvariance:
    """A quarter turn about the vertical leaves every scalar alone."""

    def test_divot_metrics_survive_a_quarter_turn(self, head, scene) -> None:
        """Rotating the scene with the trace cannot change a length or a slope."""
        trace = build_vee_trace(force_N=np.array([0.0, 0.0, 500.0]))
        rotated_trace, rotated_scene = _rotated_configuration(trace, scene)

        _assert_same(
            _divot_scalars(trace, head, scene),
            _divot_scalars(rotated_trace, head, rotated_scene),
            rel=1e-9,
        )

    def test_load_and_twist_metrics_survive_a_quarter_turn(self, head, scene) -> None:
        """The twist triad rotates with the head, so its components are unchanged."""
        trace = build_decelerating_trace(moment_Nm=np.array([0.0, 0.0, 2.0]))
        rotated_trace, rotated_scene = _rotated_configuration(trace, scene)

        _assert_same(
            _load_scalars(trace, head, scene),
            _load_scalars(rotated_trace, head, rotated_scene),
            rel=1e-9,
        )


class TestOtherMetricInvariances:
    """The two metrics that do not consume a strike trace."""

    def test_moving_the_sole_elements_moves_only_the_centre_of_pressure(self) -> None:
        """Translating the element centroids translates the CoP identically."""
        load = build_sole_load_trace()
        offset_m = np.array([0.003, -0.007, 0.001])
        moved = type(load)(
            time_s=load.time_s,
            element_centroid_body_m=load.element_centroid_body_m + offset_m,
            element_area_m2=load.element_area_m2,
            element_normal_force_N=load.element_normal_force_N,
        )

        original = bounce_utilisation(load)
        shifted = bounce_utilisation(moved)

        np.testing.assert_allclose(
            shifted.centre_of_pressure_body_m,
            original.centre_of_pressure_body_m + offset_m,
            atol=1e-12,
        )
        assert shifted.utilised_area_m2 == pytest.approx(original.utilised_area_m2)

    def test_scaling_every_element_load_leaves_the_map_unchanged(self) -> None:
        """The loaded threshold is a fraction of the peak, so it is scale-free."""
        load = build_sole_load_trace()
        scaled = build_sole_load_trace(forces_N=np.array([100.0, 0.0, 50.0, 0.0]) * 7.0)

        original = bounce_utilisation(load)
        louder = bounce_utilisation(scaled)

        np.testing.assert_array_equal(louder.loaded_mask, original.loaded_mask)
        assert louder.utilisation_fraction == pytest.approx(
            original.utilisation_fraction
        )
        np.testing.assert_allclose(
            louder.centre_of_pressure_body_m,
            original.centre_of_pressure_body_m,
            atol=1e-12,
        )

    def test_shifting_the_playability_axes_leaves_the_area_unchanged(self) -> None:
        """The window is an area, so it does not depend on where the axes start."""
        from bunkershot3d.metrics import PlayabilityAxis

        values_a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values_b = np.array([0.0, 0.5, 1.0])
        carry = np.full((5, 3), BAD_CARRY_M)
        carry[1:4, 1] = GOOD_CARRY_M

        original = playability_window(
            PlayabilityAxis(name="a", unit="m", values=values_a),
            PlayabilityAxis(name="b", unit="rad", values=values_b),
            carry,
            target_carry_m=TARGET_CARRY_M,
        )
        shifted = playability_window(
            PlayabilityAxis(name="a", unit="m", values=values_a + 12.0),
            PlayabilityAxis(name="b", unit="rad", values=values_b - 3.0),
            carry,
            target_carry_m=TARGET_CARRY_M,
        )

        assert shifted.area == pytest.approx(original.area, rel=1e-12)
        assert shifted.fraction == pytest.approx(original.fraction, rel=1e-12)
