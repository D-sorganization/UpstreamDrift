"""Physical toy-oracle tests for progressive fitting, independent of MATLAB."""

import numpy as np
import pytest

from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    PrefixFitOptions,
    PrefixStage,
    bernstein_to_simscape,
    fit_prefixes,
    normalized_to_simscape,
    reexpress_bernstein_basis,
)


pytestmark = pytest.mark.unit


def test_reexpress_bernstein_basis_preserves_continuous_torque() -> None:
    """Re-expressing Bernstein controls to a different basis duration must preserve identical torque."""
    rng = np.random.default_rng(42)
    controls = rng.normal(size=(5, 7))
    t_source = 0.80
    t_target = 1.813889

    controls_target = reexpress_bernstein_basis(
        controls, source_duration_s=t_source, target_duration_s=t_target
    )

    # Evaluate physical torque polynomials
    coeff_source = bernstein_to_simscape(controls, duration_s=t_source)
    coeff_target = bernstein_to_simscape(controls_target, duration_s=t_target)

    t_eval = np.linspace(0.0, t_source, 100)
    for j in range(5):
        tau_src = np.polyval(coeff_source[j], t_eval)
        tau_tgt = np.polyval(coeff_target[j], t_eval)
        np.testing.assert_allclose(tau_tgt, tau_src, atol=1e-12, rtol=1e-12)


def test_low_degree_controls_export_continuous_torque_in_physical_time() -> None:
    clock = np.linspace(0, 0.2, 31)
    linear = bernstein_to_simscape(np.array([[3.0, 7.0]]), duration_s=0.2)
    np.testing.assert_allclose(np.polyval(linear[0], clock), 3 + 20 * clock)
    assert linear.shape == (1, 7)
    constant = bernstein_to_simscape(np.array([[3.0]]), duration_s=0.2)
    np.testing.assert_allclose(np.polyval(constant[0], clock), 3)


@pytest.mark.parametrize("controls", [np.empty((1, 0)), np.zeros((1, 8)), [[np.nan]]])
def test_control_degree_and_finiteness_are_checked(controls: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bernstein_to_simscape(controls, duration_s=0.2)


def test_resolved_difference_step_escapes_quantized_oracle_plateau() -> None:
    time = np.array([0.0, 1.0])
    points = np.zeros((2, 1, 3))
    points[1, 0, 0] = 0.4

    def forward(parameters: np.ndarray, requested: np.ndarray) -> np.ndarray:
        result = np.zeros((len(requested), 1, 3))
        result[:, 0, 0] = np.round(parameters[0], 4) * requested**2
        return result

    target = MarkerTarget(time, points, np.ones(1))
    assert not fit_prefixes(
        target,
        forward,
        initial=np.ones(1),
        lower=np.zeros(1),
        upper=2 * np.ones(1),
        prefix_end_s=(1.0,),
        acceptance_rmse_m=1e-4,
    ).accepted
    fitted = fit_prefixes(
        target,
        forward,
        initial=np.ones(1),
        lower=np.zeros(1),
        upper=2 * np.ones(1),
        prefix_end_s=(1.0,),
        acceptance_rmse_m=1e-4,
        options=PrefixFitOptions(finite_difference_step=0.01),
    )
    assert fitted.accepted
    assert fitted.parameters[0] == pytest.approx(0.4, abs=1e-4)


@pytest.mark.parametrize("step", [0, -0.1, np.nan, np.inf, True])
def test_rejects_unresolved_difference_step(step: float) -> None:
    target = MarkerTarget(np.array([0, 1]), np.zeros((2, 1, 3)), np.ones(1))
    with pytest.raises(ValueError, match="finite_difference_step"):
        fit_prefixes(
            target,
            lambda p, t: np.zeros((len(t), 1, 3)),
            initial=np.ones(1),
            lower=np.zeros(1),
            upper=2 * np.ones(1),
            prefix_end_s=(1.0,),
            acceptance_rmse_m=0.01,
            options=PrefixFitOptions(finite_difference_step=step),
        )


def test_bernstein_export_is_bounded_and_has_correct_endpoints() -> None:
    controls = np.array([[1, -2, 4, 3, -1, 5, 2]], dtype=float)
    native = bernstein_to_simscape(controls, duration_s=1.8)
    torque = np.polyval(native[0], np.linspace(0, 1.8, 301))
    assert torque[0] == pytest.approx(controls[0, 0])
    assert torque[-1] == pytest.approx(controls[0, -1])
    assert np.min(torque) >= np.min(controls) - 1e-12
    assert np.max(torque) <= np.max(controls) + 1e-12


def test_constant_bernstein_control_values_produce_constant_torque() -> None:
    result = bernstein_to_simscape(np.full((2, 7), 3.0), duration_s=2)
    np.testing.assert_allclose(result[:, :-1], 0, atol=1e-12)
    np.testing.assert_allclose(result[:, -1], 3)


def test_simscape_conversion_preserves_polynomial_in_seconds() -> None:
    ascending = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=float)
    descending = normalized_to_simscape(ascending, duration_s=2)
    for t in [0, 0.1, 1, 2]:
        assert np.polyval(descending[0], t) == pytest.approx(
            np.polynomial.polynomial.polyval(t / 2, ascending[0])
        )
    assert descending[0, -1] == 1


@pytest.mark.parametrize("duration", [0, -1, np.nan, np.inf])
def test_conversion_rejects_bad_duration(duration: float) -> None:
    with pytest.raises(ValueError):
        normalized_to_simscape(np.zeros((1, 7)), duration_s=duration)


def test_target_rejects_empty_observations_and_bad_clock() -> None:
    with pytest.raises(ValueError, match="observ"):
        MarkerTarget(np.array([0, 1]), np.full((2, 1, 3), np.nan), np.ones(1))
    with pytest.raises(ValueError, match="time"):
        MarkerTarget(np.array([1, 2]), np.zeros((2, 1, 3)), np.ones(1))


def test_prefix_fit_recovers_constant_torque_without_resetting_state() -> None:
    time = np.linspace(0, 1, 21)
    points = np.zeros((21, 1, 3))
    points[:, 0, 0] = 0.5 * 2 * time**2  # unit mass, force=2, x(0)=v(0)=0
    calls = []

    def forward(parameters: np.ndarray, requested_time: np.ndarray) -> np.ndarray:
        calls.append(requested_time.copy())
        result = np.zeros((len(requested_time), 1, 3))
        result[:, 0, 0] = 0.5 * parameters[0] * requested_time**2
        return result

    fit = fit_prefixes(
        MarkerTarget(time, points, np.ones(1)),
        forward,
        initial=np.array([0.0]),
        lower=np.array([-5.0]),
        upper=np.array([5.0]),
        prefix_end_s=(0.2, 0.5, 1.0),
        acceptance_rmse_m=1e-6,
    )
    assert fit.parameters[0] == pytest.approx(2, abs=1e-5)
    assert fit.accepted
    assert len(fit.stages) == 3
    assert all(t[0] == 0 for t in calls)
    assert fit.stages[-1].rmse_m < 1e-6


def test_missing_target_does_not_hide_missing_simulation() -> None:
    time = np.linspace(0, 1, 5)
    target = MarkerTarget(time, np.zeros((5, 1, 3)), np.ones(1))
    with pytest.raises(ValueError, match="finite"):
        fit_prefixes(
            target,
            lambda p, t: np.full((len(t), 1, 3), np.nan),
            initial=np.zeros(1),
            lower=-np.ones(1),
            upper=np.ones(1),
            prefix_end_s=(1.0,),
            acceptance_rmse_m=0.01,
        )


def test_unattainable_target_is_not_accepted() -> None:
    target = MarkerTarget(np.array([0, 1]), np.ones((2, 1, 3)), np.ones(1))
    fit = fit_prefixes(
        target,
        lambda p, t: np.zeros((len(t), 1, 3)),
        initial=np.zeros(1),
        lower=-np.ones(1),
        upper=np.ones(1),
        prefix_end_s=(1.0,),
        acceptance_rmse_m=0.01,
    )
    assert not fit.accepted
    assert fit.stages[0].rmse_m == pytest.approx(np.sqrt(3))


def test_schedule_must_cover_target_without_reordering() -> None:
    target = MarkerTarget(np.array([0, 0.5, 1]), np.zeros((3, 1, 3)), np.ones(1))
    with pytest.raises(ValueError, match="prefix"):
        fit_prefixes(
            target,
            lambda p, t: np.zeros((len(t), 1, 3)),
            initial=np.zeros(1),
            lower=-np.ones(1),
            upper=np.ones(1),
            prefix_end_s=(0.5,),
            acceptance_rmse_m=0.01,
        )


def test_checkpoint_snapshots_and_target_are_immutable() -> None:
    time = np.array([0, 0.5, 1])
    source = np.zeros((3, 1, 3))
    target = MarkerTarget(time, source, np.ones(1))
    source[:] = 99
    checkpoints: list[PrefixStage] = []
    fit = fit_prefixes(
        target,
        lambda p, t: np.zeros((len(t), 1, 3)),
        initial=np.zeros(1),
        lower=-np.ones(1),
        upper=np.ones(1),
        prefix_end_s=(0.5, 1.0),
        acceptance_rmse_m=0.01,
        options=PrefixFitOptions(checkpoint=checkpoints.append),
    )
    assert fit.accepted
    assert len(checkpoints) == 2
    assert np.all(target.points == 0)
    with pytest.raises(ValueError):
        checkpoints[0].parameters[0] = 7


def test_observed_target_mask_and_zero_weight_do_not_bias_fit() -> None:
    time = np.array([0, 0.5, 1])
    points = np.zeros((3, 2, 3))
    points[:, 0, 0] = 2
    points[1, 0] = np.nan
    points[:, 1] = 999

    def forward(parameters: np.ndarray, requested_time: np.ndarray) -> np.ndarray:
        result = np.zeros((len(requested_time), 2, 3))
        result[:, 0, 0] = parameters[0]
        return result

    fit = fit_prefixes(
        MarkerTarget(time, points, np.array([1, 0])),
        forward,
        initial=np.zeros(1),
        lower=-np.ones(1) * 5,
        upper=np.ones(1) * 5,
        prefix_end_s=(1.0,),
        acceptance_rmse_m=1e-6,
    )
    assert fit.accepted
    assert fit.parameters[0] == pytest.approx(2, abs=1e-6)


def test_bernstein_range_finds_interior_peak_not_control_bound() -> None:
    from src.shared.python.motion_matching.prefix_fit import bernstein_effort_range

    ranges = bernstein_effort_range(np.array([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]))
    np.testing.assert_allclose(ranges, [[0.0, 1.0], [-1.0, 0.0]], atol=1e-12)


def test_bernstein_range_includes_off_grid_stationary_point_and_endpoints() -> None:
    from src.shared.python.motion_matching.prefix_fit import bernstein_effort_range

    a = 0.1234567
    b = 1 - a * a
    result = bernstein_effort_range(np.array([[b, b + a, b + 2 * a - 1]]))
    np.testing.assert_allclose(result, [[1 - (1 - a) ** 2, 1.0]], atol=1e-12)
    np.testing.assert_allclose(
        bernstein_effort_range(np.array([[3.0], [-2.0]])), [[3.0, 3.0], [-2.0, -2.0]]
    )


def test_build_anatomical_marker_weights_hierarchy() -> None:
    from src.shared.python.motion_matching.prefix_fit import (
        build_anatomical_marker_weights,
    )

    labels = [
        "WaistLeft",
        "BackTop",
        "HeadTop",
        "LShoulderTop",
        "LElbowOut",
        "Marker_2:2:1",
        "CustomMarker",
    ]
    weights = build_anatomical_marker_weights(
        labels, custom_weights={"CustomMarker": 15.0}
    )
    assert weights[0] == 100.0  # Waist
    assert weights[1] == 80.0  # Back
    assert weights[2] == 40.0  # Head
    assert weights[3] == 20.0  # Shoulder
    assert weights[4] == 5.0  # Elbow
    assert weights[5] == 25.0  # Clubhead
    assert weights[6] == 15.0  # Custom override


def test_bernstein_curvature_regularizer_zero_on_linear_positive_on_curved() -> None:
    from src.shared.python.motion_matching.prefix_fit import (
        bernstein_curvature_regularizer,
    )

    # 4 control points (cubic): linear progression has zero second differences
    linear = np.array([1.0, 2.0, 3.0, 4.0])
    reg = bernstein_curvature_regularizer(control_count=4, weight=1.0)
    residuals = reg(linear)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    # Oscillating control points: [0, 2, 0, 2] -> high curvature
    oscillating = np.array([0.0, 2.0, 0.0, 2.0])
    curv_res = reg(oscillating)
    # Second differences: (0 - 4 + 0) = -4, (2 - 0 + 2) = 4
    assert len(curv_res) == 2
    np.testing.assert_allclose(np.abs(curv_res), 4.0)

    # Degree <= 1 (control_count <= 2) has zero curvature residuals
    reg_deg1 = bernstein_curvature_regularizer(control_count=2, weight=1.0)
    assert len(reg_deg1(np.array([1.0, 5.0]))) == 0


def test_prefix_fit_with_regularization_favors_smooth_solution() -> None:
    # 3 control points (quadratic), 1 coordinate. Time points at 0 and 1.
    # Target only observes displacement at t=1.
    # Forward model: x(1) = 0.5 * (p[0] + p[2]) (multiple solutions satisfy target)
    time = np.array([0.0, 1.0])
    points = np.zeros((2, 1, 3))
    points[1, 0, 0] = 1.0
    target = MarkerTarget(time, points, np.ones(1))

    def forward(parameters: np.ndarray, requested_time: np.ndarray) -> np.ndarray:
        result = np.zeros((len(requested_time), 1, 3))
        p = parameters
        result[:, 0, 0] = (0.5 * p[0] + 0.5 * p[2]) * requested_time
        return result

    # With curvature regularization, the optimizer should prefer p[1] to be collinear with p[0] and p[2]
    # (i.e. Delta^2 p = p[2] - 2*p[1] + p[0] == 0)
    from src.shared.python.motion_matching.prefix_fit import (
        bernstein_curvature_regularizer,
    )

    reg = bernstein_curvature_regularizer(control_count=3, weight=10.0)
    fit = fit_prefixes(
        target,
        forward,
        initial=np.array([0.0, 5.0, 0.0]),  # Initial has high curvature (p[1] = 5)
        lower=np.zeros(3),
        upper=5 * np.ones(3),
        prefix_end_s=(1.0,),
        acceptance_rmse_m=0.01,
        options=PrefixFitOptions(regularization=reg),
    )
    assert fit.accepted
    p = fit.parameters
    # Curvature should be minimized: p[2] - 2*p[1] + p[0] ~ 0
    assert abs(p[2] - 2 * p[1] + p[0]) < 1e-4


def test_prefix_fit_rejects_non_finite_regularization() -> None:
    time = np.array([0.0, 1.0])
    points = np.zeros((2, 1, 3))
    target = MarkerTarget(time, points, np.ones(1))

    with pytest.raises(ValueError, match="regularization"):
        fit_prefixes(
            target,
            lambda p, t: np.zeros((len(t), 1, 3)),
            initial=np.zeros(1),
            lower=-np.ones(1),
            upper=np.ones(1),
            prefix_end_s=(1.0,),
            acceptance_rmse_m=0.01,
            options=PrefixFitOptions(regularization=lambda p: np.array([np.nan])),
        )


def test_prefix_fit_pelvis_yaw_penalty_and_gate() -> None:
    # 2 markers: WaistLeft (idx 0) and WaistRight (idx 1)
    time = np.array([0.0, 1.0])
    target_yaw_deg = 60.0
    th_t = np.radians(target_yaw_deg)
    # Waist vector of length 0.28m oriented at target_yaw_deg
    wl_pos = np.array([-0.14 * np.cos(th_t), -0.14 * np.sin(th_t), 0.0])
    wr_pos = np.array([0.14 * np.cos(th_t), 0.14 * np.sin(th_t), 0.0])
    points = np.zeros((2, 2, 3))
    points[0, 0] = [-0.14, 0.0, 0.0]
    points[0, 1] = [0.14, 0.0, 0.0]
    points[1, 0] = wl_pos
    points[1, 1] = wr_pos

    target = MarkerTarget(time, points, np.ones(2))

    # Forward model: param is yaw angle in radians at t=1
    def forward(parameters: np.ndarray, requested: np.ndarray) -> np.ndarray:
        pred = np.zeros((len(requested), 2, 3))
        pred[0, 0] = [-0.14, 0.0, 0.0]
        pred[0, 1] = [0.14, 0.0, 0.0]
        if len(requested) > 1:
            th = float(parameters[0])
            pred[1, 0] = [-0.14 * np.cos(th), -0.14 * np.sin(th), 0.0]
            pred[1, 1] = [0.14 * np.cos(th), 0.14 * np.sin(th), 0.0]
        return pred

    # Start with an initial guess 10 degrees off (target 60 deg = ~1.047 rad, guess 50 deg = ~0.873 rad)
    guess_th = np.radians(50.0)
    fit = fit_prefixes(
        target,
        forward,
        initial=np.array([guess_th]),
        lower=np.zeros(1),
        upper=np.pi * np.ones(1),
        prefix_end_s=(1.0,),
        acceptance_rmse_m=0.01,
        options=PrefixFitOptions(
            pelvis_indices=(0, 1),
            pelvis_yaw_weight=50.0,
            pelvis_yaw_max_error_pct=5.0,
        ),
    )
    assert fit.accepted
    stage = fit.stages[0]
    assert stage.pelvis_yaw_error_pct < 5.0
    assert abs(stage.pelvis_yaw_diff_deg) < 1.0  # Well within 1 degree
