"""Tests for OpenSim degree-six polynomial effort profile and controller (OS-5).

Exercises pure-Python contracts, polynomial evaluation and derivatives,
least-squares polynomial fitting, Simscape/OpenSim coefficient conventions,
DbC preconditions, and JSON serialization.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.polynomial_profile import (
    Degree6PolynomialCoefficients,
    PolynomialTorqueProfile,
    fit_degree6_from_discrete_controls,
    check_effort_and_rate_bounds,
)


@pytest.mark.unit
def test_polynomial_coefficients_evaluation() -> None:
    """Evaluate polynomial and its time derivative with descending powers."""
    # f(t) = 3*t^2 - 4*t + 5 -> descending coeffs: (0, 0, 0, 0, 3, -4, 5)
    poly = Degree6PolynomialCoefficients(
        actuator_name="tau_test",
        coefficients=(0.0, 0.0, 0.0, 0.0, 3.0, -4.0, 5.0),
        duration_s=1.0,
    )
    assert poly.ordering == "descending"
    assert poly.degree == 6

    # t = 0: f(0) = 5, f'(0) = -4
    assert poly.evaluate(0.0) == pytest.approx(5.0)
    assert poly.evaluate_rate(0.0) == pytest.approx(-4.0)

    # t = 2: f(2) = 3*4 - 8 + 5 = 9, f'(2) = 6*2 - 4 = 8
    assert poly.evaluate(2.0) == pytest.approx(9.0)
    assert poly.evaluate_rate(2.0) == pytest.approx(8.0)


@pytest.mark.unit
def test_polynomial_ordering_reversal() -> None:
    """Test conversion between descending and ascending orders."""
    descending_coeffs = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    poly = Degree6PolynomialCoefficients(
        actuator_name="tau_arm",
        coefficients=descending_coeffs,
        duration_s=0.5,
    )
    assert poly.to_descending() == descending_coeffs
    assert poly.to_ascending() == descending_coeffs[::-1]

    # Simscape vector is 1D array of descending coeffs
    simscape_vec = poly.to_simscape_vector()
    assert np.allclose(simscape_vec, np.array(descending_coeffs))


@pytest.mark.unit
def test_polynomial_dbc_preconditions() -> None:
    """Ensure invalid dimensions, non-finite values, and bad durations raise ValueError."""
    # Must have exactly 7 coefficients
    with pytest.raises(ValueError, match="must have exactly 7 coefficients"):
        Degree6PolynomialCoefficients(
            actuator_name="tau_1",
            coefficients=(1.0, 2.0, 3.0),  # type: ignore[arg-type]
            duration_s=1.0,
        )

    # Must be finite
    with pytest.raises(ValueError, match="must be finite"):
        Degree6PolynomialCoefficients(
            actuator_name="tau_1",
            coefficients=(1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0),
            duration_s=1.0,
        )

    # Duration must be positive
    with pytest.raises(ValueError, match="duration_s must be positive and finite"):
        Degree6PolynomialCoefficients(
            actuator_name="tau_1",
            coefficients=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            duration_s=-0.5,
        )


@pytest.mark.unit
def test_fit_exact_degree6_polynomial() -> None:
    """Least-squares polynomial fit recovers exact degree-6 curve with zero defect."""
    times = np.linspace(0.0, 0.85, 86)
    # y(t) = 0.5*t^6 - 2*t^5 + t^4 - 0.5*t^3 + 3*t^2 - t + 10
    true_descending = [0.5, -2.0, 1.0, -0.5, 3.0, -1.0, 10.0]
    y = np.polyval(true_descending, times)
    controls = y.reshape(-1, 1)

    profile = fit_degree6_from_discrete_controls(
        times=times,
        controls=controls,
        actuator_names=["tau_pelvis_tilt"],
        duration_s=0.85,
    )

    coeff_obj = profile.profiles["tau_pelvis_tilt"]
    recovered = coeff_obj.coefficients
    assert np.allclose(recovered, true_descending, atol=1e-8)

    # Check fit quality
    metrics = profile.fit_metrics["tau_pelvis_tilt"]
    assert metrics["r_squared"] > 0.999999
    assert metrics["max_abs_error"] < 1e-8


@pytest.mark.unit
def test_profile_serialization_roundtrip() -> None:
    """Serialization to dict and JSON round-trips identically."""
    poly = Degree6PolynomialCoefficients(
        actuator_name="tau_wrist",
        coefficients=(0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7),
        duration_s=0.85,
    )
    profile = PolynomialTorqueProfile(
        actuator_names=("tau_wrist",),
        profiles={"tau_wrist": poly},
        duration_s=0.85,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "profile.json"
        profile.save_json(json_path)

        loaded = PolynomialTorqueProfile.load_json(json_path)
        assert loaded.actuator_names == ("tau_wrist",)
        assert loaded.duration_s == 0.85
        loaded_poly = loaded.profiles["tau_wrist"]
        assert loaded_poly.coefficients == poly.coefficients
        assert loaded_poly.actuator_name == "tau_wrist"


@pytest.mark.unit
def test_check_effort_and_rate_bounds() -> None:
    """Verify effort and rate bounds detection."""
    # f(t) = 100*t -> derivative is 100, max value is 100 on [0, 1]
    poly = Degree6PolynomialCoefficients(
        actuator_name="tau_strong",
        coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
        duration_s=1.0,
    )
    profile = PolynomialTorqueProfile(
        actuator_names=("tau_strong",),
        profiles={"tau_strong": poly},
        duration_s=1.0,
    )

    # Within bounds
    ok, violations = check_effort_and_rate_bounds(
        profile=profile,
        max_effort=150.0,
        max_rate=150.0,
    )
    assert ok
    assert len(violations) == 0

    # Exceeding bounds
    ok_fail, violations_fail = check_effort_and_rate_bounds(
        profile=profile,
        max_effort=50.0,
        max_rate=50.0,
    )
    assert not ok_fail
    assert "tau_strong" in violations_fail


@pytest.mark.unit
def test_load_controls_from_sto_and_fit_os4() -> None:
    """Verify loading real OS-4 tracked_controls.sto and fitting degree-6 profile."""
    from src.engines.physics_engines.opensim.python.tour_matching.polynomial_profile import (
        load_controls_from_sto,
    )

    sto_path = (
        Path(__file__).parent.parent.parent
        / "docs"
        / "development"
        / "opensim_tour_matching"
        / "evidence"
        / "os4_moco_tracking"
        / "tracked_controls.sto"
    )
    if not sto_path.is_file():
        pytest.skip(f"OS-4 tracked_controls.sto not found at {sto_path}")

    times, controls, actuator_names = load_controls_from_sto(sto_path)
    assert len(times) == 21
    assert controls.shape == (21, 39)
    assert len(actuator_names) == 39
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(0.10)

    # Fit degree-6 profile over the 0.10 s horizon
    profile = fit_degree6_from_discrete_controls(
        times=times,
        controls=controls,
        actuator_names=actuator_names,
        duration_s=0.10,
    )
    assert len(profile.profiles) == 39
    for name in actuator_names:
        metrics = profile.fit_metrics[name]
        # Controls are smooth, polynomial degree 6 achieves high fidelity (< 0.015 N*m max error)
        assert metrics["max_abs_error"] < 0.02
        assert metrics["rms_error"] < 0.005
