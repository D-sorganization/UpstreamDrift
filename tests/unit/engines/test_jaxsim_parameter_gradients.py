"""JaxSim parameter-gradient sensitivity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.jaxsim import (
    DEFAULT_PARAMETER_VECTOR,
    PARAMETER_NAMES,
    JaxSimBackend,
    evaluate_ztcf_parameter_sensitivity_along_trajectory,
    parameter_jacobian,
    validate_parameter_jacobian,
)
from src.shared.python.engine_core.sub_protocols import SupportsParameterGradients

jax = pytest.importorskip("jax")


def _sample_state() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([0.18, -0.31], dtype=np.float64),
        np.array([0.42, -0.27], dtype=np.float64),
    )


def test_parameter_jacobian_matches_finite_difference() -> None:
    q, v = _sample_state()

    validation = validate_parameter_jacobian(DEFAULT_PARAMETER_VECTOR, q, v)

    assert validation.autodiff_jacobian.shape == (2, len(PARAMETER_NAMES))
    np.testing.assert_allclose(
        validation.autodiff_jacobian,
        validation.finite_difference_jacobian,
        rtol=2.0e-3,
        atol=2.0e-3,
    )
    assert validation.max_abs_error < 2.0e-3


def test_forward_and_reverse_autodiff_agree() -> None:
    q, v = _sample_state()

    forward = parameter_jacobian(DEFAULT_PARAMETER_VECTOR, q, v, mode="forward")
    reverse = parameter_jacobian(DEFAULT_PARAMETER_VECTOR, q, v, mode="reverse")

    np.testing.assert_allclose(forward, reverse, rtol=1.0e-8, atol=1.0e-8)


def test_trajectory_sensitivity_is_pointwise() -> None:
    q_traj = np.array([[0.1, -0.2], [0.2, -0.15], [0.3, -0.1]], dtype=np.float64)
    v_traj = np.array([[0.3, -0.4], [0.2, -0.1], [0.1, 0.05]], dtype=np.float64)

    sensitivity = evaluate_ztcf_parameter_sensitivity_along_trajectory(
        DEFAULT_PARAMETER_VECTOR,
        q_traj,
        v_traj,
    )

    assert sensitivity.shape == (3, 2, len(PARAMETER_NAMES))
    np.testing.assert_allclose(
        sensitivity[1],
        parameter_jacobian(DEFAULT_PARAMETER_VECTOR, q_traj[1], v_traj[1]),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


@pytest.mark.parametrize(
    ("params", "q", "v", "match"),
    [
        ([1.0, 2.0], [0.1, 0.2], [0.3, 0.4], "parameter_vector"),
        ([0.72, -1.0, 1.1, 0.18, 0.2], [0.1, 0.2], [0.3, 0.4], "positive"),
        ([0.72, 4.5, 1.1, 0.18, 0.2], [0.1], [0.3, 0.4], "q"),
        ([0.72, 4.5, 1.1, 0.18, 0.2], [0.1, 0.2], [np.nan, 0.4], "finite"),
    ],
)
def test_parameter_gradient_inputs_are_checked(
    params: list[float],
    q: list[float],
    v: list[float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        parameter_jacobian(params, q, v)


def test_trajectory_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="share shape"):
        evaluate_ztcf_parameter_sensitivity_along_trajectory(
            DEFAULT_PARAMETER_VECTOR,
            np.zeros((2, 2)),
            np.zeros((3, 2)),
        )

    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_ztcf_parameter_sensitivity_along_trajectory(
            DEFAULT_PARAMETER_VECTOR,
            np.zeros((0, 2)),
            np.zeros((0, 2)),
        )


def test_backend_exposes_parameter_gradient_protocol() -> None:
    backend = JaxSimBackend()
    q, v = _sample_state()

    assert isinstance(backend, SupportsParameterGradients)
    np.testing.assert_allclose(
        backend.parameter_jacobian(DEFAULT_PARAMETER_VECTOR, q, v),
        parameter_jacobian(DEFAULT_PARAMETER_VECTOR, q, v),
    )


def test_sample_sensitivity_plot_script_writes_png(tmp_path: Path) -> None:
    from scripts.jaxsim.plot_parameter_sensitivity import (
        write_parameter_sensitivity_plot,
    )

    output_path = write_parameter_sensitivity_plot(
        tmp_path / "sensitivity.png",
        samples=8,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
