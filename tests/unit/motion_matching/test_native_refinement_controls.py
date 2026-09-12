"""Refinement subspaces must preserve frozen native effort channels."""

import numpy as np
import pytest

from docs.development.simscape_tour_matching.native_evidence.reproduction.refine_native_candidate import (
    control_matrix,
)
from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

pytestmark = pytest.mark.unit


def test_root_only_preserves_torques_and_early_controls() -> None:
    controls = control_matrix(np.full(9, 1.2), 27, 4, True, 125.0)
    np.testing.assert_allclose(controls[:3, 4:], 25.0)
    np.testing.assert_array_equal(controls[3:], 0)
    np.testing.assert_array_equal(controls[:, :4], 0)


def test_historical_full_space_keeps_coordinate_order() -> None:
    parameters = np.linspace(0.8, 1.2, 27)
    controls = control_matrix(parameters, 27, 6, False, 10.0)
    np.testing.assert_allclose(controls[:, 6], 10 * (parameters - 1))
    np.testing.assert_array_equal(controls[:, :6], 0)


def test_expanded_sextic_preserves_initial_effort_and_slope() -> None:
    controls = control_matrix(np.full(135, 1.2), 27, 2, False, 10.0)
    coefficients = bernstein_to_simscape(controls, duration_s=0.8)
    np.testing.assert_array_equal(controls[:, :2], 0)
    np.testing.assert_array_equal(coefficients[:, -2:], 0)
    assert coefficients.shape == (27, 7)
    assert np.polyval(np.polyder(coefficients[0], 2), 0.0) > 0
    assert np.polyval(coefficients[0], 0.8) == pytest.approx(2.0)


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_physical_scale_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        control_matrix(np.ones(9), 27, 4, True, scale)


def test_wrong_parameter_count_rejected() -> None:
    with pytest.raises(ValueError, match="parameters"):
        control_matrix(np.ones(81), 27, 4, True, 125.0)
