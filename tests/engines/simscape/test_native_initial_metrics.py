"""State qualification must not imply perfect capture-marker fitting."""

import numpy as np
import pytest

from src.engines.Simscape_Multibody_Models.python import tour_fit_state

pytestmark = pytest.mark.unit


def inputs() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = {"q": [0.0] * 27, "qd": [0.1] * 27, "labels": ["a", "b"]}
    markers = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    return (
        seed,
        np.zeros(27),
        np.full(27, 0.1),
        markers,
        markers.copy(),
        markers + [0.01, 0, 0],
    )


def test_reports_capture_error_without_rejecting_correct_native_state() -> None:
    result = tour_fit_state.verify_native_initial_state(*inputs())
    assert result["initial_state_verified"] is True
    assert result["initial_projection_max_error_m"] == 0
    assert result["initial_target_rms_m"] == pytest.approx(0.01)
    assert result["initial_target_max_error_m"] == pytest.approx(0.01)


@pytest.mark.parametrize("argument", [1, 2, 3])
def test_rejects_native_state_or_projection_mismatch(argument: int) -> None:
    seed, q, qd, markers, reference, target = inputs()
    values = [q, qd, markers, reference, target]
    values[argument - 1] = values[argument - 1] + 0.001
    with pytest.raises(ValueError, match="native initial"):
        tour_fit_state.verify_native_initial_state(seed, *values)


def test_rejects_nonfinite_reference() -> None:
    seed, q, qd, markers, reference, target = inputs()
    reference[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        tour_fit_state.verify_native_initial_state(
            seed, q, qd, markers, reference, target
        )
