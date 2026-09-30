"""DbC tests: ``FitOptions.max_marker_rmse_m`` must be finite and positive when set."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from src.shared.python.motion_matching.provider import FitOptions


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), 0.0, -0.01],
)
def test_fit_options_rejects_invalid_max_marker_rmse_m(bad: float) -> None:
    """NaN silently disables a gate and non-positive ceilings reject planar data."""
    with pytest.raises(
        ValueError, match="max_marker_rmse_m must be finite and positive"
    ):
        FitOptions(max_marker_rmse_m=bad)


@pytest.mark.parametrize("good", [None, 0.01, 1e-3, 105.5])
def test_fit_options_allows_none_or_positive_ceiling(good) -> None:
    assert FitOptions(max_marker_rmse_m=good).max_marker_rmse_m == good
