"""Optimizer boxes must imply every explicitly declared chart restriction."""

from types import SimpleNamespace

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
    _declared_chart_blockers,
)

pytestmark = pytest.mark.unit


def _request(chart: dict, linear: dict, bounds: dict) -> SimpleNamespace:
    return SimpleNamespace(
        constrained_cold_start=SimpleNamespace(
            chart_bounds=chart, linear_chart_bounds=linear
        ),
        bindings=SimpleNamespace(state_bounds=bounds),
    )


def test_native_chart_rejects_optimizer_bounds_outside_valid_initial_chart() -> None:
    request = _request({"/q": (0.30, 0.32)}, {}, {"/q/value": (0.2, 0.4)})
    assert _declared_chart_blockers(request) == ["native-moco-chart-bounds-unqualified"]


@pytest.mark.parametrize("limits,accepted", [((-1.0, 1.0), True), ((-0.2, 0.2), False)])
def test_linear_chart_requires_all_box_corners_to_fit(
    limits: tuple, accepted: bool
) -> None:
    request = _request(
        {},
        {"rule": ({"/q": 2.0, "/x": -1.0}, *limits)},
        {"/q/value": (0.2, 0.4), "/x/value": (-0.1, 0.1)},
    )
    assert (not _declared_chart_blockers(request)) == accepted


def test_linear_chart_rejects_unknown_native_coordinate_bound() -> None:
    request = _request({}, {"rule": ({"/unknown": 1.0}, -1.0, 1.0)}, {})
    assert _declared_chart_blockers(request) == [
        "native-moco-linear-chart-bounds-unqualified"
    ]


def test_linear_chart_rejects_nonfinite_interval_proof() -> None:
    request = _request(
        {},
        {"rule": ({"/q": 1e308, "/x": -1e308}, -1e308, 1e308)},
        {"/q/value": (1e308, 1e308), "/x/value": (1e308, 1e308)},
    )
    assert _declared_chart_blockers(request) == [
        "native-moco-linear-chart-bounds-unqualified"
    ]
