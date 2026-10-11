"""Shared figure-option contracts must fail before consuming plot data."""

import pytest

pytestmark = pytest.mark.unit


def _plots():
    pytest.importorskip("matplotlib")
    from src.tools.shot_pattern_analysis import academic_plots

    return academic_plots


@pytest.mark.parametrize("dpi", [72, 96, 300, 600])
@pytest.mark.parametrize("presentation", ["academic", "social"])
def test_valid_options(dpi, presentation):
    assert _plots()._validate_figure_options("Study", dpi, presentation) is None


@pytest.mark.parametrize(
    "scenario,dpi,presentation,message",
    [
        (None, 0, "invalid", "scenario must be a non-empty string"),
        ("  ", 0, "invalid", "scenario must be a non-empty string"),
        ("Study", 71, "invalid", "dpi must be an integer in [72, 600]"),
        ("Study", 601, "social", "dpi must be an integer in [72, 600]"),
        ("Study", 100.0, "social", "dpi must be an integer in [72, 600]"),
        ("Study", True, "social", "dpi must be an integer in [72, 600]"),
        ("Study", 96, "invalid", "presentation must be academic or social"),
    ],
)
def test_both_renderers_preserve_validation_order(
    tmp_path, scenario, dpi, presentation, message
):
    api = _plots()
    options = {
        "scenario": scenario,
        "dpi": dpi,
        "presentation": presentation,
        "metadata": {},
    }
    for render, extra in [
        (api.write_academic_dispersion_figure, {"target_point_m": (100.0, 0.0)}),
        (api.write_overhead_trajectory_figure, {"target_x_m": 100.0}),
    ]:
        with pytest.raises(ValueError) as raised:
            render({}, tmp_path / "unused", **options, **extra)
        assert str(raised.value) == message
    assert not list(tmp_path.iterdir())
