"""Additivity charts for the pinned WSCG run (Issue #14440).

The chart plots, for the lead-arm joints, the canonical pointwise
counterfactuals of ``wscg_ztcf_additivity`` (#14438): ZTCF (the drift), ZVCF
(the control component), their sum and the full response, in each joint's
generalized acceleration (rad/s^2, positive along the MJCF joint axis), and the
additivity residual against its declared tolerance band. A model-scenario
result, not empirical validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("matplotlib")

from scripts.research.proximal_distal_energy.wscg_ztcf_additivity import (
    ABS_TOLERANCE,
    RELATIVE_TOLERANCE,
    run_additivity,
)
from scripts.research.proximal_distal_energy.wscg_ztcf_additivity_charts import (
    LEAD_ARM_JOINTS,
    AdditivityChartData,
    chart_data,
    draw_chart,
    write_charts,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


@pytest.fixture(scope="module")
def data() -> AdditivityChartData:
    return chart_data()


@pytest.fixture(scope="module")
def series() -> dict[str, np.ndarray]:
    return run_additivity()


def test_chart_data_is_the_computed_additivity_series(
    data: AdditivityChartData, series: dict[str, np.ndarray]
) -> None:
    np.testing.assert_array_equal(data.time_s, series["time_s"])
    np.testing.assert_array_equal(data.residual, series["additivity_residual"])
    np.testing.assert_array_equal(
        data.tolerance, ABS_TOLERANCE + RELATIVE_TOLERANCE * series["full_norm"]
    )
    assert data.joints == LEAD_ARM_JOINTS
    assert data.full.shape == (len(series["time_s"]), len(LEAD_ARM_JOINTS))
    assert np.all(np.abs(data.full - data.ztcf - data.zvcf) <= data.tolerance[:, None])


def test_events_mark_peak_speed_and_peak_full_response(
    data: AdditivityChartData, series: dict[str, np.ndarray]
) -> None:
    times = series["time_s"]
    assert data.events == {
        "Peak Speed": float(times[np.argmax(series["speed"])]),
        "Peak |a_full|": float(times[np.argmax(series["full_norm"])]),
    }


def test_plotted_lines_carry_exactly_the_chart_data(data: AdditivityChartData) -> None:
    fig = draw_chart(data)
    *joint_axes, residual_axis = fig.axes

    assert len(joint_axes) == len(LEAD_ARM_JOINTS)
    for column, ax in enumerate(joint_axes):
        lines = {line.get_label(): line for line in ax.get_lines()}
        expected = {
            "Full": data.full[:, column],
            "ZTCF": data.ztcf[:, column],
            "ZVCF": data.zvcf[:, column],
            "ZTCF + ZVCF": data.ztcf[:, column] + data.zvcf[:, column],
        }
        for label, values in expected.items():
            np.testing.assert_array_equal(lines[label].get_xdata(), data.time_s)
            np.testing.assert_array_equal(lines[label].get_ydata(), values)
    residual = {line.get_label(): line for line in residual_axis.get_lines()}
    np.testing.assert_array_equal(residual["Residual"].get_ydata(), data.residual)


def test_axes_carry_units_and_event_markers(data: AdditivityChartData) -> None:
    fig = draw_chart(data)
    *joint_axes, residual_axis = fig.axes

    assert residual_axis.get_xlabel() == "Time (s)"
    assert residual_axis.get_ylabel() == "Residual (rad/s²)"
    assert residual_axis.get_yscale() == "log"
    for ax in fig.axes:
        markers = {line.get_label(): line.get_xdata()[0] for line in ax.get_lines()}
        for name, time_s in data.events.items():
            assert markers[name] == time_s
    for joint, ax in zip(LEAD_ARM_JOINTS, joint_axes, strict=True):
        assert ax.get_ylabel() == "Acceleration (rad/s²)"
        assert joint in ax.get_title()


def test_svg_and_png_are_written_and_reproducible(
    data: AdditivityChartData, tmp_path: Path
) -> None:
    first = write_charts(data, tmp_path / "a" / "chart")
    second = write_charts(data, tmp_path / "b" / "chart")

    assert [p.suffix for p in first] == [".svg", ".png"]
    assert first[0].read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert first[1].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()


def test_unknown_joints_are_refused() -> None:
    with pytest.raises(ValueError, match="not_a_joint"):
        chart_data(("left_elbow", "not_a_joint"))
