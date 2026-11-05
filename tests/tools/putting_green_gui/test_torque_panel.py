"""Counter-torque explorer in the Putting Green Simulator (#14469, epic #14452).

The desktop simulator charts the shared core's two-hand couple model: the
in-plane hand couples over the stroke and the face twist a gust starts with
and without the grip preload. Golden parity: every plotted value is the
core's :func:`counter_torque.stroke_series` value in the user's units.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.putting_dynamics import counter_torque as ct  # noqa: E402
from src.shared.python.putting_dynamics.putter_presets import BALANCE_PRESETS  # noqa: E402
from src.shared.python.ui.units import UnitSystem, to_display_torque  # noqa: E402
from src.tools.putting_green_gui._torque_panel import TorquePanel  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv[:1])


def test_defaults_are_the_core_defaults(qapp: QApplication) -> None:
    panel = TorquePanel(lambda: UnitSystem.METRIC)
    assert panel.current_inputs() == ct.CounterTorqueInputs()
    putters = panel.putter_combo
    assert [putters.itemData(i) for i in range(putters.count())] == list(
        BALANCE_PRESETS
    )


def test_chart_plots_the_core_series_in_si(qapp: QApplication) -> None:
    panel = TorquePanel(lambda: UnitSystem.METRIC)
    series = ct.stroke_series(panel.current_inputs())
    left, right = panel.couple_lines()
    assert left == pytest.approx([s.tau_left_in_nm for s in series], abs=1e-12)
    assert right == pytest.approx([s.tau_right_in_nm for s in series], abs=1e-12)
    preload, slack = panel.twist_lines()
    assert preload == pytest.approx([s.phi_preload_deg for s in series], abs=1e-12)
    assert slack == pytest.approx([s.phi_slack_deg for s in series], abs=1e-12)
    assert panel.couple_axis_label().endswith("(N·m)")


def test_imperial_units_convert_the_couples_and_the_inputs(
    qapp: QApplication,
) -> None:
    units = [UnitSystem.METRIC]
    panel = TorquePanel(lambda: units[0])
    panel.set_preload_nm(1.2)
    units[0] = UnitSystem.IMPERIAL
    panel.redraw()
    assert panel.current_inputs().preload_nm == pytest.approx(1.2, abs=1e-3)
    assert panel.couple_axis_label().endswith("(in·lbf)")
    series = ct.stroke_series(panel.current_inputs())
    left, _ = panel.couple_lines()
    imperial = [
        to_display_torque(s.tau_left_in_nm, UnitSystem.IMPERIAL) for s in series
    ]
    assert left == pytest.approx(imperial, abs=1e-9)
    assert "lbf" in panel.summary_text()


def test_preload_shrinks_the_gust_twist_in_the_summary(qapp: QApplication) -> None:
    panel = TorquePanel(lambda: UnitSystem.METRIC)
    panel.set_preload_nm(1.5)
    series = ct.stroke_series(panel.current_inputs())
    held = max(abs(s.phi_preload_deg) for s in series)
    slack = max(abs(s.phi_slack_deg) for s in series)
    assert held < slack
    assert f"{held:.2f}°" in panel.summary_text()
    assert f"{slack:.2f}°" in panel.summary_text()


def test_changing_the_putter_replots(qapp: QApplication) -> None:
    panel = TorquePanel(lambda: UnitSystem.METRIC)
    panel.putter_combo.setCurrentIndex(list(BALANCE_PRESETS).index("face_balanced"))
    assert panel.current_inputs().putter == "face_balanced"
    face = ct.evaluate(0.0, panel.current_inputs()).face_angle_impact_deg
    assert f"{face:+.2f}°" in panel.summary_text()


def test_the_chart_titles_fit_the_phone_width(qapp: QApplication) -> None:
    # At the 500 px window the charts are about 205 px wide.
    panel = TorquePanel(lambda: UnitSystem.IMPERIAL)
    figure = panel.canvas.figure
    figure.set_size_inches(205 / figure.dpi, 380 / figure.dpi)
    panel.canvas.draw()
    for box in panel.title_extents():
        assert box.x0 >= 0.0 and box.x1 <= figure.bbox.width


def test_each_couple_input_takes_its_own_core_range(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gust input is bounded by GUST_RANGE_NM, not the preload range (#14617)."""
    monkeypatch.setattr(ct, "GUST_RANGE_NM", (0.0, 3.0))
    units = [UnitSystem.METRIC]
    panel = TorquePanel(lambda: units[0])
    expected = {
        "preload_nm": ct.PRELOAD_RANGE_NM,
        "off_preload_nm": ct.PRELOAD_RANGE_NM,
        "gust_impulse_nm": (0.0, 3.0),
    }
    for system in (UnitSystem.METRIC, UnitSystem.IMPERIAL):
        units[0] = system
        panel.redraw()
        ranges = panel.torque_input_ranges()
        assert ranges.keys() == expected.keys()
        for field, bounds in expected.items():
            assert ranges[field] == pytest.approx(bounds, abs=1e-3), (system, field)
