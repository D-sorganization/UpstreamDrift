"""NSX-100: the PyQt6 Variability & Consistency tab and its controls (headless)."""

from __future__ import annotations

import pytest

try:
    import matplotlib  # noqa: F401
    import PyQt6.QtCore  # noqa: F401
    import pytestqt  # noqa: F401
except (ImportError, OSError) as exc:  # pragma: no cover - environment
    pytest.skip(f"PyQt6/matplotlib/pytest-qt not loadable: {exc}", allow_module_level=True)

from src.shared.python.loop_nullspace.plot_model import NSXPlotModel  # noqa: E402
from src.shared.python.loop_nullspace.sensitivity_options import option_catalogue  # noqa: E402
from src.tools.grip_wrench_plots.nsx_tabs import NSXTabs  # noqa: E402
from src.tools.grip_wrench_plots.variability_tab import VariabilityTab  # noqa: E402
from src.tools.grip_wrench_plots.window import GripWrenchTabsWidget  # noqa: E402
from tests.unit.tools.grip_wrench_plots.golden_support import (  # noqa: E402
    golden_model,
    variability_golden_model,
    view_text_fixture,
)

pytestmark = pytest.mark.unit
TITLE = "Variability & Consistency"


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_tab_renders_every_string_of_the_shared_golden(qtbot) -> None:
    tab = VariabilityTab(catalogue=option_catalogue())
    qtbot.addWidget(tab)
    tab.set_plot_model(variability_golden_model())
    shown = _norm(tab.all_text())
    missing = [
        s for s in view_text_fixture()["complete"]["expected"] if _norm(s) not in shown
    ]
    assert missing == []
    assert len(tab.figure.axes) > 0


def test_capability_limited_engine_shows_unavailable_reason(qtbot) -> None:
    fx = view_text_fixture()["unavailable"]
    tab = VariabilityTab(catalogue=option_catalogue())
    qtbot.addWidget(tab)
    tab.set_plot_model(NSXPlotModel.from_dict(fx["plot_model"]))
    shown = _norm(tab.all_text())
    assert "UNAVAILABLE: " in shown
    assert all(_norm(s) in shown for s in fx["expected"])


def test_no_sensitivity_model_says_not_requested(qtbot) -> None:
    tab = VariabilityTab(catalogue=option_catalogue())
    qtbot.addWidget(tab)
    tab.set_plot_model(golden_model())
    assert "not requested" in tab.all_text()


def test_tabs_host_the_variability_tab_last(qtbot) -> None:
    tabs = NSXTabs(catalogue=option_catalogue())
    qtbot.addWidget(tabs)
    assert tabs.tabText(tabs.count() - 1) == TITLE
    tabs.set_plot_model(variability_golden_model())
    assert "H-NOISE-ADDITIVE" in tabs.variability_tab.all_text()


def test_controls_offer_only_the_catalogue(qtbot) -> None:
    cat = option_catalogue()
    tab = VariabilityTab(catalogue=cat)
    qtbot.addWidget(tab)
    c = tab.controls
    assert [c.stage_combo.itemData(i) for i in range(c.stage_combo.count())] == [
        s["id"] for s in cat["stages"]
    ]
    assert [c.spectrum_combo.itemData(i) for i in range(c.spectrum_combo.count())] == [
        s["id"] for s in cat["spectra"]
    ]
    assert set(c.mechanism_checks) == {m["id"] for m in cat["mechanisms"]}
    assert set(c.strategy_checks) == {s["id"] for s in cat["strategies"]}
    assert c.outcome_combo.currentData() == "face_angle"
    engines = [c.engine_combo.itemData(i) for i in range(c.engine_combo.count())]
    assert engines == [e["id"] for e in cat["engines"]]
    sim = c.engine_combo.findData("simscape")
    assert "UNAVAILABLE: " in c.engine_combo.itemText(sim)


def test_controls_build_the_request_options(qtbot) -> None:
    tab = VariabilityTab(catalogue=option_catalogue())
    qtbot.addWidget(tab)
    c = tab.controls
    assert c.request_options() is None  # disabled by default
    c.enable_check.setChecked(True)
    c.channels_edit.setText("shoulder=0.004, elbow=0.009")
    c.strategy_checks["S3"].setChecked(False)
    opts = c.request_options()
    assert opts is not None
    assert opts["disturbance_spec"][0]["covariance"] == [[0.004, 0.0], [0.0, 0.009]]
    assert "S3" not in opts["strategy_set"]
    assert opts["outcome_priority"] == ["face_angle"]
    assert c.engine() == c.engine_combo.currentData()


def test_bad_channel_text_is_a_value_error(qtbot) -> None:
    tab = VariabilityTab(catalogue=option_catalogue())
    qtbot.addWidget(tab)
    tab.controls.enable_check.setChecked(True)
    tab.controls.channels_edit.setText("shoulder")
    with pytest.raises(ValueError):
        tab.controls.request_options()


def test_window_request_carries_sensitivity_and_engine(qtbot) -> None:
    seen = []

    def factory(request):
        seen.append(request)
        return lambda s, c: variability_golden_model()

    w = GripWrenchTabsWidget(job_factory=factory, sources=["fake"])
    qtbot.addWidget(w)
    c = w.tabs.variability_tab.controls
    assert w.current_request().sensitivity is None
    c.enable_check.setChecked(True)
    c.channels_edit.setText("shoulder=0.004, elbow=0.009")
    c.engine_combo.setCurrentIndex(c.engine_combo.findData("simscape"))
    req = w.current_request()
    assert req.engine == "simscape"
    assert req.sensitivity is not None
    w.run_analysis()
    qtbot.waitUntil(lambda: not w.is_running(), timeout=30000)
    assert seen[-1].sensitivity == req.sensitivity
    assert w.plot_model() is not None
    w.cleanup()
