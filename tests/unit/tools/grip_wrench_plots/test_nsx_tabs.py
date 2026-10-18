"""NSX-43: the PyQt tabs render the shared golden and nothing else.

Headless (``QT_QPA_PLATFORM=offscreen``); skipped when PyQt6/matplotlib cannot
be loaded (AGENTS.md E).
"""

from __future__ import annotations

import threading

import pytest

try:
    import matplotlib  # noqa: F401
    import PyQt6.QtCore  # noqa: F401
    import pytestqt  # noqa: F401
except (ImportError, OSError) as exc:  # pragma: no cover - environment
    pytest.skip(
        f"PyQt6/matplotlib/pytest-qt not loadable: {exc}", allow_module_level=True
    )

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.loop_nullspace.facade import AnalysisCancelledError  # noqa: E402
from src.shared.python.loop_nullspace.registry import load_registry  # noqa: E402
from src.tools.grip_wrench_plots import nsx_view  # noqa: E402
from src.tools.grip_wrench_plots.nsx_tabs import NSXTabs  # noqa: E402
from src.tools.grip_wrench_plots.window import (  # noqa: E402
    GripWrenchTabsWidget,
    parse_frames,
)
from tests.unit.tools.grip_wrench_plots.golden_support import (  # noqa: E402
    golden_dict,
    golden_model,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def window(qtbot):
    widget = GripWrenchTabsWidget(
        job_factory=lambda request: lambda s, c: golden_model(), sources=["fake"]
    )
    qtbot.addWidget(widget)
    yield widget
    widget.cleanup()


def _gids(ax) -> list[str]:
    arts = [*ax.collections, *ax.patches, *ax.lines]
    return [a.get_gid() for a in arts if a.get_gid()]


def test_first_tab_is_the_existing_grip_view_then_the_four_panels(window) -> None:
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == [
        "Grip Wrench",
        "Invariants",
        "Envelopes",
        "Hand Split",
        "Diagnostics",
        "Variability & Consistency",
    ]
    assert window.tabs.widget(0) is window.grip_widget


def test_tabs_show_what_the_golden_says(qtbot) -> None:
    tabs = NSXTabs()
    qtbot.addWidget(tabs)
    raw = golden_dict()
    tabs.set_plot_model(golden_model())
    env = tabs.panel_tabs["envelopes"]
    expected = nsx_view.panel_views(golden_model())["envelopes"]
    assert env.badge_label.text() == expected.badge
    assert raw["envelopes"]["bands"][0]["decision"]["label"] in env.badge_label.text()
    assert [ax.get_title(loc="left") for ax in env.figure.axes] == [
        b["quantity_id"] for b in raw["envelopes"]["bands"]
    ]
    rows = {r[0]: r for r in env.row_texts()}
    assert rows["tau_4"][2] == "withheld"
    assert rows["tau_4"][4] == raw["envelopes"]["withheld"][0]["reason"]
    assert rows["tau_1"][2] == "shown"


def test_band_segments_are_drawn_by_display_class(qtbot) -> None:
    tabs = NSXTabs()
    qtbot.addWidget(tabs)
    tabs.set_plot_model(golden_model())
    raw = golden_dict()
    drawn = {
        ax.get_title(loc="left"): _gids(ax)
        for ax in tabs.panel_tabs["envelopes"].figure.axes
    }
    for band in raw["envelopes"]["bands"]:
        want = sorted(f"segment:{s['display_class']}" for s in band["segments"])
        got = sorted(g for g in drawn[band["quantity_id"]] if g.startswith("segment:"))
        assert got == want
    tau1 = drawn["tau_1"]
    assert "overlay" in tau1, "the representative is drawn on its band"
    assert "tau_4" not in drawn


def test_withheld_values_are_not_plotted(qtbot) -> None:
    tabs = NSXTabs()
    qtbot.addWidget(tabs)
    tabs.set_plot_model(golden_model())
    inv = tabs.panel_tabs["invariants"]
    titles = [ax.get_title(loc="left") for ax in inv.figure.axes]
    assert "net_club_wrench" not in titles
    rows = {r[0]: r for r in inv.row_texts()}
    assert rows["net_club_wrench"][2] == "withheld"
    assert "club_ne" in rows["net_club_wrench"][4]
    assert "SHOW_REASON" in inv.badge_label.text()


def test_hand_split_and_diagnostics_tabs(qtbot) -> None:
    tabs = NSXTabs()
    qtbot.addWidget(tabs)
    tabs.set_plot_model(golden_model())
    split = tabs.panel_tabs["hand_split"]
    assert [ax.get_title(loc="left") for ax in split.figure.axes] == [
        b["quantity_id"] for b in golden_dict()["hand_split"]["bands"]
    ]
    assert "frame: grip frame" in split.notes_label.text()
    diag = {r[0]: r for r in tabs.panel_tabs["diagnostics"].row_texts()}
    assert diag["residual phi"][2] == "withheld"
    assert diag["admission"][2] == golden_dict()["diagnostics"]["admission"]["verdict"]


def test_clearing_returns_to_the_empty_state(qtbot) -> None:
    tabs = NSXTabs()
    qtbot.addWidget(tabs)
    tabs.set_plot_model(golden_model())
    tabs.set_plot_model(None)
    for tab in tabs.panel_tabs.values():
        assert tab.table.rowCount() == 0
        assert tab.figure.axes == []
        assert tab.badge_label.text() == "no analysis run yet"
    with pytest.raises(TypeError):
        tabs.set_plot_model({})  # type: ignore[arg-type]


def test_preset_selector_lists_the_registry_presets(window) -> None:
    reg = load_registry()
    assert [
        window.preset_combo.itemData(i) for i in range(window.preset_combo.count())
    ] == [p.id for p in reg.presets]
    window.preset_combo.setCurrentIndex(window.preset_combo.count() - 1)
    assert window.selected_preset() == reg.presets[-1].id


def test_run_renders_the_model_off_the_gui_thread(window, qtbot) -> None:
    threads: list[int] = []

    def factory(request):
        def job(on_stage, should_cancel):
            threads.append(threading.get_ident())
            on_stage("invariants", 30.0, None)
            return golden_model()

        return job

    window._job_factory = factory
    with qtbot.waitSignal(window.analysis_finished, timeout=20000):
        window.run_analysis()
    assert threads and threads[0] != threading.get_ident()
    assert window.plot_model() == golden_model()
    assert window.tabs.currentIndex() == 1
    assert window.tabs.panel_tabs["envelopes"].table.rowCount() > 0
    assert not window.is_running()
    assert window.run_button.isEnabled()


def test_cancel_stops_the_run_and_keeps_the_gui_alive(window, qtbot) -> None:
    started = threading.Event()

    def factory(request):
        def job(on_stage, should_cancel):
            started.set()
            while not should_cancel():
                threading.Event().wait(0.01)
            raise AnalysisCancelledError("cancelled")

        return job

    window._job_factory = factory
    with qtbot.waitSignal(window.analysis_cancelled, timeout=20000):
        window.run_analysis()
        assert started.wait(10)
        assert window.cancel_button.isEnabled()
        window.cancel_analysis()
    assert window.plot_model() is None
    assert window.status_label.text() == "cancelled"
    assert not window.is_running()


def test_a_failed_job_is_reported_not_raised(window, qtbot) -> None:
    def factory(request):
        def job(on_stage, should_cancel):
            raise ValueError("refused: nothing to analyse")

        return job

    window._job_factory = factory
    with qtbot.waitSignal(window.analysis_failed, timeout=20000) as blocker:
        window.run_analysis()
    assert "refused: nothing to analyse" in blocker.args[0]
    assert window.status_label.text().startswith("failed:")


def test_invalid_frames_do_not_start_a_run(window) -> None:
    window.frames_edit.setText("1, 1")
    window.run_analysis()
    assert not window.is_running()
    assert window.status_label.text().startswith("cannot run")
    with pytest.raises(ValueError):
        parse_frames("")
    with pytest.raises(ValueError):
        parse_frames("a")
    assert parse_frames("0, 3 6") == (0, 3, 6)


def test_no_registered_source_disables_run(qtbot) -> None:
    widget = GripWrenchTabsWidget(sources=[])
    qtbot.addWidget(widget)
    assert not widget.run_button.isEnabled()
    assert "no trajectory source" in widget.status_label.text()
    widget.cleanup()
    widget.cleanup()


def test_application_exists() -> None:
    assert QApplication.instance() is not None


def test_embed_adapter_builds_the_tabbed_widget_lazily(qtbot) -> None:
    from src.tools.grip_wrench_plots._embed_adapter import GripWrenchPlotsEmbedAdapter

    adapter = GripWrenchPlotsEmbedAdapter()
    widget = adapter.create_main_widget(None)
    qtbot.addWidget(widget)
    assert isinstance(widget, GripWrenchTabsWidget)
    assert adapter.create_main_widget(None) is widget
    adapter.cleanup()
    adapter.cleanup()
