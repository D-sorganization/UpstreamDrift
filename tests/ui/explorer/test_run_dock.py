"""Offscreen: the Run Simulation dock runs a pinned case off the GUI thread
and hands a playable archive to the explorer (#15150, R1)."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="the explorer needs PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.model_explorer import run_dock  # noqa: E402

pytestmark = [pytest.mark.ui, pytest.mark.integration]

NPZ = Path(__file__).parents[2] / "fixtures/results_playback/mujoco_fit"
NPZ /= "dynamics_record.npz"
GOLF, ZTCF = "golf_simulator_golden_shot", "counterfactual_ztcf_golden"
OK = {"status": "succeeded", "run_id": "agent_1"}


class _FakeService:
    """The sim_run boundary: two engines; each run returns or raises ``outcome``."""

    def __init__(self, outcome: Any = None) -> None:
        self.outcome = outcome or run_dock.RunOutcome(OK, {"golden_case_id": "c_v1"})
        self.threads: list[int] = []

    def describe_engines(self) -> list[dict[str, Any]]:
        down = {"runtime_status": "unavailable", "unavailable_reason": "not installed"}
        return [{"id": "pendulum", "runtime_status": "available"}, {"id": "dk", **down}]

    def run_preset(self, preset_id: str) -> run_dock.RunOutcome:
        self.threads.append(threading.get_ident())
        time.sleep(0.2)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10.0
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    assert condition(), "timed out waiting on the run dock"


@pytest.fixture
def make_panel() -> Generator[Callable[..., Any], None, None]:
    made: list[Any] = [QApplication.instance() or QApplication(["run"])]

    def _make(service: Any, host: Any = None) -> Any:
        made.append(run_dock.RunSimulationPanel(service, host))
        _until(lambda: made[-1].engine_list.count() > 0)
        return made[-1]

    yield _make
    for panel in made[1:]:
        panel.cleanup()
        panel.deleteLater()


def _run(panel: Any, preset_id: str = GOLF) -> None:
    panel.preset_combo.setCurrentIndex(panel.preset_combo.findData(preset_id))
    panel.run_button.click()
    assert not panel.run_button.isEnabled()
    assert panel.is_running and panel.progress.isVisibleTo(panel)
    _until(lambda: not panel.is_running)
    assert panel.run_button.isEnabled() and not panel.progress.isVisibleTo(panel)


def test_run_off_the_gui_thread_shows_a_summary(make_panel: Any) -> None:
    service, host = _FakeService(), Mock()
    panel = make_panel(service, host)
    rows = [panel.engine_list.item(i).text() for i in range(2)]
    assert rows == ["pendulum: available", "dk: unavailable"]
    assert "not installed" in panel.engine_list.item(1).toolTip()
    assert [panel.preset_combo.itemData(i) for i in range(2)] == [GOLF, ZTCF]
    _run(panel)
    assert service.threads and service.threads[0] != threading.get_ident()
    assert "succeeded" in panel.status_label.text()
    assert "c_v1" in panel.summary_view.toPlainText()
    host.load_results.assert_not_called()  # a summary-only run opens no playback


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (RuntimeError("backend exploded"), "Error: backend exploded"),
        (run_dock.RunOutcome({"error": {"message": "bad"}}, {}), "Run failed: bad"),
    ],
)
def test_errors_reach_the_status(make_panel: Any, outcome: Any, message: str) -> None:
    panel = make_panel(_FakeService(outcome))
    _run(panel, ZTCF)
    assert panel.status_label.text() == message


def test_playable_archive_auto_plays_then_cleanup(make_panel: Any) -> None:
    key, host = run_dock.load_run_dock_settings().playable_result_key, Mock()
    panel = make_panel(_FakeService(run_dock.RunOutcome(OK, {key: str(NPZ)})), host)
    _run(panel)
    host.load_results.assert_called_once_with(NPZ, wait=False)
    panel.run_button.click()  # a second run, abandoned in flight
    panel.cleanup()
    panel.cleanup()
    _until(lambda: run_dock.live_call_count() == 0)
    assert host.load_results.call_count == 1


def test_service_reads_a_spilled_payload(tmp_path: Path, monkeypatch: Any) -> None:
    from src.sim_run import agent_capabilities

    points = [[0.0, 0.0, 0.0], [100.0, 2.0, 30.0], [200.0, 4.0, 0.0]]
    body = {
        "golden_case_id": "agent_api_golf_shot_v1",
        "simulation": {"trajectory": [{"position": p} for p in points]},
        "qualification_banner": {"model_limits": ["ODE reference only."]},
    }

    def _capability(capability: str, request: Any, store: Any) -> dict[str, Any]:
        store.save_artifact("agent_9", "result.json", json.dumps(body))
        ref = {"run_id": "agent_9", "artifact_name": "result.json"}
        return {**OK, "delivery": {"encoding": "artifact", "artifact_ref": ref}}

    monkeypatch.setattr(agent_capabilities, "run_agent_capability", _capability)
    outcome = run_dock.SimRunService(tmp_path).run_preset(GOLF)
    assert outcome.payload == body and outcome.error is None
    text = run_dock.summarize(outcome, max_lines=12)
    assert "3 points, apex 30.0 m" in text and "(200.0, 4.0)" in text
    assert "Limit: ODE reference only." in text
    assert len(run_dock.summarize(outcome, max_lines=2).splitlines()) == 2


def test_tools_menu_opens_a_tabbed_dock(make_panel: Any, monkeypatch: Any) -> None:
    from src.tools.model_explorer.gui import MainWidget

    monkeypatch.setattr(run_dock, "SimRunService", _FakeService)
    widget = MainWidget()
    actions = widget.findChildren(type(widget.export_video_action))
    action = next(a for a in actions if a.text() == "&Run Simulation...")
    action.trigger()
    dock = widget.run_dock
    action.trigger()  # reopening reuses the dock
    assert widget.run_dock is dock and dock.widget() is widget.run_panel
    assert widget.inner_main_window.tabifiedDockWidgets(dock) == [widget.segment_dock]
    widget.cleanup()
    widget.deleteLater()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"presets": []}, "presets"),
        ({"presets": [{"id": "nope", "label": "x"}]}, "unknown preset"),
        ({"presets": [{"id": GOLF, "label": " "}]}, "label"),
        ({"summary_max_lines": -1}, "summary_max_lines"),
        ({"playable_result_key": ""}, "playable_result_key"),
        ({"store_subdir": "../escape"}, "store_subdir"),
    ],
)
def test_settings_reject_bad_values(tmp_path: Path, overrides: Any, match: str) -> None:
    raw = json.loads(run_dock.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}, "undocumented key"
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({**raw, **overrides}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        run_dock.load_run_dock_settings(path)
