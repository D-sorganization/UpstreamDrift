"""Run Simulation dock (#15150 R1): run a pinned sim_run case in a QThread
through the service the MCP tools call (``engines_for_describe``,
``run_agent_capability``); a playable result archive goes to ``load_results``."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast, get_args

from PyQt6 import QtWidgets as Qw, sip
from PyQt6.QtCore import QStandardPaths, QThread, pyqtSignal
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.shared.python.logging_pkg.logging_config import get_logger
from src.sim_run.agent_capabilities import CapabilityId
from src.tools.model_explorer.results_loader_worker import ResultsLoadSettings

logger = get_logger(__name__)
SETTINGS_PATH = Path(__file__).with_name("run_dock_settings.json")
#: Failures a service call reports as ``failed`` instead of aborting the thread.
CALL_ERRORS = (OSError, ImportError, RuntimeError, ValueError, TypeError, KeyError)


@dataclass(frozen=True)
class RunDockSettings:
    """Validated settings (``sources`` in the JSON documents each key);
    ``presets`` are ``(sim_run capability id, label)`` pairs."""

    presets: tuple[tuple[str, str], ...]
    store_subdir: str
    playable_result_key: str
    summary_max_lines: int

    def __post_init__(self) -> None:
        if not self.presets:
            raise ValueError("presets must list at least one pinned case")
        for preset_id, label in self.presets:
            if preset_id not in get_args(CapabilityId):
                raise ValueError(f"unknown preset {preset_id!r}")
            if not label.strip():
                raise ValueError(f"preset {preset_id} needs a label")
        sub = PurePosixPath(self.store_subdir)
        if not sub.parts or ".." in sub.parts or sub.is_absolute():
            raise ValueError(f"store_subdir must be relative: {self.store_subdir!r}")
        if not self.playable_result_key.strip():
            raise ValueError("playable_result_key must be non-empty")
        if self.summary_max_lines <= 0:
            raise ValueError(f"summary_max_lines must be positive: {self}")


def load_run_dock_settings(path: Path | None = None) -> RunDockSettings:
    """Read the settings; ``ValueError`` on an invalid value."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return RunDockSettings(
        tuple((str(p["id"]), str(p["label"])) for p in raw["presets"]),
        str(raw["store_subdir"]),
        str(raw["playable_result_key"]),
        int(raw["summary_max_lines"]),
    )


@dataclass(frozen=True)
class RunOutcome:
    """A v1 capability run record and its resolved result payload."""

    record: dict[str, Any]
    payload: dict[str, Any]

    @property
    def error(self) -> str | None:
        if self.record.get("status") == "succeeded":
            return None
        return str((self.record.get("error") or {}).get("message", "run failed"))


class SimRunService:
    """The sim_run Agent API service as the MCP tools call it (faked in tests:
    ``describe_engines() -> list[dict]``, ``run_preset(id) -> RunOutcome``)."""

    def __init__(self, store_root: Path | None = None) -> None:
        if store_root is None:
            base = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.GenericDataLocation
            )
            store_root = Path(base) / load_run_dock_settings().store_subdir
        self._store_root = store_root
        self._store: Any = None

    def describe_engines(self) -> list[dict[str, Any]]:
        from src.sim_run.agent_api import engines_for_describe

        return engines_for_describe()

    def run_preset(self, preset_id: str) -> RunOutcome:
        """Run one pinned case; a spilled payload is read back from the store."""
        from src.sim_run.agent_capabilities import run_agent_capability
        from src.sim_run.store import SimulationRunStore

        if self._store is None:
            self._store = SimulationRunStore(results_root=self._store_root)
        record = run_agent_capability(cast(CapabilityId, preset_id), {}, self._store)
        delivery = record.get("delivery") or {}
        if delivery.get("encoding") != "artifact":
            return RunOutcome(record, dict(delivery.get("value") or {}))
        ref = delivery["artifact_ref"]
        run_dir = self._store.get_run_dir(ref["run_id"])
        text = (run_dir / "artifacts" / ref["artifact_name"]).read_text("utf-8")
        return RunOutcome(record, dict(json.loads(text)))


def summarize(outcome: RunOutcome, max_lines: int) -> str:
    """Human-readable result lines (at most ``max_lines``)."""
    body = outcome.payload
    lines = [
        f"Run: {outcome.record.get('run_id', '?')}",
        f"Golden case: {body.get('golden_case_id', '?')}",
    ]
    if trajectory := (body.get("simulation") or {}).get("trajectory"):
        final = trajectory[-1]["position"]
        apex = max(point["position"][2] for point in trajectory)
        lines.append(f"Ball flight: {len(trajectory)} points, apex {apex:.1f} m")
        lines.append(f"Final position: ({final[0]:.1f}, {final[1]:.1f}) m")
    if analysis := body.get("analysis"):
        lines.append(f"ZTCF/ZVCF split: {len(analysis.get('t', []))} samples")
    limits = (body.get("qualification_banner") or {}).get("model_limits") or []
    lines.extend(f"Limit: {text}" for text in limits)
    return "\n".join(lines[:max_lines])


_LIVE: set[_ServiceCall] = set()


def live_call_count() -> int:
    """Service calls still running (or not yet retired), app-wide."""
    return len(_LIVE)


class _ServiceCall(QThread):
    """Run one service call off the GUI thread: ``succeeded`` or ``failed``."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, call: Callable[[], object]) -> None:
        super().__init__()
        self._call = call
        _LIVE.add(self)  # outlive the panel: a QThread collected running aborts
        self.finished.connect(self._retire)

    def _retire(self) -> None:
        _LIVE.discard(self)
        self.deleteLater()

    def run(self) -> None:
        try:
            result = self._call()
        except CALL_ERRORS as exc:
            logger.warning("Simulation service call failed: %s", exc)
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.succeeded.emit(result)


class RunSimulationPanel(QWidget):
    """Engines, a pinned-case picker, Run, status and summary; ``service`` is a
    :class:`SimRunService` (or fake), ``host`` offers ``load_results``."""

    def __init__(self, service: Any, host: Any = None) -> None:
        super().__init__()
        self._service, self._host = service, host
        self._settings = load_run_dock_settings()
        self._run_call: _ServiceCall | None = None
        self._closed = False
        self.engine_list = Qw.QListWidget()
        self.preset_combo = Qw.QComboBox()
        for preset_id, label in self._settings.presets:
            self.preset_combo.addItem(label, preset_id)
        self.run_button = Qw.QPushButton("Run")
        self.run_button.clicked.connect(self.run_selected)
        self.progress = Qw.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(ResultsLoadSettings().progress_width_px)
        self.progress.setVisible(False)
        self.status_label = QLabel("Checking engines...")
        self.status_label.setWordWrap(True)
        self.summary_view = Qw.QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        layout = QVBoxLayout(self)
        widgets: tuple[QWidget, ...] = (
            *(QLabel("Engines"), self.engine_list, QLabel("Pinned case")),
            *(self.preset_combo, self.run_button, self.progress, self.status_label),
        )
        for widget in widgets:
            layout.addWidget(widget)
        layout.addWidget(self.summary_view, 1)
        self._start(service.describe_engines, self._show_engines)

    @property
    def is_running(self) -> bool:
        return self._run_call is not None

    def _start(
        self, call: Callable[[], object], on_done: Callable[[Any], None]
    ) -> _ServiceCall:
        worker = _ServiceCall(call)
        worker.succeeded.connect(lambda result: self._deliver(on_done, result))
        worker.failed.connect(lambda text: self._deliver(self._on_failed, text))
        worker.start()
        return worker

    def _deliver(self, slot: Callable[[Any], None], value: object) -> None:
        """Run ``slot`` only while the panel is alive (a call can outlive it)."""
        if not (self._closed or sip.isdeleted(self)):
            slot(value)

    def _show_engines(self, engines: list[dict[str, Any]]) -> None:
        for row in engines:
            item = Qw.QListWidgetItem(f"{row['id']}: {row['runtime_status']}")
            item.setToolTip(str(row.get("unavailable_reason") or "ready"))
            self.engine_list.addItem(item)
        row_px = self.engine_list.sizeHintForRow(0)
        height = row_px * max(len(engines), 1) + 2 * self.engine_list.frameWidth()
        self.engine_list.setFixedHeight(height)
        if not self.is_running:
            self.status_label.setText("Ready")

    def run_selected(self) -> None:
        """Run the chosen pinned case in the background (no-op while busy)."""
        preset_id = self.preset_combo.currentData()
        if self.is_running or self._closed or not preset_id:
            return
        self.run_button.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText(f"Running {self.preset_combo.currentText()}...")
        self.summary_view.clear()
        self._run_call = self._start(
            lambda: self._service.run_preset(str(preset_id)), self._on_run_done
        )

    def _finish_run(self, status: str) -> None:
        self._run_call = None
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(status)

    def _on_failed(self, text: str) -> None:
        self._finish_run(f"Error: {text}")

    def _on_run_done(self, outcome: RunOutcome) -> None:
        if outcome.error is not None:
            self._finish_run(f"Run failed: {outcome.error}")
            return
        self.summary_view.setPlainText(
            summarize(outcome, self._settings.summary_max_lines)
        )
        raw = outcome.payload.get(self._settings.playable_result_key)
        playable = Path(raw) if isinstance(raw, str) and raw.endswith(".npz") else None
        if playable is None or not playable.is_file() or self._host is None:
            self._finish_run("Run succeeded; summary below.")
            return
        self._finish_run(f"Run succeeded; playing {playable.name}.")
        self._host.load_results(playable, wait=False)

    def cleanup(self) -> None:
        """Stop reacting to calls in flight (they retire themselves). Idempotent."""
        if self._closed:
            return
        self._closed, self._run_call = True, None
