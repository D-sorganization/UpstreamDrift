"""Export Clips dialog: pure request mapping and the offscreen dialog (#15150 G11)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from src.tools.native_viewer_export import dialog_core as dc
from src.tools.native_viewer_export.core import DEFAULT_SPEEDS, ENGINES

pytestmark = pytest.mark.unit


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "b.npz").write_bytes(b"")
    monkeypatch.setenv("UD_NATIVE_EXPORT_INPUT_ROOTS", str(tmp_path / "in"))
    monkeypatch.setenv("UD_NATIVE_EXPORT_OUTPUT_ROOT", str(tmp_path / "out"))
    return tmp_path


def _choices(root: Path, **changes: Any) -> dc.ClipChoices:
    base = dc.default_choices()
    fields = {"bundle": str(root / "in" / "b.npz"), "swing": "s1"}
    return dc.ClipChoices(**{**base.__dict__, **fields, **changes})


def test_defaults_come_from_schema_and_settings(roots: Path) -> None:
    choices = dc.default_choices()
    assert choices.engines == ENGINES
    assert choices.speeds == tuple(DEFAULT_SPEEDS)
    assert choices.impact_clip is False
    assert Path(choices.out).is_relative_to(roots / "out")
    assert dc.impact_window_seconds() > 0
    assert dc.load_dialog_settings()["sources"]


def test_request_carries_each_choice(roots: Path) -> None:
    choices = _choices(roots, engines=("mujoco",), speeds=(1.0, 0.25), impact_clip=True)
    options = dc.build_request(choices)["options"]
    assert options["engines"] == "mujoco"
    assert options["speeds"] == "1,0.25"
    assert options["impact_window"] == dc.impact_window_seconds()
    assert "impact_window" not in dc.build_request(_choices(roots))["options"]


def test_validate_returns_the_exporter_command_line(roots: Path) -> None:
    checked = dc.validate_choices(_choices(roots, engines=("mujoco",)))
    assert "--engines" in checked["argv"] and "mujoco" in checked["argv"]
    assert checked["command_line"].startswith("python3 -m")


@pytest.mark.parametrize(
    "changes",
    [
        {"engines": ()},
        {"speeds": ()},
        {"swing": ""},
        {"bundle": "/etc/passwd"},
        {"speeds": (9.0,)},
    ],
)
def test_invalid_choices_raise_value_error(roots: Path, changes: dict) -> None:
    with pytest.raises(ValueError):
        dc.validate_choices(_choices(roots, **changes))


def test_manifest_lines_list_clips_and_skips() -> None:
    manifest = {
        "engines": [
            {"engine": "mujoco", "skipped_reason": None, "clips": {"2x2": "/o/a.mp4"}},
            {"engine": "drake", "skipped_reason": "no pydrake", "clips": {}},
        ]
    }
    lines = dc.manifest_lines(manifest)
    assert lines == ["mujoco 2x2: /o/a.mp4", "drake skipped: no pydrake"]
    assert dc.manifest_lines(None) == []


class FakeQueue:
    def __init__(self) -> None:
        self.submitted: list[tuple[list[str], str]] = []
        self.states = ["running", "succeeded"]

    def submit(self, argv: list[str], command_line: str) -> dict[str, Any]:
        self.submitted.append((argv, command_line))
        return {"job_id": "export_x", "state": "queued"}

    def status(self, job_id: str) -> dict[str, Any]:
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        manifest = {"engines": [{"engine": "mujoco", "clips": {"2x2": "/o/a.mp4"}}]}
        return {
            "job_id": job_id,
            "state": state,
            "error": None,
            "manifest": manifest if state == "succeeded" else None,
        }


def _qt() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication

        from src.tools.native_viewer_export import export_dialog
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    return QApplication.instance() or QApplication([]), export_dialog


def test_dialog_submits_polls_and_lists_clips(roots: Path) -> None:
    _, module = _qt()
    queue = FakeQueue()
    dialog = module.ExportClipsDialog(queue=queue)
    dialog.set_choices(_choices(roots, engines=("mujoco",)))
    dialog.start_export()
    assert len(queue.submitted) == 1 and not dialog.export_button.isEnabled()
    dialog.poll_once()
    assert dialog.state_label.text().endswith("running")
    dialog.poll_once()
    assert dialog.state_label.text().endswith("succeeded")
    assert "mujoco 2x2: /o/a.mp4" in dialog.result_view.toPlainText()
    assert dialog.export_button.isEnabled() and not dialog.timer.isActive()


def test_dialog_reports_invalid_choice_without_submitting(roots: Path) -> None:
    _, module = _qt()
    queue = FakeQueue()
    dialog = module.ExportClipsDialog(queue=queue)
    dialog.set_choices(_choices(roots, swing=""))
    dialog.start_export()
    assert queue.submitted == [] and dialog.state_label.text().startswith("Invalid")


def test_model_explorer_tools_menu_opens_the_dialog() -> None:
    _, module = _qt()
    try:
        from src.tools.model_explorer.gui import MainWidget
    except (ImportError, OSError) as exc:
        pytest.skip(f"Model Explorer not loadable: {exc}")
    widget = MainWidget()
    titles = [a.text() for a in widget.findChildren(type(widget.export_video_action))]
    assert "Export Cl&ips..." in titles
    dialog = widget.open_export_clips()
    assert isinstance(dialog, module.ExportClipsDialog)
    dialog.close()
    widget.cleanup()
