"""Reviewed fitting windows remain source-owned and are read off the Qt path."""

from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.workspace.test_necromatcher_source_scope import make_scope

pytestmark = pytest.mark.unit


def test_native_raw_review_uses_same_portable_registration(
    tmp_path: Any, monkeypatch: Any
) -> None:
    pytest.importorskip("PyQt6.QtWidgets")
    from src.tools.necromatcher import refit_dialog as module

    _, scope = make_scope(tmp_path)
    raw = b'{"schema":"necromatcher/source-fit-scope-review/1"}\n'
    path = tmp_path / "raw.json"
    path.write_bytes(raw)
    library = SimpleNamespace()
    calls: list[Any] = []
    monkeypatch.setattr(
        module,
        "import_fit_source_scope_review",
        lambda *args: calls.append(args) or scope,
        raising=False,
    )
    result = module.read_reviewed_source_scope(
        path, SimpleNamespace(library=library), "parent"
    )
    assert result == scope
    assert calls == [(library, "parent", raw)]


def test_native_export_recall_discloses_review_and_actual_domain(tmp_path: Any) -> None:
    pytest.importorskip("PyQt6.QtWidgets")
    from src.tools.necromatcher.video_dialog import VideoExportDialog
    from tests.unit.tools.test_necromatcher_video_dialog import _stored_overlay_probe

    _, scope = make_scope(tmp_path)
    probe = _stored_overlay_probe()
    probe.run["source_fit_scope"] = scope.to_record()
    probe.run["source_fit_scope_binding"] = {
        "frame_indices": [0, 3],
        "first_pts": [100, 30],
        "last_pts": [103, 30],
        "source_clock_sha256": scope.source_clock_sha256,
    }
    VideoExportDialog._render(probe)
    assert "0 to 4 (Exclusive)" in probe.status.text
    assert "Frames 0 to 3; Source PTS 100/30 to 103/30" in probe.status.text


def test_native_scope_import_is_deferred_and_inherited_scope_visible(
    tmp_path: Any, monkeypatch: Any
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    from src.tools.necromatcher import refit_dialog as module
    import json

    app = widgets.QApplication.instance() or widgets.QApplication([])
    _, scope = make_scope(tmp_path)
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope.to_record()), encoding="utf-8")
    tasks: list[Any] = []
    calls: list[Any] = []

    class Worker:
        def __init__(self) -> None:
            self.finished = SimpleNamespace(connect=lambda _slot: None)
            self.error = SimpleNamespace(connect=lambda _slot: None)

        def start(self) -> None:
            pass

    monkeypatch.setattr(
        module, "get_worker_adapter", lambda task, **_kw: tasks.append(task) or Worker()
    )
    monkeypatch.setattr(
        widgets.QFileDialog, "getOpenFileName", lambda *_args: (str(path), "")
    )
    dialog = module.ResearchRefitDialog(
        "parent",
        {
            "frame_indices": [0, 3],
            "coordinate_order": ["hip"],
            "coordinate_units": ["rad"],
            "recorded_options": None,
            "source_scope": scope.to_record(),
        },
        SimpleNamespace(
            submit=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or {"run_id": "r"}
            )
        ),
    )
    assert "0 to 4 (Exclusive)" in dialog.scope_status.text()
    dialog.scope_import.click()
    assert "validation pending" in dialog.scope_status.text()
    dialog.identity.setText("new")
    dialog.scales.setText("1")
    dialog.start.click()
    assert not calls
    result = tasks[0]()
    assert result == {"run_id": "r"}
    assert len(calls[0][0]) == 3
    assert calls[0][1] == {"source_scope": scope}
    dialog._worker = None
    dialog._remove_scope()
    assert "0 to 4 (Exclusive)" in dialog.scope_status.text()
    calls.clear()
    dialog._start()
    tasks[-1]()
    assert calls[0][1] == {}
    dialog._worker = None
    dialog._scope_path = path
    dialog._scope_owner = "foreign"
    count = len(tasks)
    dialog._start()
    assert len(tasks) == count
    assert "another source fit" in dialog.status.text()
    dialog._worker = None
    dialog.close()
    app.processEvents()


def test_native_scope_reader_rejects_nonobject(tmp_path: Any) -> None:
    pytest.importorskip("PyQt6.QtWidgets")
    from src.tools.necromatcher.refit_dialog import read_reviewed_source_scope

    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_reviewed_source_scope(path)
    assert read_reviewed_source_scope(None) is None
