"""Desktop Necromatcher uses the actual library; GUI dependencies are optional."""

import pytest
from src.shared.python.workspace import NecromatcherLibrary

pytestmark = pytest.mark.unit


def test_native_fit_projection_keeps_qt_responsive_and_discards_old_frames(
    tmp_path, monkeypatch
):
    import threading

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtCore import QBuffer, QIODevice, QTimer
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from src.tools.necromatcher import gui

    app = QApplication.instance() or QApplication([])
    image = QImage(32, 32, QImage.Format.Format_RGB32)
    image.fill(0)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    original_bytes = bytes(buffer.data())
    entered, release = threading.Event(), threading.Event()
    calls, ticks = [], []

    class Review:
        def frame(self, index):
            return {
                "frame_count": 2,
                "frame": {
                    "pts_ticks": index,
                    "timebase_numerator": 1,
                    "timebase_denominator": 10,
                },
                "observation": {"landmarks": {}, "status": "missing"},
            }

        def image(self, index):
            return original_bytes

        def close(self):
            pass

    def project(library, fit_id, index):
        calls.append(index)
        if index == 0:
            entered.set()
            if not release.wait(2):
                raise RuntimeError("Projection test was not released")
        return {
            "fit_id": fit_id,
            "frame_index": index,
            "frame": Review().frame(index)["frame"],
            "points": {"wrist": {"x": 16, "y": 16, "visibility": None}},
        }

    monkeypatch.setattr(
        gui.NativeFitProjectionProcess,
        "project",
        lambda service, fit_id, index: project(None, fit_id, index),
    )
    widget = gui.NecromatcherWidget(
        library=NecromatcherLibrary.create(tmp_path / "library")
    )
    try:
        widget._review = Review()
        widget._fit_id = "fit-v2"
        widget.slider.setRange(0, 1)
        widget._show_frame(0)
        assert entered.wait(1)
        QTimer.singleShot(0, lambda: ticks.append(threading.get_ident()))
        app.processEvents()
        assert ticks == [threading.get_ident()]
        widget.slider.setValue(1)
        release.set()
        assert widget._worker.wait(2)
        widget._poll()
        assert widget.image.pixmap() is None or widget.image.pixmap().isNull()
        assert widget._worker.wait(2)
        widget._poll()
        assert calls == [0, 1]
        assert "Frame 2/2" in widget.status.text()
        assert "Native Projection" in widget.status.text()
        assert Review().image(0) == original_bytes
    finally:
        release.set()
        widget.cleanup()
        widget.close()


def test_library_operation_keeps_qt_responsive_and_completes_on_owner_thread(
    tmp_path, monkeypatch
):
    import threading

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    from src.tools.necromatcher.gui import NecromatcherWidget

    app = QApplication.instance() or QApplication([])
    widget = NecromatcherWidget(
        library=NecromatcherLibrary.create(tmp_path / "library")
    )
    owner = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    callbacks, ticks = [], []

    def operation():
        entered.set()
        if not release.wait(2):
            raise RuntimeError("Test did not release background operation")
        return threading.get_ident()

    try:
        widget._run(
            operation, lambda worker: callbacks.append((worker, threading.get_ident()))
        )
        assert entered.wait(1)
        QTimer.singleShot(0, lambda: ticks.append(threading.get_ident()))
        app.processEvents()
        assert ticks == [owner]
        assert callbacks == []
        release.set()
        assert widget._worker.wait(2)
        widget._poll()
        assert len(callbacks) == 1
        worker_thread, callback_thread = callbacks[0]
        assert worker_thread != owner
        assert callback_thread == owner
    finally:
        release.set()
        widget.cleanup()
        widget.close()


def test_desktop_player_and_swing_recall(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    from src.tools.necromatcher.gui import NecromatcherWidget

    app = QApplication.instance() or QApplication([])
    library = NecromatcherLibrary.create(tmp_path / "library")
    library.add_player("hogan", "Ben Hogan")
    library.add_swing("practice", "hogan", "Practice Swing")
    widget = NecromatcherWidget(library=library)
    try:
        assert widget.player_list.count() == 1
        widget.player_list.setCurrentRow(0)
        app.processEvents()
        assert widget.swing_list.count() == 1
        assert widget.swing_list.item(0).text() == "Practice Swing"
    finally:
        widget.cleanup()
        widget.close()


def test_native_model_import_persists_candidate_version(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication, QFileDialog, QInputDialog
    from src.tools.necromatcher.gui import NecromatcherWidget

    app = QApplication.instance() or QApplication([])
    library = NecromatcherLibrary.create(tmp_path / "library")
    library.add_player("hogan", "Ben Hogan")
    library.add_swing("practice", "hogan", "Practice")
    source = tmp_path / "candidate.xml"
    source.write_text('<mujoco model="unit-candidate"/>', encoding="utf-8")
    choices = iter([("Native Model", True), ("mujoco", True)])
    texts = iter([("candidate-v1", True), ("hip", True)])
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: next(choices))
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: next(texts))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    widget = NecromatcherWidget(library=library)
    try:
        widget.player_list.setCurrentRow(0)
        widget.swing_list.setCurrentRow(0)
        widget._import_version()
        assert widget._worker.wait(2)
        app.processEvents()
        widget._poll()
        recalled = NecromatcherLibrary(library.root).load_asset("candidate-v1")
        assert recalled.metadata["qualification"] == "unqualified_candidate"
        assert widget.asset_list.count() == 1
    finally:
        widget.cleanup()
        widget.close()


def test_background_failure_is_reported_without_touching_qt_in_worker(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from src.tools.necromatcher.gui import NecromatcherWidget

    app = QApplication.instance() or QApplication([])
    widget = NecromatcherWidget(
        library=NecromatcherLibrary.create(tmp_path / "library")
    )
    called = []

    def fail():
        raise KeyError("missing-capture")

    try:
        widget._run(fail, called.append)
        widget._worker.wait(2)
        app.processEvents()
        widget._poll()
        assert "missing-capture" in widget.status.text()
        assert called == []
    finally:
        widget.cleanup()
        widget.close()
