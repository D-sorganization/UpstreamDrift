"""Native historical library review backed by the shared workspace spine."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.theme.tool_stylesheet import apply_tool_theme
from src.shared.python.ui.adapters import BackgroundWorker, get_worker_adapter
from src.shared.python.workspace import (
    CaptureReview,
    NecromatcherLibrary,
    default_necromatcher_library,
)
from .image_review import landmark_overlay


class NecromatcherWidget(QWidget):
    """Recall actual players, versions and source imagery without fitting claims."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        library: NecromatcherLibrary | None = None,
    ) -> None:
        super().__init__(parent)
        self.library = library or default_necromatcher_library()
        self._review: CaptureReview | None = None
        self._worker: BackgroundWorker | None = None
        self._completed: Callable[[Any], None] | None = None
        self._generation = 0
        self._closed = False
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        apply_tool_theme(self)
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Necromatcher — Historical Swing Library"))
        layout.addWidget(
            QLabel(
                "Source images are observations. Stored models remain unqualified candidates."
            )
        )
        actions = QHBoxLayout()
        for name, callback in (
            ("Add Player", self._add_player),
            ("Add Swing", self._add_swing),
            ("Import Version", self._import_version),
            ("Export Swing", self._export),
        ):
            button = QPushButton(name)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        lists = QHBoxLayout()
        self.player_list = QListWidget()
        self.swing_list = QListWidget()
        self.asset_list = QListWidget()
        for widget in (self.player_list, self.swing_list, self.asset_list):
            lists.addWidget(widget)
        layout.addLayout(lists)
        self.player_list.currentItemChanged.connect(self._players_changed)
        self.swing_list.currentItemChanged.connect(self._swings_changed)
        self.asset_list.currentItemChanged.connect(self._assets_changed)
        self.image = QLabel("Select a capture to inspect its original frames")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(320, 240)
        layout.addWidget(self.image, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_frame)
        layout.addWidget(self.slider)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    @staticmethod
    def _id(widget: QListWidget) -> str | None:
        item = widget.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    @staticmethod
    def _item(widget: QListWidget, name: str, identity: str) -> None:
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, identity)
        widget.addItem(item)

    def refresh(self) -> None:
        selected = self._id(self.player_list)
        self.player_list.clear()
        for player in self.library.players():
            self._item(self.player_list, player.display_name, player.subject_id)
            if selected == player.subject_id:
                self.player_list.setCurrentRow(self.player_list.count() - 1)

    def _players_changed(self) -> None:
        self.swing_list.clear()
        player = self._id(self.player_list)
        if player:
            for swing in self.library.swings(player):
                self._item(self.swing_list, swing.name, swing.session_id)

    def _swings_changed(self) -> None:
        self.asset_list.clear()
        swing = self._id(self.swing_list)
        if swing:
            for asset in self.library.assets(swing):
                qualification = asset.metadata.get("qualification", "Unqualified")
                self._item(
                    self.asset_list,
                    f"{asset.dataset_id} · {asset.kind} · {qualification}",
                    asset.dataset_id,
                )

    def _assets_changed(self) -> None:
        self._generation += 1
        self.slider.setEnabled(False)
        self.image.clear()
        if self._review:
            self._review.close()
            self._review = None
        identity = self._id(self.asset_list)
        swing = self._id(self.swing_list)
        if not identity or not swing:
            return
        asset = next(x for x in self.library.assets(swing) if x.dataset_id == identity)
        if asset.kind != "image_capture":
            self.status.setText(
                "Saved version; native accuracy and dynamics require independent qualification."
            )
            return
        generation = self._generation
        self._run(
            lambda: CaptureReview(self.library, identity),
            lambda review: self._capture_loaded(review, generation),
        )

    def _capture_loaded(self, review: CaptureReview, generation: int) -> None:
        if generation != self._generation or self._closed:
            review.close()
            if not self._closed:
                self._assets_changed()
            return
        self._review = review
        self.slider.setRange(0, review.frame_count - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self._show_frame(0)

    def _show_frame(self, index: int) -> None:
        if not self._review:
            return
        try:
            frame = self._review.frame(index)
            pixmap = QPixmap()
            if not pixmap.loadFromData(self._review.image(index), "PNG"):
                raise ValueError("Stored source image cannot be decoded")
            pixmap = landmark_overlay(pixmap, frame["observation"]["landmarks"])
            self.image.setPixmap(
                pixmap.scaled(
                    self.image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            clock = frame["frame"]
            self.status.setText(
                f"Frame {index + 1}/{frame['frame_count']} · PTS {clock['pts_ticks']} × {clock['timebase_numerator']}/{clock['timebase_denominator']} s · Physical Time Unknown · {frame['observation']['status']}"
            )
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            self.image.clear()
            self.status.setText(str(exc))

    def _run(self, target: Callable[[], Any], completed: Callable[[Any], None]) -> None:
        if self._worker:
            self.status.setText(
                "A library operation is running; retry when it finishes."
            )
            return

        def guarded() -> Any:
            try:
                return target()
            except (KeyError, IndexError, TypeError) as exc:
                # Canonical worker reports ValueError/OSError/RuntimeError.
                raise ValueError(str(exc)) from exc

        self._worker = get_worker_adapter(guarded, force_threading=True)
        self._completed = completed
        self.status.setText("Verifying Library Bytes…")
        self._worker.start()
        self._timer.start()

    def _poll(self) -> None:
        worker = self._worker
        if not worker or worker.is_running():
            return
        self._timer.stop()
        self._worker = None
        completed, self._completed = self._completed, None
        if worker.error:
            self.status.setText(str(worker.error))
        elif completed:
            completed(worker.result)

    def _identity(self, title: str) -> tuple[str, str] | None:
        identity, accepted = QInputDialog.getText(self, title, "Permanent ID")
        if not accepted or not identity:
            return None
        name, accepted = QInputDialog.getText(self, title, "Display Name")
        return (identity, name) if accepted and name else None

    def _add_player(self) -> None:
        values = self._identity("Add Player")
        if values:
            self._run(
                lambda: self.library.add_player(*values), lambda _: self.refresh()
            )

    def _add_swing(self) -> None:
        player = self._id(self.player_list)
        if not player:
            self.status.setText("Select a player first.")
            return
        values = self._identity("Add Swing")
        if values:
            self._run(
                lambda: self.library.add_swing(values[0], player, values[1]),
                lambda _: self._players_changed(),
            )

    def _export(self) -> None:
        swing = self._id(self.swing_list)
        if not swing:
            self.status.setText("Select a swing first.")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export Swing", "", "Swing Package (*.zip)"
        )
        if destination:
            self._run(
                lambda: self.library.export_swing(swing, Path(destination)),
                lambda _: self.status.setText("Verified Swing Package Saved"),
            )

    def _import_version(self) -> None:
        """Save one immutable candidate, capture or authored control version."""
        swing = self._id(self.swing_list)
        if not swing:
            self.status.setText("Select a swing first.")
            return
        kind, accepted = QInputDialog.getItem(
            self,
            "Import Version",
            "Evidence Type",
            ["Capture Folder", "Native Model", "Torque Profile"],
            0,
            False,
        )
        if not accepted:
            return
        identity, accepted = QInputDialog.getText(
            self, "Import Version", "Permanent Version ID"
        )
        if not accepted or not identity:
            return
        if kind == "Capture Folder":
            source = QFileDialog.getExistingDirectory(self, "Completed Capture Folder")
        else:
            source, _ = QFileDialog.getOpenFileName(self, "Version Source File")
        if not source:
            return
        target: Callable[[], Any]
        if kind == "Capture Folder":
            target = partial(self.library.add_capture, identity, swing, Path(source))
        elif kind == "Torque Profile":
            target = partial(self.library.add_profile, identity, swing, Path(source))
        else:
            engine, accepted = QInputDialog.getItem(
                self,
                "Model Candidate",
                "Native Engine",
                ["mujoco", "drake", "pinocchio", "opensim", "simscape"],
                0,
                False,
            )
            if not accepted:
                return
            dofs, accepted = QInputDialog.getText(
                self, "Model Candidate", "Ordered DOFs (Comma Separated)"
            )
            if not accepted:
                return
            ordered = tuple(x.strip() for x in dofs.split(","))
            target = partial(
                self.library.add_model,
                identity,
                swing,
                Path(source),
                engine=engine,
                dofs=ordered,
            )
        self._run(target, lambda _: self._swings_changed())

    def cleanup(self) -> None:
        """Drain owned I/O before releasing archive and worker handles."""
        self._closed = True
        self._timer.stop()
        if self._worker:
            self._worker.wait()
            result = self._worker.result
            if isinstance(result, CaptureReview):
                result.close()
            self._worker = None
        if self._review:
            self._review.close()
            self._review = None
