"""Pose Studio main window.

Composes the pure-data controllers (:class:`EngineController`,
:class:`HistoryController`) with the per-component widgets
(:class:`EnginePicker`, :class:`JointPanel`, :class:`View3D`,
:class:`UnitsBadge`) into a single :class:`QMainWindow`.

This file is deliberately thin: layout + signal wiring only.  Math
lives in :mod:`src.tools.pose_studio.core` and the controllers, and
each widget owns its own internal state.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_interchange.canonical import (
    CanonicalPose,
    canonical_from_reference_setup,
    canonical_zero_pose,
)
from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
)
from src.tools.pose_studio.controllers import (
    EngineController,
    HistoryController,
)
from src.tools.pose_studio.core import SUPPORTED_ENGINES
from src.tools.pose_studio.widgets import (
    EnginePicker,
    JointPanel,
    UnitsBadge,
    View3D,
)

logger = get_logger(__name__)


_SAVE_TOOLTIP = (
    "Save formats coming in #4900 (Subtask 6 of EPIC #4895). Currently a stub."
)
_LOAD_TOOLTIP = (
    "Load formats coming in #4900 (Subtask 6 of EPIC #4895). Currently a stub."
)


class MainWidget(QtWidgets.QWidget):
    """Central widget containing the Pose Studio UI content.

    This widget contains all the Pose Studio UI components and can be
    embedded inside a QMainWindow shell or hosted directly by the launcher.

    See PoseStudioWindow for the standalone QMainWindow wrapper.
    """

    def __init__(
        self,
        initial_engine: str = "drake",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if initial_engine not in SUPPORTED_ENGINES:
            initial_engine = SUPPORTED_ENGINES[0]

        self._engine_controller = EngineController(initial_engine)
        self._history = HistoryController(canonical_zero_pose())

        self._build_widgets(initial_engine)
        self._build_layout()
        self._wire_signals()

        # Push the initial pose through the engine and refresh the view.
        self._apply_pose(canonical_zero_pose(), record_history=False)

    # ---- construction --------------------------------------------------

    def _build_widgets(self, initial_engine: str) -> None:
        self.engine_picker = EnginePicker(initial_engine)
        self.engine_picker.set_status(self._engine_controller.status)

        self.units_badge = UnitsBadge(initial_engine)

        self.joint_panel = JointPanel()
        self.view_3d = View3D()

        self.btn_save = QtWidgets.QPushButton("Save Pose...")
        self.btn_save.setToolTip(_SAVE_TOOLTIP)
        self.btn_save.clicked.connect(self._on_save_clicked)

        self.btn_load = QtWidgets.QPushButton("Load Pose...")
        self.btn_load.setToolTip(_LOAD_TOOLTIP)
        self.btn_load.clicked.connect(self._on_load_clicked)

        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_undo.setToolTip("Undo the last edit (Ctrl+Z).")
        self.btn_undo.clicked.connect(self._on_undo)

        self.btn_redo = QtWidgets.QPushButton("Redo")
        self.btn_redo.setToolTip("Redo the last undone edit (Ctrl+Shift+Z).")
        self.btn_redo.clicked.connect(self._on_redo)

    def _build_layout(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # Top bar: engine picker on the left, units badge on the right.
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self.engine_picker, stretch=1)
        top_bar.addWidget(self.units_badge)
        outer.addLayout(top_bar)

        # Body: 3D view on the left, joint panel on the right.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.view_3d)
        splitter.addWidget(self.joint_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, stretch=1)

        # Footer: undo/redo, save/load.
        footer = QtWidgets.QHBoxLayout()
        footer.addWidget(self.btn_undo)
        footer.addWidget(self.btn_redo)
        footer.addStretch(1)
        footer.addWidget(self.btn_load)
        footer.addWidget(self.btn_save)
        outer.addLayout(footer)

    def _wire_signals(self) -> None:
        self.engine_picker.engine_selected.connect(self._on_engine_selected)
        self.joint_panel.angle_edited.connect(self._on_angle_edited)

    # ---- handlers ------------------------------------------------------

    def _on_engine_selected(self, engine_name: str) -> None:
        status = self._engine_controller.switch_engine(engine_name)
        self.engine_picker.set_status(status)
        self.units_badge.set_engine(engine_name)
        self.view_3d.update_pose(self._engine_controller.pose)

    def _on_angle_edited(self, name: str, value_deg: float) -> None:
        current = self._engine_controller.pose
        new_angles = dict(current.joint_angles_deg)
        new_angles[name] = float(value_deg)
        try:
            new_pose = CanonicalPose(
                pelvis_translation_m=np.asarray(
                    current.pelvis_translation_m, dtype=float
                ),
                pelvis_rotation_xyz_deg=np.asarray(
                    current.pelvis_rotation_xyz_deg, dtype=float
                ),
                joint_angles_deg=new_angles,
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Pose edit rejected: %s", exc)
            return
        self._apply_pose(new_pose, record_history=True)

    def _on_undo(self) -> None:
        pose = self._history.undo()
        if pose is None:
            return
        self._apply_pose(pose, record_history=False)

    def _on_redo(self) -> None:
        pose = self._history.redo()
        if pose is None:
            return
        self._apply_pose(pose, record_history=False)

    def _on_load_zero(self) -> None:
        self._apply_pose(canonical_zero_pose(), record_history=True)

    def _on_load_reference(self) -> None:
        self._apply_pose(canonical_from_reference_setup(), record_history=True)

    def _on_save_clicked(self) -> None:
        QtWidgets.QToolTip.showText(
            self.btn_save.mapToGlobal(self.btn_save.rect().bottomLeft()),
            _SAVE_TOOLTIP,
            self.btn_save,
        )

    def _on_load_clicked(self) -> None:
        QtWidgets.QToolTip.showText(
            self.btn_load.mapToGlobal(self.btn_load.rect().bottomLeft()),
            _LOAD_TOOLTIP,
            self.btn_load,
        )

    # ---- core state plumbing -------------------------------------------

    def _apply_pose(self, pose: CanonicalPose, *, record_history: bool) -> None:
        """Apply *pose* to controller, view, and joint panel."""
        self._engine_controller.set_pose(pose)
        # Pass the service to the view so it can render engine-specific kinematics
        self.view_3d.set_service(self._engine_controller.service)
        self.view_3d.update_pose(pose)
        self.joint_panel.set_angles(pose.angles_full_dict_deg())
        if record_history:
            self._history.push(pose)
        self._refresh_undo_redo_actions()

    def _refresh_undo_redo_actions(self) -> None:
        self.btn_undo.setEnabled(self._history.can_undo)
        self.btn_redo.setEnabled(self._history.can_redo)
        self.act_undo.setEnabled(self._history.can_undo)
        self.act_redo.setEnabled(self._history.can_redo)

    def create_menu_bar(self, parent: QtWidgets.QMainWindow) -> QtGui.QMenuBar:
        """Create and return a menu bar for the given parent window.

        Args:
            parent: The QMainWindow that will host the menu bar.

        Returns:
            The created QMenuBar with all Pose Studio menus.
        """
        menubar = parent.menuBar()
        assert menubar is not None  # noqa: S101 — Qt invariant

        file_menu = menubar.addMenu("&File")
        edit_menu = menubar.addMenu("&Edit")
        pose_menu = menubar.addMenu("&Pose Library")
        view_menu = menubar.addMenu("&View")
        assert file_menu is not None  # noqa: S101 — Qt invariant
        assert edit_menu is not None  # noqa: S101 — Qt invariant
        assert pose_menu is not None  # noqa: S101 — Qt invariant
        assert view_menu is not None  # noqa: S101 — Qt invariant

        # File menu.
        act_save = QtGui.QAction("&Save Pose...", parent)
        act_save.setToolTip(_SAVE_TOOLTIP)
        act_save.triggered.connect(self._on_save_clicked)
        file_menu.addAction(act_save)

        act_load = QtGui.QAction("&Load Pose...", parent)
        act_load.setToolTip(_LOAD_TOOLTIP)
        act_load.triggered.connect(self._on_load_clicked)
        file_menu.addAction(act_load)
        file_menu.addSeparator()
        act_quit = QtGui.QAction("&Quit", parent)
        act_quit.triggered.connect(parent.close)
        file_menu.addAction(act_quit)

        # Edit menu.
        self.act_undo = QtGui.QAction("&Undo", parent)
        self.act_undo.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        self.act_undo.triggered.connect(self._on_undo)
        edit_menu.addAction(self.act_undo)
        self.act_redo = QtGui.QAction("&Redo", parent)
        self.act_redo.setShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"))
        self.act_redo.triggered.connect(self._on_redo)
        edit_menu.addAction(self.act_redo)

        # Pose Library menu.
        act_zero = QtGui.QAction("Load &Zero Pose", parent)
        act_zero.setToolTip("Reset to canonical_zero_pose() (T-pose at origin).")
        act_zero.triggered.connect(self._on_load_zero)
        pose_menu.addAction(act_zero)

        act_ref = QtGui.QAction("Load &Reference Golfer Setup", parent)
        act_ref.setToolTip(
            "Reset to canonical_from_reference_setup() (anatomical address)."
        )
        act_ref.triggered.connect(self._on_load_reference)
        pose_menu.addAction(act_ref)

        # View menu.
        self.act_show_radians = QtGui.QAction("Show angles in &radians", parent)
        self.act_show_radians.setCheckable(True)
        self.act_show_radians.toggled.connect(self.joint_panel.set_show_radians)
        view_menu.addAction(self.act_show_radians)

        return menubar


class PoseStudioWindow(QtWidgets.QMainWindow):
    """Top-level :class:`QMainWindow` for the Pose Studio tool.

    This QMainWindow wraps the MainWidget content and provides a standalone
    window for running Pose Studio independently of the launcher.
    """

    def __init__(
        self,
        initial_engine: str = "drake",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pose Studio")
        self.resize(1200, 800)

        # Create the main widget and set it as central widget
        self._main_widget = MainWidget(initial_engine, self)
        self.setCentralWidget(self._main_widget)

        # Create menu bar
        self._main_widget.create_menu_bar(self)

    @property
    def main_widget(self) -> MainWidget:
        """Return the embedded MainWidget."""
        return self._main_widget


class _EmbedAdapter:
    """Embed adapter for Pose Studio.

    Implements the EmbeddableTool protocol for the launcher to embed
    Pose Studio as a tab or dock widget.
    """

    tool_id = "pose_studio"

    def __init__(self) -> None:
        self._widget: MainWidget | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,  # tab is fine
            min_size=(640, 480),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Create and return the MainWidget for embedding.

        Args:
            parent: The intended Qt parent widget.

        Returns:
            MainWidget instance for embedding.
        """
        self._widget = MainWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        """Release any resources held by the embedded widget."""
        self._widget = None

    def is_dirty(self) -> bool:
        """Return True if the tool has unsaved state.

        Pose Studio does not currently track dirty state.
        """
        return False


# Register the embed adapter when this module is imported
register_embeddable_tool(_EmbedAdapter())


def get_dockable_ui() -> QtWidgets.QMainWindow:
    """Return the main window instance for docking in the unified launcher."""
    return PoseStudioWindow()


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m src.tools.pose_studio``."""
    if argv is None:
        argv = sys.argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    win = PoseStudioWindow()
    win.show()
    return app.exec()


__all__ = ["PoseStudioWindow", "main"]
