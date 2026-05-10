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

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_interchange.canonical import (
    CanonicalPose,
    canonical_from_reference_setup,
    canonical_zero_pose,
)
from src.tools.pose_studio.controllers import (
    EngineController,
    HistoryController,
)
from src.tools.pose_studio.core import SUPPORTED_ENGINES, EngineStatus
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


class PoseStudioWindow(QtWidgets.QMainWindow):
    """Top-level :class:`QMainWindow` for the Pose Studio tool."""

    def __init__(
        self,
        initial_engine: str = "drake",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if initial_engine not in SUPPORTED_ENGINES:
            initial_engine = SUPPORTED_ENGINES[0]
        self.setWindowTitle("Pose Studio")
        self.resize(1200, 800)

        self._engine_controller = EngineController(initial_engine)
        self._history = HistoryController(canonical_zero_pose())

        self._build_widgets(initial_engine)
        self._build_layout()
        self._build_menu()
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
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
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

        self.setCentralWidget(central)

    def _build_menu(self) -> None:
        # QMainWindow.menuBar() and QMenuBar.addMenu() return Optional in
        # the Qt stubs, but always return real objects on a constructed
        # main window. Pin them with asserts once here so mypy can follow
        # without scattering casts through every addAction call.
        menubar = self.menuBar()
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
        act_save = QtGui.QAction("&Save Pose...", self)
        act_save.setToolTip(_SAVE_TOOLTIP)
        act_save.triggered.connect(self._on_save_clicked)
        file_menu.addAction(act_save)

        act_load = QtGui.QAction("&Load Pose...", self)
        act_load.setToolTip(_LOAD_TOOLTIP)
        act_load.triggered.connect(self._on_load_clicked)
        file_menu.addAction(act_load)
        file_menu.addSeparator()
        act_quit = QtGui.QAction("&Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Edit menu.
        self.act_undo = QtGui.QAction("&Undo", self)
        self.act_undo.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        self.act_undo.triggered.connect(self._on_undo)
        edit_menu.addAction(self.act_undo)
        self.act_redo = QtGui.QAction("&Redo", self)
        self.act_redo.setShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"))
        self.act_redo.triggered.connect(self._on_redo)
        edit_menu.addAction(self.act_redo)

        # Pose Library menu.
        act_zero = QtGui.QAction("Load &Zero Pose", self)
        act_zero.setToolTip("Reset to canonical_zero_pose() (T-pose at origin).")
        act_zero.triggered.connect(self._on_load_zero)
        pose_menu.addAction(act_zero)

        act_ref = QtGui.QAction("Load &Reference Golfer Setup", self)
        act_ref.setToolTip(
            "Reset to canonical_from_reference_setup() (anatomical address)."
        )
        act_ref.triggered.connect(self._on_load_reference)
        pose_menu.addAction(act_ref)

        # View menu.
        self.act_show_radians = QtGui.QAction("Show angles in &radians", self)
        self.act_show_radians.setCheckable(True)
        self.act_show_radians.toggled.connect(self.joint_panel.set_show_radians)
        view_menu.addAction(self.act_show_radians)

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
        """Apply *pose* to controller, view, and joint panel.
        
        When a live kinematics service is available, the viewport renders
        from the service's computed transforms to show engine-specific
        kinematics, constraints, or convention differences. Otherwise,
        it renders from the canonical pose via forward_kinematics.
        """
        self._engine_controller.set_pose(pose)
        
        # Try to render from live service transforms if available
        service = self._engine_controller.service
        if service is not None and hasattr(service, 'get_link_transforms'):
            try:
                transforms = service.get_link_transforms()
                self.view_3d.update_from_service_transforms(transforms)
            except (NotImplementedError, AttributeError, RuntimeError):
                # Fall back to canonical forward kinematics
                self.view_3d.update_pose(pose)
        else:
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


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m src.tools.pose_studio``."""
    if argv is None:
        argv = sys.argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    win = PoseStudioWindow()
    win.show()
    return app.exec()


__all__ = ["PoseStudioWindow", "main"]
