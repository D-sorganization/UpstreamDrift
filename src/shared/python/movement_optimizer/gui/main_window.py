# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""PyQt6 GUI for the Movement Optimizer.

Layout:
    Left sidebar  -- body parameters, segment multipliers, barbell,
                     optimisation controls, smoothness slider, results.
    Right area    -- tabbed exercise views (Squat, Full Squat, Deadlift)
                     with matplotlib animation + analysis plots.
    Bottom bar    -- playback controls and status.

Design Principles:
    LoD  -- each widget class manages its own state and signals.
    DRY  -- shared helpers (style_axis, Palette) imported from rendering.
    DBC  -- methods document and check their preconditions.
"""

from __future__ import annotations

import logging
import threading
import traceback
from typing import Any

import matplotlib
from PyQt6.QtCore import QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QWidget,
)

from ..comparison import ComparisonStore
from ..persistence import InvalidStateFileError, load_app_state, save_app_state
from ..rendering import ThemedWindowMixin, refresh_palette, restyle_figure
from ..trajectory import SolutionCache
from .animation_control import AnimationControlMixin
from .app_icon import movement_optimizer_icon
from .commands import SliderChangeCommand, UndoStack
from .comparison_mixin import ComparisonMixin
from .exercise_state import ExerciseRuntimeState
from .file_operations import FileOperationsMixin
from .help_dialog import HELP_TOPICS, HelpCenterDialog
from .motion_tabs import create_chain_tab, create_swingset_tab
from .optimization_mixin import OptimizationMixin
from .session_state import collect_results, collect_slider_values, restore_slider_values
from .stylesheet import build_qss
from .ui_builder import build_central_widget

try:
    matplotlib.use("QtAgg")
except ImportError:
    matplotlib.use("Agg")

logger = logging.getLogger(__name__)


# ==============================================================
# Labelled Slider Widget
# ==============================================================


# ==============================================================
# Comparison Dialog
# ==============================================================


class MainWindow(
    ThemedWindowMixin,
    FileOperationsMixin,
    AnimationControlMixin,
    ComparisonMixin,
    OptimizationMixin,
    QMainWindow,
):
    """Top-level application window.

    File I/O, animation playback, optimization, and comparison actions are provided
    by mixin classes in their respective submodules.
    """

    # Signals for thread-safe GUI updates from the optimizer worker.
    # Using signals instead of QTimer.singleShot is the Qt-correct way
    # to communicate from a background thread to the main thread.
    _sig_done = pyqtSignal(
        int, object, object, float, object
    )  # idx, result, body, bar, then_chain
    _sig_cancelled = pyqtSignal()
    _sig_error = pyqtSignal(object)  # MovementOptimizerError or str
    _sig_progress = pyqtSignal(object)  # ProgressReport

    EXERCISE_CONFIGS = (
        ("Bottoms Up Squat", "squat"),
        ("Full Squat", "full_squat"),
        ("Deadlift", "deadlift"),
        ("Bench Press", "bench_press"),
        ("Clean", "clean"),
        ("Jerk", "jerk"),
        ("Snatch", "snatch"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Movement Optimizer")
        self.setWindowIcon(movement_optimizer_icon())
        self.setMinimumSize(800, 600)
        self.resize(1100, 700)

        self.exercise_states = [
            ExerciseRuntimeState() for _name, _etype in self.EXERCISE_CONFIGS
        ]
        self.is_playing = False
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._anim_step)

        self._cancel_event = threading.Event()
        self._opt_running = False
        # RLock so the same thread may re-enter critical sections without
        # deadlocking when one locked helper calls another locked helper.
        self._opt_lock: threading.RLock = threading.RLock()
        self._cache = SolutionCache()
        self._comparison_store = ComparisonStore()
        self._last_config: tuple[Any, ...] = ()

        # Connect thread-safe signals
        self._sig_done.connect(self._on_done)
        self._sig_cancelled.connect(self._on_cancelled)
        self._sig_error.connect(self._on_err)
        self._sig_progress.connect(self._update_progress)

        self._settings = QSettings("D-sorganization", "Movement-Optimizer")

        self._undo_stack = UndoStack()
        self.action_undo: QAction | None = None
        self.action_redo: QAction | None = None
        self._motion_tab_button_states: dict[Any, bool] = {}

        self._build_ui()
        # Theme support pulls colours from the fleet shared theme and adds a
        # Theme menu. The shared base stylesheet is applied first; our
        # palette-derived stylesheet is layered on top to preserve the app's
        # bespoke widget styling (primary/cancel buttons, result labels, ...).
        self.setup_theme_support(add_menu=True, settings_app="Movement-Optimizer")
        refresh_palette()
        self.setStyleSheet(build_qss())
        self._restore_layout()
        self._sync_motion_tab_controls()
        self._connect_slider_undo()
        self._install_shortcuts()

    def _on_theme_changed(self, _theme_name: str) -> None:
        """Recolour the app when the shared theme changes.

        The theme manager applies its base stylesheet before emitting
        ``themeChanged``; here we refresh the palette-derived stylesheet, the
        motion-canvas palette, every matplotlib figure, and repaint all widgets.
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from .motion_tabs import refresh_motion_palette

        refresh_palette()
        refresh_motion_palette()
        self.setStyleSheet(build_qss())
        for canvas in self.findChildren(FigureCanvasQTAgg):
            restyle_figure(canvas.figure)
            canvas.draw_idle()
        for widget in self.findChildren(QWidget):
            widget.update()

    def _build_ui(self) -> None:
        (
            central,
            self.sidebar,
            self.tabs,
            self.exercise_tabs,
            self.controls,
            self.status_label,
            self._left_sidebar_toggle_btn,
            self._right_sidebar_toggle_btn,
        ) = build_central_widget(self, self.EXERCISE_CONFIGS)
        self._left_sidebar_toggle_btn.clicked.connect(self._toggle_sidebar)
        self._right_sidebar_toggle_btn.clicked.connect(self._toggle_right_sidebar)
        self.tabs.addTab(create_swingset_tab(), "  Swingset Model  ")
        self.tabs.addTab(create_chain_tab(), "  Chain Dynamics  ")
        for index in range(self.tabs.count() - 2, self.tabs.count()):
            signal = getattr(self.tabs.widget(index), "playbackStateChanged", None)
            if signal is not None:
                signal.connect(self._sync_analysis_playback_controls)
        self.tabs.currentChanged.connect(self._sync_motion_tab_controls)
        self.setCentralWidget(central)
        self._build_menu_bar()
        self._connect_signals()

    def _toggle_sidebar(self) -> None:
        """Collapse or expand the parameter sidebar."""
        visible = self.sidebar.isVisible()
        self.sidebar.setVisible(not visible)
        if visible:
            self._left_sidebar_toggle_btn.setText("Show left")
            self._left_sidebar_toggle_btn.setAccessibleName("Show left sidebar")
            self._left_sidebar_toggle_btn.setAccessibleDescription(
                "Show the left parameter sidebar."
            )
        else:
            self._left_sidebar_toggle_btn.setText("Hide left")
            self._left_sidebar_toggle_btn.setAccessibleName("Hide left sidebar")
            self._left_sidebar_toggle_btn.setAccessibleDescription(
                "Hide the left parameter sidebar."
            )

    def _toggle_right_sidebar(self) -> None:
        """Collapse or expand the active analysis tab's right parameter panel."""
        panel = self._active_right_panel()
        if panel is None:
            self._sync_right_sidebar_toggle()
            return
        panel.set_control_panel_visible(not panel.control_panel_visible())
        self._sync_right_sidebar_toggle()

    def _active_right_panel(self) -> Any | None:
        """Return the active tab if it exposes a right-side control panel."""
        tab = self._active_analysis_tab()
        if (
            tab is None
            or not hasattr(tab, "set_control_panel_visible")
            or not hasattr(tab, "control_panel_visible")
        ):
            return None
        return tab

    def _sync_right_sidebar_toggle(self) -> None:
        """Reflect the active analysis panel state in the top-strip toggle."""
        panel = self._active_right_panel()
        if panel is None:
            self._right_sidebar_toggle_btn.setEnabled(False)
            self._right_sidebar_toggle_btn.setText("Right panel")
            self._right_sidebar_toggle_btn.setAccessibleName("Right panel unavailable")
            self._right_sidebar_toggle_btn.setAccessibleDescription(
                "No active right parameter panel is available."
            )
            return
        visible = panel.control_panel_visible()
        if visible:
            self._right_sidebar_toggle_btn.setText("Hide right")
            self._right_sidebar_toggle_btn.setAccessibleName("Hide right panel")
            self._right_sidebar_toggle_btn.setAccessibleDescription(
                "Hide the active right parameter panel."
            )
        else:
            self._right_sidebar_toggle_btn.setText("Show right")
            self._right_sidebar_toggle_btn.setAccessibleName("Show right panel")
            self._right_sidebar_toggle_btn.setAccessibleDescription(
                "Show the active right parameter panel."
            )
        self._right_sidebar_toggle_btn.setEnabled(True)

    def _build_menu_bar(self) -> None:
        """Add application menus."""
        menu_bar = self.menuBar()
        if menu_bar is None:
            raise RuntimeError("MainWindow menu bar is unavailable")

        edit_menu = menu_bar.addMenu("&Edit")
        if edit_menu is None:
            raise RuntimeError("MainWindow Edit menu could not be created")
        self.action_undo = QAction("&Undo", self)
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.setStatusTip("Undo the last parameter change")
        self.action_undo.triggered.connect(self._undo)
        edit_menu.addAction(self.action_undo)

        self.action_redo = QAction("&Redo", self)
        self.action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.action_redo.setStatusTip("Redo the last undone parameter change")
        self.action_redo.triggered.connect(self._redo)
        edit_menu.addAction(self.action_redo)
        self._update_undo_redo_actions()

        help_menu = menu_bar.addMenu("&Help")
        if help_menu is None:
            raise RuntimeError("MainWindow Help menu could not be created")

        for topic_id, topic in HELP_TOPICS.items():
            action = QAction(f"&{topic.title}", self)
            action.setStatusTip(f"Open help topic: {topic.title}")
            action.triggered.connect(
                lambda _checked=False, tid=topic_id: self._show_help_topic(tid)
            )
            if topic_id == "parameters":
                action.setShortcut(QKeySequence("F1"))
            help_menu.addAction(action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.setStatusTip("Show information about Movement Optimizer")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self) -> None:
        """Display an About dialog with app name, version, and description."""
        from .. import __version__

        QMessageBox.about(
            self,
            "About Movement Optimizer",
            f"<b>Movement Optimizer</b> v{__version__}<br><br>"
            "Computes optimal barbell exercise trajectories using Lagrangian "
            "inverse dynamics in the sagittal plane.<br><br>"
            "The body is modelled as a 3-link planar chain (shank, thigh, trunk) "
            "with a barbell load at the top. Trajectories are found via multi-start "
            "parallel SLSQP optimisation subject to balance and joint constraints.<br><br>"
            "&#169; 2026 D-Sorganization. All rights reserved.",
        )

    def _show_help_topic(self, topic_id: str = "parameters") -> None:
        """Open the offline help center at a specific topic."""
        dlg = HelpCenterDialog(self, initial_topic=topic_id)
        dlg.exec()

    def _connect_signals(self) -> None:
        # Window-level Escape shortcut for cancel — works even when the cancel
        # button is hidden (QPushButton.setShortcut is inactive for hidden buttons).
        self._esc_shortcut = QShortcut(QKeySequence("Escape"), self)
        self._esc_shortcut.activated.connect(self._cancel_optimization)

        self.sidebar.connect_action_handlers(
            {
                "optimize_current": self._run_current,
                "optimize_both": self._run_all,
                "cancel_requested": self._cancel_optimization,
                "export_requested": self._export,
                "reset_requested": self._reset,
                "save_solution_requested": self._save_solution,
                "load_solution_requested": self._load_solution,
                "export_video_requested": self._export_video,
                "export_plots_requested": self._export_plots,
                "export_excel_requested": self._export_excel,
                "add_comparison_requested": self._add_comparison,
                "compare_trials_requested": self._compare_trials,
                "clear_comparison_requested": self._clear_comparison,
            }
        )
        self.controls.connect_action_handlers(
            {
                "play_toggled": self._toggle_play,
                "step_fwd": self._step_fwd,
                "step_back": self._step_back,
                "rewind": self._rewind,
                "jump_to_end": self._jump_to_end,
                "speed_changed": self._on_speed,
            }
        )

    def _install_shortcuts(self) -> None:
        """Install application-wide keyboard shortcuts."""
        QShortcut(QKeySequence("Space"), self).activated.connect(self._toggle_play)
        QShortcut(QKeySequence("Home"), self).activated.connect(self._rewind)
        QShortcut(QKeySequence("End"), self).activated.connect(self._jump_to_end)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_solution)

    def _connect_slider_undo(self) -> None:
        """Wire sidebar sliders so value changes are recorded on the undo stack."""
        slider_names = [
            "mass_slider",
            "height_slider",
            "ll_slider",
            "ul_slider",
            "to_slider",
            "bar_slider",
            "bar_depth_slider",
            "bar_height_slider",
            "dur_slider",
            "smooth_slider",
        ]
        for name in slider_names:
            labelled = getattr(self.sidebar, name, None)
            if labelled is None:
                logger.warning(
                    "_connect_slider_undo: sidebar has no attribute %r", name
                )
                continue
            raw = labelled.slider

            def _make_handler(raw_slider):
                prev: list[int] = [raw_slider.value()]

                def _on_press():
                    prev[0] = raw_slider.value()

                def _on_release():
                    new_val = raw_slider.value()
                    old_val = prev[0]
                    if new_val != old_val:
                        cmd = SliderChangeCommand(raw_slider, old_val, new_val)
                        self._undo_stack.record_executed(cmd)
                        self._update_undo_redo_actions()
                        logger.debug(
                            "Slider %s: %d -> %d recorded",
                            raw_slider.accessibleName(),
                            old_val,
                            new_val,
                        )

                return _on_press, _on_release

            on_press, on_release = _make_handler(raw)
            raw.sliderPressed.connect(on_press)
            raw.sliderReleased.connect(on_release)

    def _update_undo_redo_actions(self) -> None:
        """Keep Edit menu actions synchronized with the undo stack."""
        if self.action_undo is not None:
            self.action_undo.setEnabled(self._undo_stack.can_undo)
        if self.action_redo is not None:
            self.action_redo.setEnabled(self._undo_stack.can_redo)

    def _is_exercise_tab(self) -> bool:
        """Return true when the active top-level tab is a barbell exercise."""
        return self.tabs.currentIndex() < len(self.EXERCISE_CONFIGS)

    def _sync_motion_tab_controls(self, _index: int | None = None) -> None:
        """Disable barbell-only controls while an analysis tab is selected."""
        enabled = self._is_exercise_tab()
        if not enabled and (self.is_playing or self.anim_timer.isActive()):
            self._stop_anim()
        buttons = (
            self.sidebar.opt_btn,
            self.sidebar.both_btn,
            self.sidebar.export_btn,
            self.sidebar.save_btn,
            self.sidebar.export_video_btn,
            self.sidebar.export_plots_btn,
            self.sidebar.export_excel_btn,
            self.sidebar.add_compare_btn,
            self.sidebar.compare_btn,
        )
        if enabled:
            for button, prior_state in self._motion_tab_button_states.items():
                button.setEnabled(prior_state)
            self._motion_tab_button_states.clear()
        else:
            if not self._motion_tab_button_states:
                self._motion_tab_button_states = {
                    button: button.isEnabled() for button in buttons
                }
            for button in buttons:
                button.setEnabled(False)
        self.controls.setEnabled(True)
        if enabled:
            self.status_label.setText("Ready")
        else:
            self.status_label.setText(
                "Analysis tabs use local and bottom playback controls."
            )
        self._sync_right_sidebar_toggle()

    def _active_analysis_tab(self) -> Any | None:
        """Return the active analysis tab widget when one is selected."""
        if self._is_exercise_tab():
            return None
        return self.tabs.currentWidget()

    def _sync_analysis_playback_controls(self) -> None:
        """Reflect analysis tab playback state in the shared bottom controls."""
        tab = self._active_analysis_tab()
        if tab is None or not hasattr(tab, "playback_status"):
            return
        current, total, playing = tab.playback_status()
        self.controls.set_playing(playing)
        if total:
            self.controls.set_playback_status(
                current,
                total,
                self.controls.speed_multiplier(),
            )

    def _toggle_play(self) -> None:
        tab = self._active_analysis_tab()
        if tab is not None and hasattr(tab, "playback_toggle"):
            tab.set_playback_speed(self.controls.speed_multiplier())
            tab.playback_toggle()
            self._sync_analysis_playback_controls()
            return
        AnimationControlMixin._toggle_play(self)

    def _step_fwd(self) -> None:
        tab = self._active_analysis_tab()
        if tab is not None and hasattr(tab, "playback_step_forward"):
            tab.playback_step_forward()
            self._sync_analysis_playback_controls()
            return
        AnimationControlMixin._step_fwd(self)

    def _step_back(self) -> None:
        tab = self._active_analysis_tab()
        if tab is not None and hasattr(tab, "playback_step_back"):
            tab.playback_step_back()
            self._sync_analysis_playback_controls()
            return
        AnimationControlMixin._step_back(self)

    def _rewind(self) -> None:
        tab = self._active_analysis_tab()
        if tab is not None and hasattr(tab, "playback_rewind"):
            tab.playback_rewind()
            self._sync_analysis_playback_controls()
            return
        AnimationControlMixin._rewind(self)

    def _jump_to_end(self) -> None:
        tab = self._active_analysis_tab()
        if tab is not None and hasattr(tab, "playback_jump_to_end"):
            tab.playback_jump_to_end()
            self._sync_analysis_playback_controls()
            return
        AnimationControlMixin._jump_to_end(self)

    def _on_speed(self, speed: float) -> None:
        self.controls.set_speed_multiplier_text(speed)
        tab = self._active_analysis_tab()
        if tab is not None and hasattr(tab, "set_playback_speed"):
            tab.set_playback_speed(speed)
            self._sync_analysis_playback_controls()

    def _undo(self) -> None:
        """Undo the last slider change via the undo stack."""
        if self._undo_stack.undo():
            logger.debug("Undo triggered")
        self._update_undo_redo_actions()

    def _redo(self) -> None:
        """Redo the last undone slider change via the undo stack."""
        if self._undo_stack.redo():
            logger.debug("Redo triggered")
        self._update_undo_redo_actions()

    def closeEvent(self, event: Any) -> None:
        self._save_layout()
        self._save_session_state()
        self._stop_anim()
        self._cancel_event.set()
        event.accept()

    def _save_session_state(self) -> None:
        """Persist current results and slider values on close."""
        try:
            save_app_state(collect_results(self), collect_slider_values(self.sidebar))
        except (OSError, TypeError, ValueError):
            logger.warning("Failed to save session state: %s", traceback.format_exc())

    def try_restore_session(self) -> None:
        """Check for a saved session and offer to restore it."""
        try:
            state = load_app_state()
        except InvalidStateFileError as exc:
            logger.warning("Saved session failed schema validation: %s", exc)
            QMessageBox.warning(
                self,
                "Session State Invalid",
                f"Saved session could not be restored:\n{exc}",
            )
            return
        if state is None:
            return
        reply = QMessageBox.question(
            self,
            "Restore Session",
            "Restore previous session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            restore_slider_values(self.sidebar, state.get("slider_values", {}))
            self._undo_stack.clear()
            self._update_undo_redo_actions()
            self.status_label.setText("Previous session restored (slider values).")
        except (OSError, KeyError, TypeError, ValueError):
            logger.warning("Failed to restore session: %s", traceback.format_exc())

    def _save_layout(self) -> None:
        """Persist window geometry and active tab to QSettings."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("activeTab", self.tabs.currentIndex())

    def _restore_layout(self) -> None:
        """Restore window geometry and active tab from QSettings."""
        geom = self._settings.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        tab_idx = self._settings.value("activeTab", 0, type=int)
        if 0 <= tab_idx < self.tabs.count():
            self.tabs.setCurrentIndex(tab_idx)

    def _cancel_optimization(self) -> None:
        with self._opt_lock:
            if not self._opt_running:
                return
        self._cancel_event.set()
        self.status_label.setText("Cancelling...")
        self.sidebar.set_cancelling()

    def _run_current(self) -> None:
        if not self._is_exercise_tab():
            self.status_label.setText("Select a barbell exercise tab to optimize.")
            return
        self._run_exercise(self.tabs.currentIndex())

    def _run_all(self) -> None:
        self._run_exercise(0, then_chain=list(range(1, len(self.EXERCISE_CONFIGS))))

    def _run_exercise(self, idx: int, then_chain: list[int] | None = None) -> None:
        self._stop_anim()
        self._cancel_event.clear()

        with self._opt_lock:
            if self._opt_running:
                logger.warning("Optimization already running")
                return
            self._opt_running = True

        self.sidebar.show_optimizing()
        name = self.EXERCISE_CONFIGS[idx][0]
        self.status_label.setText(f"Optimizing {name}...")

        threading.Thread(
            target=self._opt_worker,
            args=(idx, then_chain),
            daemon=True,
        ).start()

    # _on_done, _on_cancelled, _update_result_summary, _on_err, and _reset are
    # provided by OptimizationMixin (optimization_mixin.py).
    # Animation, file I/O, and comparison methods are provided by the
    # mixin classes: AnimationControlMixin, FileOperationsMixin, and
    # ComparisonMixin.  See animation_control.py, file_operations.py,
    # and comparison_mixin.py.
