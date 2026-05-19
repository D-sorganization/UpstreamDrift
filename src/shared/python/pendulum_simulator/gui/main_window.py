# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Main application window for the Double Pendulum Golf Swing Simulator.

Orchestrates sub-widgets, manages simulation lifecycle, drives animation.

New in UI/UX upgrade:
- QSettings persistence for window geometry + splitters
- Gravity toggle wired to g=0/9.81 in params builders
- Menu bar: View → Themes (fleet ThemeManager) + quick-switch submenu
- Current dark style preserved as "Pendulum Dark" fallback
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QByteArray, QSettings, Qt
from PyQt6.QtGui import QAction, QShortcut
from PyQt6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .analysis_tab import AnalysisTab
from .controls_utils import PENDULUM_DARK_STYLE as _PENDULUM_DARK_STYLE
from .panel_builders import (
    build_double_panel,
    build_golfer_panel,
    build_triple_panel,
    wire_toolstrip,
)
from .simulation_panel import SimulationPanel
from .toolstrip_widget import ToolStrip
from .unit_converter import UnitConverter

logger = logging.getLogger(__name__)

_SETTINGS_ORG = "D-sorganization"
_SETTINGS_APP = "PendulumSimulator"

# ── Try to import fleet ThemeManager ─────────────────────────────────────────
_THEME_AVAILABLE = False
ThemeManager: Any = None
ThemeManagerDialog: Any = None
create_theme_menu: Any = None


def _find_sibling_package(marker_path: str) -> Path | None:
    """Walk up from this file to find a sibling package directory.

    Searches up to 10 parent levels for the given relative path.
    Returns the parent directory containing the marker, or None.

    Design by Contract
    ------------------
    Pre:  marker_path is a non-empty relative path string.
    Post: returns a valid directory Path or None.
    """
    if not (marker_path):
        raise ValueError("marker_path must be non-empty")
    p = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = p / marker_path
        if candidate.exists():
            return p
        p = p.parent
    return None


try:
    _src_root = _find_sibling_package("shared/python")
    if _src_root is not None:
        _shared_root = _src_root / "shared" / "python"
        if str(_shared_root) not in sys.path:
            sys.path.insert(0, str(_shared_root))
        from theme import ThemeManager as _ThemeManager
        from theme import ThemeManagerDialog as _ThemeManagerDialog
        from theme import create_theme_menu as _create_theme_menu

        ThemeManager = _ThemeManager
        ThemeManagerDialog = _ThemeManagerDialog
        create_theme_menu = _create_theme_menu
        _THEME_AVAILABLE = True
        logger.info("ThemeManager loaded successfully")
    else:
        logger.info("ThemeManager not available; using default theme")
except ImportError:
    logger.info("ThemeManager not available; using default theme")

# ── Try to import shared PlotThemeManager ──────────────────────────────────
_PLOT_THEME_AVAILABLE = False
create_plot_theme_menu: Any = None
try:
    from plot_theme.integration import (
        create_plot_theme_menu as _shared_create_plot_theme_menu,
    )

    create_plot_theme_menu = _shared_create_plot_theme_menu
    _PLOT_THEME_AVAILABLE = True
except ImportError:
    pass


class MainWindow(QMainWindow):
    """Top-level window for the double pendulum simulator."""

    WINDOW_TITLE = "Pendulums"

    # Font zoom bounds (#1147)
    _FONT_MIN_PT = 8
    _FONT_MAX_PT = 24
    _panels: tuple[SimulationPanel, ...]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1400, 800)
        self.setMinimumSize(900, 550)

        # Apply base dark style (always)
        self.setStyleSheet(_PENDULUM_DARK_STYLE)

        # Set app favicon
        _icon_path = Path(__file__).parent / "pendulum_icon.png"
        if _icon_path.exists():
            from PyQt6.QtGui import QIcon

            self.setWindowIcon(QIcon(str(_icon_path)))

        self._theme_manager: object | None = None

        # Unit conversion state (#1137, #1124)
        self._unit_converter = UnitConverter()

        # Ctrl+mousewheel font zoom (#1147)
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._font_zoom_pt: int = int(settings.value("font_zoom_pt", 0))
        if self._font_zoom_pt:
            self._apply_font_zoom()

        self._build_menu()
        self._build_ui()
        self._setup_theme()
        self._restore_geometry()

    def wheelEvent(self, event: object) -> None:
        """Ctrl+mousewheel scales all UI fonts (#1147)."""
        if not (event is not None):
            raise ValueError("event must be provided")
        from PyQt6.QtGui import QWheelEvent

        if not isinstance(event, QWheelEvent):
            return
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._font_zoom_pt = min(self._FONT_MAX_PT, self._font_zoom_pt + 1)
            elif delta < 0:
                self._font_zoom_pt = max(self._FONT_MIN_PT - 10, self._font_zoom_pt - 1)
            self._apply_font_zoom()
            event.accept()
            return
        super().wheelEvent(event)

    def _apply_font_zoom(self) -> None:
        """Apply font zoom offset to the application font."""
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if not isinstance(app, QApplication):
            return
        font = app.font()
        base_pt = 10  # default base
        new_pt = max(
            self._FONT_MIN_PT, min(self._FONT_MAX_PT, base_pt + self._font_zoom_pt)
        )
        font.setPointSize(new_pt)
        app.setFont(font)
        # Persist
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("font_zoom_pt", self._font_zoom_pt)
        logger.info("Font zoom: %d pt (offset %+d)", new_pt, self._font_zoom_pt)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        _mb = self.menuBar()
        if not (_mb is not None):
            raise ValueError("DbC Blocked: Precondition failed.")
        menubar: QMenuBar = _mb

        # View menu
        _view = menubar.addMenu("&View")
        if not (_view is not None):
            raise ValueError("DbC Blocked: Precondition failed.")
        view_menu: QMenu = _view

        # Quick theme submenu
        self._quick_theme_menu = view_menu.addMenu("🎨 Quick Theme")

        # Full theme manager action
        self._action_theme_mgr = QAction("Theme Manager…", self)
        self._action_theme_mgr.setShortcut("Ctrl+Shift+T")
        self._action_theme_mgr.triggered.connect(self._open_theme_manager)
        view_menu.addAction(self._action_theme_mgr)

        view_menu.addSeparator()

        # Analysis dock toggle
        self._action_analysis = QAction("📊 Analysis Panel", self)
        self._action_analysis.setCheckable(True)
        self._action_analysis.setChecked(False)
        self._action_analysis.setShortcut("Ctrl+Shift+A")
        self._action_analysis.triggered.connect(self._toggle_analysis_dock)
        view_menu.addAction(self._action_analysis)

        view_menu.addSeparator()

        # Always-available "Pendulum Dark" built-in
        action_pend_dark = QAction("Pendulum Dark (default)", self)
        action_pend_dark.triggered.connect(self._apply_pendulum_dark)
        view_menu.addAction(action_pend_dark)

        # Plot Theme submenu (for pyqtgraph / matplotlib colours)
        if _PLOT_THEME_AVAILABLE and create_plot_theme_menu is not None:
            view_menu.addSeparator()
            create_plot_theme_menu(self, menubar)

        # Help menu
        _help = menubar.addMenu("&Help")
        if not (_help is not None):
            raise ValueError("DbC Blocked: Precondition failed.")
        action_about = QAction("About…", self)
        action_about.triggered.connect(self._show_about)
        _help.addAction(action_about)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Persistent toolstrip — always visible, regardless of scroll
        self._toolstrip = ToolStrip()
        main_layout.addWidget(self._toolstrip)

        from src.shared.python.screw_theory.ui import ScrewVisualizationTab

        self._tabs = QTabWidget()
        self._double_panel = build_double_panel(self)
        self._triple_panel = build_triple_panel(self)
        self._golfer_panel = build_golfer_panel(self)
        self._screw_panel = ScrewVisualizationTab()

        self._tabs.addTab(self._double_panel, "⚙ Double Pendulum")
        self._tabs.addTab(self._triple_panel, "⚙ Triple Pendulum")
        self._tabs.addTab(self._golfer_panel, "⚙ Golfer Upper Body")
        self._tabs.addTab(self._screw_panel, "⚙ Screw Kinematics")
        # Hide tab bar — model selection is via toolstrip dropdown (#1149)
        tab_bar = self._tabs.tabBar()
        if tab_bar is not None:
            tab_bar.setVisible(False)
        main_layout.addWidget(self._tabs, stretch=1)

        # ── Analysis dock (docked bottom by default) ──────────────────
        self._analysis_tab = AnalysisTab(self)
        self._analysis_dock = QDockWidget("📊 Analysis", self)
        self._analysis_dock.setWidget(self._analysis_tab.widget())
        self._analysis_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._analysis_dock)
        self._analysis_dock.setVisible(False)  # start hidden
        self._analysis_dock.visibilityChanged.connect(
            lambda vis: self._action_analysis.setChecked(vis)
        )

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Ready  ·  Scroll=zoom  ·  Drag=pan  ·  Dbl-click=reset view",
        )

        wire_toolstrip(self)
        self._setup_keyboard_shortcuts()
        self._wire_analysis_tab()

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up global keyboard shortcuts for simulation control."""
        from PyQt6.QtGui import QKeySequence

        # Space = play/pause toggle
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._on_shortcut_play_pause)
        # R = reset simulation
        QShortcut(QKeySequence(Qt.Key.Key_R), self, self._on_shortcut_reset)
        # Ctrl+E = export CSV
        QShortcut(QKeySequence("Ctrl+E"), self, self._on_shortcut_export_data)
        # F5 = run simulation
        QShortcut(QKeySequence(Qt.Key.Key_F5), self, self._on_shortcut_run)
        # Escape = stop animation
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._on_shortcut_stop)

    def _on_shortcut_play_pause(self) -> None:
        """Space key: toggle play/pause."""
        panel = self._active_panel()
        current_state = panel.controls.btn_play.isChecked()
        panel.controls.btn_play.setChecked(not current_state)

    def _on_shortcut_reset(self) -> None:
        """R key: reset simulation."""
        self._active_panel().controls.reset_requested.emit()

    def _on_shortcut_export_data(self) -> None:
        """Ctrl+E: export CSV data."""
        self._active_panel().controls.export_data_requested.emit()

    def _on_shortcut_run(self) -> None:
        """F5: run simulation."""
        self._active_panel().controls.run_requested.emit()

    def _on_shortcut_stop(self) -> None:
        """Escape: stop animation."""
        self._active_panel().controls.stop_playback()

    def _wire_analysis_tab(self) -> None:
        """Connect each panel's sim_finished signal to push results to analysis.

        Design by Contract
        ------------------
        Pre:  self._panels and self._analysis_tab exist.
        Post: Any simulation completion updates the analysis tab.
        """
        model_map = {0: "double", 1: "triple", 2: "golfer"}

        for idx, panel in enumerate(self._panels):
            model_type = model_map[idx]

            def _on_finished(
                _p: SimulationPanel = panel, _mt: str = model_type
            ) -> None:
                result = _p._result
                if result is not None:
                    self._analysis_tab.set_result(result, model_type=_mt)
                    logger.info("Analysis tab updated with %s result", _mt)

            panel.sim_finished.connect(_on_finished)

    def _active_panel(self) -> SimulationPanel:
        """Return the SimulationPanel for the currently visible tab."""
        idx = self._tabs.currentIndex()
        return self._panels[idx]

    # ------------------------------------------------------------------
    # Per-segment visibility (#1100, #1101, #1102)
    # ------------------------------------------------------------------

    # Joint (key, display_label) per model type.
    # key = internal physics key, label = human-readable for toolstrip.
    _SEGMENTS_DOUBLE: list[tuple[str, str]] = [
        ("shoulder", "Shoulder"),
        ("wrist", "Wrist"),
        ("tip", "Tip"),
    ]
    _SEGMENTS_TRIPLE: list[tuple[str, str]] = [
        ("shoulder", "Shoulder"),
        ("wrist1", "Wrist 1"),
        ("wrist2", "Wrist 2"),
        ("tip", "Tip"),
    ]
    _SEGMENTS_GOLFER: list[tuple[str, str]] = [
        ("hub", "Hub"),
        ("rs", "Right Shoulder"),
        ("re", "Right Elbow"),
        ("rh", "Right Hand"),
        ("ls", "Left Shoulder"),
        ("le", "Left Elbow"),
        ("lh", "Left Hand"),
        ("club_tip", "Club Tip"),
    ]

    def _on_tab_changed(self, index: int) -> None:  # noqa: C901
        """Update toolstrip segment checkboxes and sync overlay state for the active tab.

        When the user switches tabs the new panel's pendulum widget must
        receive the current overlay toggle states from the toolstrip so
        that forces, ellipsoids, COM, etc. match the checkbox display.
        """
        if not (index is not None):
            raise ValueError("index must be provided")
        segment_map = {
            0: self._SEGMENTS_DOUBLE,
            1: self._SEGMENTS_TRIPLE,
            2: self._SEGMENTS_GOLFER,
        }
        names = segment_map.get(index, self._SEGMENTS_DOUBLE)
        self._toolstrip.set_segment_names(names)

        # ── Sync overlay toggle states to the newly-active panel ──────
        pw = self._active_panel().pendulum
        ts = self._toolstrip
        if hasattr(pw, "set_show_forces"):
            pw.set_show_forces(ts.chk_forces.isChecked())
        if hasattr(pw, "set_show_zero_torque_forces"):
            pw.set_show_zero_torque_forces(ts.chk_zero_torque.isChecked())
        if hasattr(pw, "set_show_mob_ellipsoids"):
            pw.set_show_mob_ellipsoids(ts.chk_mob.isChecked())
        if hasattr(pw, "set_show_force_ellipsoids"):
            pw.set_show_force_ellipsoids(ts.chk_force_ell.isChecked())
        if hasattr(pw, "set_show_com"):
            pw.set_show_com(ts.chk_com.isChecked())

        # Sync scale slider values
        if hasattr(pw, "set_force_scale"):
            pw.set_force_scale(ts._sld_force.value() / 10.0)
        if hasattr(pw, "set_mob_ellipsoid_scale"):
            pw.set_mob_ellipsoid_scale(ts._sld_mob.value() / 10.0)
        if hasattr(pw, "set_force_ellipsoid_scale"):
            pw.set_force_ellipsoid_scale(ts._sld_force_ell.value() / 10.0)

        # Sync segment visibility from current checkbox state
        if hasattr(pw, "set_visible_segments"):
            ts._on_segment_toggled()  # re-emits segment_visibility_changed

    # ------------------------------------------------------------------
    # Pop-out chart (#1135)
    # ------------------------------------------------------------------

    def _on_popout_chart(self) -> None:
        """Open a pop-out chart with user-selected data variables.

        Design by Contract
        ------------------
        Pre: A simulation must have been run (result is not None).
        Post: A detachable chart window opens with the selected data.
        """
        from ..data_extractor import extract_series
        from .chart_data_dialog import ChartDataDialog
        from .popout_chart import PopOutChart

        panel = self._active_panel()
        result = panel._result
        if result is None:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Pop-Out Chart",
                "Run a simulation first to generate data for charting.",
            )
            return

        # Determine model type from active panel
        model_type = "double"
        if hasattr(panel, "_triple"):
            model_type = "triple"
        elif hasattr(panel, "_golfer"):
            model_type = "golfer"

        dlg = ChartDataDialog(self, model_type=model_type)
        if not dlg.exec():
            return

        x_key, y_key, reg_degree = dlg.get_selection()

        try:
            x_vals, x_desc, x_unit = extract_series(result, x_key, model_type)
            y_vals, y_desc, y_unit = extract_series(result, y_key, model_type)
        except (KeyError, AttributeError) as exc:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Data Error",
                f"Could not extract data: {exc}",
            )
            return

        chart = PopOutChart(self)
        chart.plot_data(
            x_vals,
            y_vals,
            f"{x_desc} ({x_unit})",
            f"{y_desc} ({y_unit})",
            f"{y_desc} vs {x_desc}",
        )
        if reg_degree > 0:
            chart.add_regression(degree=reg_degree)
        chart.show()

        # Keep reference to prevent garbage collection
        if not hasattr(self, "_popout_charts"):
            self._popout_charts: list = []
        self._popout_charts.append(chart)
        logger.info("Pop-out chart opened: %s vs %s", y_key, x_key)

    # ------------------------------------------------------------------
    # Theme management
    # ------------------------------------------------------------------

    def _setup_theme(self) -> None:
        """Wire fleet ThemeManager if available; populate quick-theme menu."""
        if not _THEME_AVAILABLE or ThemeManager is None:
            logger.info("theme package unavailable — using Pendulum Dark built-in")
            return

        try:
            self._theme_manager = ThemeManager.instance(
                main_window=self,
                app_context="PendulumSimulator",
                settings_org=_SETTINGS_ORG,
                settings_app=_SETTINGS_APP,
            )
            # Apply saved theme
            self._theme_manager.apply_theme()  # type: ignore[union-attr]
            self._theme_manager.themeChanged.connect(self._on_theme_changed)  # type: ignore[union-attr]

            # Use shared helper to build a full theme submenu (window first, then parent)
            if not (self._quick_theme_menu is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            if create_theme_menu is not None:
                create_theme_menu(
                    self,
                    self._quick_theme_menu,
                    show_custom_options=True,
                )

        except (ImportError, AttributeError, RuntimeError):
            logger.exception("Failed to initialise ThemeManager")
            self._theme_manager = None

    def _on_theme_changed(self, name: str) -> None:
        self.status.showMessage(f"Theme changed to: {name}", 3000)

    def _open_theme_manager(self) -> None:
        if (
            not _THEME_AVAILABLE
            or self._theme_manager is None
            or ThemeManagerDialog is None
        ):
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Themes",
                "The fleet theme package is not installed.\n\n"
                "Use View → Pendulum Dark to reset to the default style.",
            )
            return
        dlg = ThemeManagerDialog(self._theme_manager, self)
        dlg.exec()

    def _toggle_analysis_dock(self, checked: bool) -> None:
        """Show or hide the docked analysis panel."""
        self._analysis_dock.setVisible(checked)

    def _apply_pendulum_dark(self) -> None:
        """Force-reset to the built-in pendulum dark stylesheet."""
        self.setStyleSheet(_PENDULUM_DARK_STYLE)
        self.status.showMessage("Theme: Pendulum Dark", 3000)

    def _show_about(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About",
            "<b>Double Pendulum Golf Swing Simulator</b><br><br>"
            "Interactive simulation of 2-, 3-segment, and golfer"
            " upper-body pendulum dynamics.<br><br>"
            "Built with PyQt6 · NumPy · SciPy<br>"
            "D-sorganization Tools Repository",
        )

    # ------------------------------------------------------------------
    # Geometry persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        geom = settings.value("window_geometry")
        if isinstance(geom, QByteArray):
            self.restoreGeometry(geom)

    def closeEvent(self, event: object) -> None:
        if not (event is not None):
            raise ValueError("event must be provided")
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("window_geometry", self.saveGeometry())
        self._double_panel.save_layout()
        self._triple_panel.save_layout()
        self._golfer_panel.save_layout()
        super().closeEvent(event)  # type: ignore[arg-type]


def get_dockable_ui() -> MainWindow:
    """Return the main window instance for docking in the unified launcher."""
    return MainWindow()
