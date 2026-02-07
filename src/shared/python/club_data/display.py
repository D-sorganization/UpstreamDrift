"""Club data display widgets for PyQt6.

Provides widgets for displaying club specifications and target trajectories
across all physics engines (Drake, Pinocchio, MuJoCo).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# PyQt6 imports
try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    QtCore = None  # type: ignore[misc, assignment]
    QtGui = None  # type: ignore[misc, assignment]
    QtWidgets = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from .loader import ClubSpecification, ProPlayerData


class ClubDataDisplayWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Widget for displaying and selecting club data as simulation targets."""

    # Signals
    club_selected = QtCore.pyqtSignal(object)  # Emits ClubSpecification
    player_selected = QtCore.pyqtSignal(object)  # Emits ProPlayerData
    target_enabled_changed = QtCore.pyqtSignal(bool)  # Target overlay toggle

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the club data display widget."""
        if not PYQT6_AVAILABLE:
            raise ImportError("PyQt6 is required for ClubDataDisplayWidget")

        super().__init__(parent)

        self._clubs: list[ClubSpecification] = []
        self._players: list[ProPlayerData] = []
        self._current_club: ClubSpecification | None = None
        self._current_player: ProPlayerData | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QtWidgets.QLabel("<b>Club Data & Targets</b>")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Load buttons
        load_layout = QtWidgets.QHBoxLayout()
        self.btn_load_clubs = QtWidgets.QPushButton("Load Club Data")
        self.btn_load_clubs.setToolTip("Load club specifications from Excel file")
        self.btn_load_clubs.clicked.connect(self._on_load_clubs)
        load_layout.addWidget(self.btn_load_clubs)

        self.btn_load_players = QtWidgets.QPushButton("Load Player Data")
        self.btn_load_players.setToolTip("Load professional player data from Excel")
        self.btn_load_players.clicked.connect(self._on_load_players)
        load_layout.addWidget(self.btn_load_players)
        layout.addLayout(load_layout)

        # Club selection group
        club_group = QtWidgets.QGroupBox("Club Selection")
        club_layout = QtWidgets.QVBoxLayout(club_group)

        # Club type filter
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Type:"))
        self.combo_club_type = QtWidgets.QComboBox()
        self.combo_club_type.addItems(
            ["All", "Driver", "Wood", "Hybrid", "Iron", "Wedge", "Putter"]
        )
        self.combo_club_type.currentTextChanged.connect(self._filter_clubs)
        filter_layout.addWidget(self.combo_club_type)
        club_layout.addLayout(filter_layout)

        # Club list
        self.list_clubs = QtWidgets.QListWidget()
        self.list_clubs.setMaximumHeight(120)
        self.list_clubs.currentItemChanged.connect(self._on_club_changed)
        club_layout.addWidget(self.list_clubs)

        layout.addWidget(club_group)

        # Player selection group
        player_group = QtWidgets.QGroupBox("Target Player Data")
        player_layout = QtWidgets.QVBoxLayout(player_group)

        self.list_players = QtWidgets.QListWidget()
        self.list_players.setMaximumHeight(80)
        self.list_players.currentItemChanged.connect(self._on_player_changed)
        player_layout.addWidget(self.list_players)

        layout.addWidget(player_group)

        # Club specifications display
        spec_group = QtWidgets.QGroupBox("Club Specifications")
        spec_layout = QtWidgets.QFormLayout(spec_group)

        self.lbl_length = QtWidgets.QLabel("--")
        self.lbl_head_mass = QtWidgets.QLabel("--")
        self.lbl_loft = QtWidgets.QLabel("--")
        self.lbl_lie_angle = QtWidgets.QLabel("--")
        self.lbl_total_mass = QtWidgets.QLabel("--")
        self.lbl_moi = QtWidgets.QLabel("--")

        spec_layout.addRow("Length:", self.lbl_length)
        spec_layout.addRow("Head Mass:", self.lbl_head_mass)
        spec_layout.addRow("Loft:", self.lbl_loft)
        spec_layout.addRow("Lie Angle:", self.lbl_lie_angle)
        spec_layout.addRow("Total Mass:", self.lbl_total_mass)
        spec_layout.addRow("MOI:", self.lbl_moi)

        layout.addWidget(spec_group)

        # Target metrics display
        metrics_group = QtWidgets.QGroupBox("Target Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_group)

        self.lbl_club_speed = QtWidgets.QLabel("--")
        self.lbl_ball_speed = QtWidgets.QLabel("--")
        self.lbl_launch_angle = QtWidgets.QLabel("--")
        self.lbl_carry_distance = QtWidgets.QLabel("--")
        self.lbl_smash_factor = QtWidgets.QLabel("--")

        metrics_layout.addRow("Club Speed:", self.lbl_club_speed)
        metrics_layout.addRow("Ball Speed:", self.lbl_ball_speed)
        metrics_layout.addRow("Launch Angle:", self.lbl_launch_angle)
        metrics_layout.addRow("Carry Distance:", self.lbl_carry_distance)
        metrics_layout.addRow("Smash Factor:", self.lbl_smash_factor)

        layout.addWidget(metrics_group)

        # Target overlay controls
        overlay_group = QtWidgets.QGroupBox("Target Overlay")
        overlay_layout = QtWidgets.QVBoxLayout(overlay_group)

        self.chk_show_target = QtWidgets.QCheckBox("Show Target Trajectory")
        self.chk_show_target.setToolTip(
            "Display target trajectory from player data in the visualization"
        )
        self.chk_show_target.toggled.connect(self._on_target_enabled)
        overlay_layout.addWidget(self.chk_show_target)

        # Target display options
        options_layout = QtWidgets.QHBoxLayout()
        self.chk_show_path = QtWidgets.QCheckBox("Path")
        self.chk_show_path.setChecked(True)
        self.chk_show_path.toggled.connect(self._on_overlay_options_changed)
        options_layout.addWidget(self.chk_show_path)

        self.chk_show_velocity = QtWidgets.QCheckBox("Velocity")
        self.chk_show_velocity.toggled.connect(self._on_overlay_options_changed)
        options_layout.addWidget(self.chk_show_velocity)

        self.chk_show_markers = QtWidgets.QCheckBox("Phase Markers")
        self.chk_show_markers.setChecked(True)
        self.chk_show_markers.toggled.connect(self._on_overlay_options_changed)
        options_layout.addWidget(self.chk_show_markers)

        overlay_layout.addLayout(options_layout)

        # Opacity slider
        opacity_layout = QtWidgets.QHBoxLayout()
        opacity_layout.addWidget(QtWidgets.QLabel("Opacity:"))
        self.slider_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_opacity.setMinimum(10)
        self.slider_opacity.setMaximum(100)
        self.slider_opacity.setValue(70)
        self.slider_opacity.valueChanged.connect(self._on_overlay_options_changed)
        opacity_layout.addWidget(self.slider_opacity)
        self.lbl_opacity = QtWidgets.QLabel("70%")
        opacity_layout.addWidget(self.lbl_opacity)
        overlay_layout.addLayout(opacity_layout)

        layout.addWidget(overlay_group)

        layout.addStretch(1)

    def load_clubs(self, clubs: list[ClubSpecification]) -> None:
        """Load club specifications into the widget."""
        self._clubs = clubs
        self._filter_clubs()
        logger.info("Loaded %d clubs into display widget", len(clubs))

    def load_players(self, players: list[ProPlayerData]) -> None:
        """Load player data into the widget."""
        self._players = players
        self.list_players.clear()
        for player in players:
            item = QtWidgets.QListWidgetItem(player.player_name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, player)
            self.list_players.addItem(item)
        logger.info("Loaded %d players into display widget", len(players))

    def get_current_club(self) -> ClubSpecification | None:
        """Get the currently selected club."""
        return self._current_club

    def get_current_player(self) -> ProPlayerData | None:
        """Get the currently selected player."""
        return self._current_player

    def is_target_enabled(self) -> bool:
        """Check if target overlay is enabled."""
        return self.chk_show_target.isChecked()

    def get_overlay_options(self) -> dict[str, Any]:
        """Get current overlay display options."""
        return {
            "show_path": self.chk_show_path.isChecked(),
            "show_velocity": self.chk_show_velocity.isChecked(),
            "show_markers": self.chk_show_markers.isChecked(),
            "opacity": self.slider_opacity.value() / 100.0,
        }

    def _filter_clubs(self, type_filter: str | None = None) -> None:
        """Filter displayed clubs by type."""
        if type_filter is None:
            type_filter = self.combo_club_type.currentText()

        self.list_clubs.clear()
        for club in self._clubs:
            if type_filter == "All" or club.club_type == type_filter:
                item = QtWidgets.QListWidgetItem(club.name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, club)
                self.list_clubs.addItem(item)

    def _on_load_clubs(self) -> None:
        """Handle load clubs button click."""
        from .loader import load_club_data

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Club Data",
            "",
            "Excel Files (*.xlsx *.xls);;JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            try:
                clubs = load_club_data(file_path)
                self.load_clubs(clubs)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Load Error", f"Failed to load club data: {e}"
                )

    def _on_load_players(self) -> None:
        """Handle load players button click."""
        from .loader import ClubDataLoader

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Player Data",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)",
        )
        if file_path:
            try:
                loader = ClubDataLoader()
                players = loader.load_player_data_from_excel(file_path)
                self.load_players(players)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Load Error", f"Failed to load player data: {e}"
                )

    def _on_club_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        """Handle club selection change."""
        if current is None:
            self._current_club = None
            self._update_club_display(None)
            return

        club = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._current_club = club
        self._update_club_display(club)
        self.club_selected.emit(club)

    def _on_player_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        """Handle player selection change."""
        if current is None:
            self._current_player = None
            self._update_metrics_display(None)
            return

        player = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._current_player = player
        self._update_metrics_display(player)
        self.player_selected.emit(player)

    def _update_club_display(self, club: ClubSpecification | None) -> None:
        """Update club specifications display."""
        if club is None:
            self.lbl_length.setText("--")
            self.lbl_head_mass.setText("--")
            self.lbl_loft.setText("--")
            self.lbl_lie_angle.setText("--")
            self.lbl_total_mass.setText("--")
            self.lbl_moi.setText("--")
            return

        self.lbl_length.setText(
            f"{club.length_inches:.1f} in ({club.length_meters:.3f} m)"
        )
        self.lbl_head_mass.setText(
            f"{club.head_mass_grams:.0f} g ({club.head_mass_kg:.3f} kg)"
        )
        self.lbl_loft.setText(f"{club.loft_degrees:.1f}\u00b0")
        self.lbl_lie_angle.setText(f"{club.lie_angle_degrees:.1f}\u00b0")
        self.lbl_total_mass.setText(
            f"{club.total_mass_grams:.0f} g ({club.total_mass_kg:.3f} kg)"
        )
        self.lbl_moi.setText(f"{club.moment_of_inertia:.0f} g\u00b7cm\u00b2")

    def _update_metrics_display(self, player: ProPlayerData | None) -> None:
        """Update target metrics display."""
        if player is None:
            self.lbl_club_speed.setText("--")
            self.lbl_ball_speed.setText("--")
            self.lbl_launch_angle.setText("--")
            self.lbl_carry_distance.setText("--")
            self.lbl_smash_factor.setText("--")
            return

        m = player.metrics
        self.lbl_club_speed.setText(
            f"{m.club_head_speed_mph:.1f} mph ({m.club_head_speed_ms:.1f} m/s)"
        )
        self.lbl_ball_speed.setText(
            f"{m.ball_speed_mph:.1f} mph ({m.ball_speed_ms:.1f} m/s)"
        )
        self.lbl_launch_angle.setText(f"{m.launch_angle_degrees:.1f}\u00b0")
        self.lbl_carry_distance.setText(
            f"{m.carry_distance_yards:.0f} yds ({m.carry_distance_meters:.0f} m)"
        )
        self.lbl_smash_factor.setText(f"{m.smash_factor:.3f}")

    def _on_target_enabled(self, enabled: bool) -> None:
        """Handle target overlay toggle."""
        self.target_enabled_changed.emit(enabled)

    def _on_overlay_options_changed(self) -> None:
        """Handle overlay options change."""
        self.lbl_opacity.setText(f"{self.slider_opacity.value()}%")


class ClubTargetOverlay(ABC):
    """Manages target trajectory overlay for visualization.

    This is a base class that can be extended for different visualization backends
    (MuJoCo custom rendering, Meshcat for Drake/Pinocchio, etc.).
    """

    def __init__(self) -> None:
        """Initialize the overlay manager."""
        self._enabled = False
        self._player: ProPlayerData | None = None
        self._options: dict[str, Any] = {
            "show_path": True,
            "show_velocity": False,
            "show_markers": True,
            "opacity": 0.7,
        }
        self._path_color = (0.2, 0.8, 0.2)  # Green
        self._marker_colors = {
            "address": (0.2, 0.2, 0.8),  # Blue
            "top": (0.8, 0.8, 0.2),  # Yellow
            "impact": (0.8, 0.2, 0.2),  # Red
            "finish": (0.8, 0.2, 0.8),  # Magenta
        }

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the overlay."""
        self._enabled = enabled

    def is_enabled(self) -> bool:
        """Check if overlay is enabled."""
        return self._enabled

    def set_player_data(self, player: ProPlayerData | None) -> None:
        """Set the player data to display as target."""
        self._player = player

    def set_options(self, options: dict[str, Any]) -> None:
        """Update overlay display options."""
        self._options.update(options)

    def get_trajectory_points(self, num_points: int = 100) -> np.ndarray | None:
        """Get interpolated trajectory points for visualization.

        Returns:
            (N, 3) array of positions or None if no trajectory data
        """
        if self._player is None or not self._player.has_trajectory_data():
            return None

        ts = self._player.time_series
        if ts is None or self._player.club_head_positions is None:
            return None

        # Interpolate to requested number of points
        t_interp = np.linspace(ts[0], ts[-1], num_points)
        points = np.zeros((num_points, 3))

        for i, t in enumerate(t_interp):
            pos = self._player.get_position_at_time(t)
            if pos is not None:
                points[i] = pos

        return points

    def get_phase_markers(self) -> dict[str, np.ndarray]:
        """Get positions of swing phase markers.

        Returns:
            Dictionary mapping phase name to position
        """
        if self._player is None or not self._player.has_trajectory_data():
            return {}

        markers = {}
        phases = [
            ("address", self._player.address_time),
            ("top", self._player.top_of_backswing_time),
            ("impact", self._player.impact_time),
            ("finish", self._player.finish_time),
        ]

        for phase_name, time in phases:
            if time > 0:
                pos = self._player.get_position_at_time(time)
                if pos is not None:
                    markers[phase_name] = pos

        return markers

    @abstractmethod
    def render(self, renderer: Any) -> None:
        """Render the overlay.

        This method must be overridden by subclasses for specific renderers.

        Args:
            renderer: The visualization renderer (MuJoCo, Meshcat, etc.)
        """
        if not self._enabled or self._player is None:
            return
