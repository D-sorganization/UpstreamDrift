"""Club Data Tab for GUI Integration.

Provides a complete tab widget for loading, displaying, and managing
club data and player targets across all physics engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class ClubDataTab(QtWidgets.QWidget):  # type: ignore[misc]
    """Tab widget for club data and target management.

    This widget can be added to any physics engine GUI to provide
    club data display and target trajectory visualization.
    """

    # Signals
    club_selected = QtCore.pyqtSignal(object)  # ClubSpecification
    player_selected = QtCore.pyqtSignal(object)  # ProPlayerData
    target_enabled = QtCore.pyqtSignal(bool)
    target_options_changed = QtCore.pyqtSignal(dict)

    def __init__(
        self,
        data_dir: str | Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the club data tab.

        Args:
            data_dir: Default directory for data files
            parent: Parent widget
        """
        if not PYQT6_AVAILABLE:
            raise ImportError("PyQt6 is required for ClubDataTab")

        super().__init__(parent)

        self._data_dir = Path(data_dir) if data_dir else None
        self._clubs: list[Any] = []
        self._players: list[Any] = []
        self._current_club: Any = None
        self._current_player: Any = None
        self._target_manager: Any = None

        self._setup_ui()
        self._auto_load_data()

    def _setup_ui(self) -> None:
        """Create the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QtWidgets.QLabel("<b>Club Data & Target Matching</b>")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # Create splitter for resizable sections
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # -------- Club Data Section --------
        club_group = QtWidgets.QGroupBox("Club Specifications")
        club_layout = QtWidgets.QVBoxLayout(club_group)

        # Load buttons
        load_layout = QtWidgets.QHBoxLayout()
        self.btn_load_clubs = QtWidgets.QPushButton("Load Club Data")
        self.btn_load_clubs.setToolTip("Load club specifications from Excel/JSON")
        self.btn_load_clubs.clicked.connect(self._on_load_clubs)
        load_layout.addWidget(self.btn_load_clubs)

        self.btn_load_default = QtWidgets.QPushButton("Load Default")
        self.btn_load_default.setToolTip("Load default Club_Data.xlsx")
        self.btn_load_default.clicked.connect(self._load_default_club_data)
        load_layout.addWidget(self.btn_load_default)

        club_layout.addLayout(load_layout)

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
        self.list_clubs.setMaximumHeight(100)
        self.list_clubs.currentItemChanged.connect(self._on_club_changed)
        club_layout.addWidget(self.list_clubs)

        # Club details
        details_layout = QtWidgets.QFormLayout()
        self.lbl_club_length = QtWidgets.QLabel("--")
        self.lbl_club_mass = QtWidgets.QLabel("--")
        self.lbl_club_loft = QtWidgets.QLabel("--")
        self.lbl_club_lie = QtWidgets.QLabel("--")
        self.lbl_club_moi = QtWidgets.QLabel("--")

        details_layout.addRow("Length:", self.lbl_club_length)
        details_layout.addRow("Head Mass:", self.lbl_club_mass)
        details_layout.addRow("Loft:", self.lbl_club_loft)
        details_layout.addRow("Lie Angle:", self.lbl_club_lie)
        details_layout.addRow("MOI:", self.lbl_club_moi)

        club_layout.addLayout(details_layout)
        splitter.addWidget(club_group)

        # -------- Player Target Section --------
        player_group = QtWidgets.QGroupBox("Professional Player Targets")
        player_layout = QtWidgets.QVBoxLayout(player_group)

        # Load player data
        load_player_layout = QtWidgets.QHBoxLayout()
        self.btn_load_players = QtWidgets.QPushButton("Load Player Data")
        self.btn_load_players.clicked.connect(self._on_load_players)
        load_player_layout.addWidget(self.btn_load_players)

        self.btn_load_trajectory = QtWidgets.QPushButton("Load Trajectory")
        self.btn_load_trajectory.setToolTip("Load time-series trajectory data")
        self.btn_load_trajectory.clicked.connect(self._on_load_trajectory)
        load_player_layout.addWidget(self.btn_load_trajectory)

        player_layout.addLayout(load_player_layout)

        # Player list
        self.list_players = QtWidgets.QListWidget()
        self.list_players.setMaximumHeight(80)
        self.list_players.currentItemChanged.connect(self._on_player_changed)
        player_layout.addWidget(self.list_players)

        # Target metrics
        metrics_group = QtWidgets.QGroupBox("Target Swing Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_group)

        self.lbl_club_speed = QtWidgets.QLabel("--")
        self.lbl_ball_speed = QtWidgets.QLabel("--")
        self.lbl_launch_angle = QtWidgets.QLabel("--")
        self.lbl_spin_rate = QtWidgets.QLabel("--")
        self.lbl_carry = QtWidgets.QLabel("--")
        self.lbl_smash = QtWidgets.QLabel("--")

        metrics_layout.addRow("Club Speed:", self.lbl_club_speed)
        metrics_layout.addRow("Ball Speed:", self.lbl_ball_speed)
        metrics_layout.addRow("Launch Angle:", self.lbl_launch_angle)
        metrics_layout.addRow("Spin Rate:", self.lbl_spin_rate)
        metrics_layout.addRow("Carry:", self.lbl_carry)
        metrics_layout.addRow("Smash Factor:", self.lbl_smash)

        player_layout.addWidget(metrics_group)
        splitter.addWidget(player_group)

        # -------- Target Overlay Section --------
        overlay_group = QtWidgets.QGroupBox("Target Visualization")
        overlay_layout = QtWidgets.QVBoxLayout(overlay_group)

        # Enable toggle
        self.chk_show_target = QtWidgets.QCheckBox(
            "Show Target Trajectory in Visualization"
        )
        self.chk_show_target.setToolTip(
            "Display the target player's swing path in the 3D view"
        )
        self.chk_show_target.toggled.connect(self._on_target_toggled)
        overlay_layout.addWidget(self.chk_show_target)

        # Display options
        options_layout = QtWidgets.QHBoxLayout()

        self.chk_show_path = QtWidgets.QCheckBox("Path")
        self.chk_show_path.setChecked(True)
        self.chk_show_path.toggled.connect(self._emit_options)
        options_layout.addWidget(self.chk_show_path)

        self.chk_show_velocity = QtWidgets.QCheckBox("Velocity")
        self.chk_show_velocity.toggled.connect(self._emit_options)
        options_layout.addWidget(self.chk_show_velocity)

        self.chk_show_markers = QtWidgets.QCheckBox("Phase Markers")
        self.chk_show_markers.setChecked(True)
        self.chk_show_markers.toggled.connect(self._emit_options)
        options_layout.addWidget(self.chk_show_markers)

        overlay_layout.addLayout(options_layout)

        # Opacity
        opacity_layout = QtWidgets.QHBoxLayout()
        opacity_layout.addWidget(QtWidgets.QLabel("Opacity:"))

        self.slider_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_opacity.setMinimum(10)
        self.slider_opacity.setMaximum(100)
        self.slider_opacity.setValue(70)
        self.slider_opacity.valueChanged.connect(self._emit_options)
        opacity_layout.addWidget(self.slider_opacity)

        self.lbl_opacity = QtWidgets.QLabel("70%")
        opacity_layout.addWidget(self.lbl_opacity)

        overlay_layout.addLayout(opacity_layout)

        # Color selection
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(QtWidgets.QLabel("Path Color:"))

        self.btn_color = QtWidgets.QPushButton()
        self.btn_color.setStyleSheet("background-color: #4CAF50;")
        self.btn_color.setMaximumWidth(40)
        self.btn_color.clicked.connect(self._choose_color)
        self._path_color = (0.3, 0.69, 0.31)  # Green
        color_layout.addWidget(self.btn_color)
        color_layout.addStretch()

        overlay_layout.addLayout(color_layout)

        splitter.addWidget(overlay_group)

        # -------- Tracking Section --------
        tracking_group = QtWidgets.QGroupBox("Real-time Tracking Error")
        tracking_layout = QtWidgets.QFormLayout(tracking_group)

        self.lbl_position_error = QtWidgets.QLabel("-- m")
        self.lbl_velocity_error = QtWidgets.QLabel("-- m/s")
        self.lbl_phase = QtWidgets.QLabel("--")

        tracking_layout.addRow("Position Error:", self.lbl_position_error)
        tracking_layout.addRow("Velocity Error:", self.lbl_velocity_error)
        tracking_layout.addRow("Current Phase:", self.lbl_phase)

        splitter.addWidget(tracking_group)

        # Set initial sizes
        splitter.setSizes([200, 250, 150, 100])

    def _auto_load_data(self) -> None:
        """Automatically load data from default location if available."""
        if self._data_dir is None:
            # Try to find project root
            try:
                current = Path(__file__).resolve()
                project_root = current.parent.parent.parent.parent.parent
                self._data_dir = project_root / "data"
            except (FileNotFoundError, OSError):
                return

        # Try to load default club data
        if self._data_dir and self._data_dir.exists():
            club_file = self._data_dir / "Club_Data.xlsx"
            if club_file.exists():
                self._load_club_file(str(club_file))

    def _on_load_clubs(self) -> None:
        """Handle load clubs button click."""
        start_dir = str(self._data_dir) if self._data_dir else ""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Club Data",
            start_dir,
            "Excel Files (*.xlsx *.xls);;JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            self._load_club_file(file_path)

    def _load_default_club_data(self) -> None:
        """Load the default Club_Data.xlsx file."""
        if self._data_dir is None:
            return

        club_file = self._data_dir / "Club_Data.xlsx"
        if club_file.exists():
            self._load_club_file(str(club_file))
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"Default club data file not found at:\n{club_file}",
            )

    def _load_club_file(self, file_path: str) -> None:
        """Load clubs from a file."""
        try:
            from .loader import ClubDataLoader

            loader = ClubDataLoader()
            self._clubs = loader.load_clubs_from_excel(file_path)
            self._filter_clubs()
            logger.info("Loaded %d clubs from %s", len(self._clubs), file_path)

        except ImportError as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Dependencies",
                f"Required libraries not installed:\n{e}\n\n"
                "Install with: pip install pandas openpyxl",
            )
        except (RuntimeError, TypeError, AttributeError) as e:
            QtWidgets.QMessageBox.critical(
                self, "Load Error", f"Failed to load club data:\n{e}"
            )

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

    def _on_club_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        """Handle club selection change."""
        if current is None:
            self._current_club = None
            self._clear_club_details()
            return

        club = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._current_club = club
        self._update_club_details(club)
        self.club_selected.emit(club)

    def _update_club_details(self, club: Any) -> None:
        """Update club details display."""
        self.lbl_club_length.setText(
            f"{club.length_inches:.1f} in ({club.length_meters:.3f} m)"
        )
        self.lbl_club_mass.setText(
            f"{club.head_mass_grams:.0f} g ({club.head_mass_kg:.3f} kg)"
        )
        self.lbl_club_loft.setText(f"{club.loft_degrees:.1f}\u00b0")
        self.lbl_club_lie.setText(f"{club.lie_angle_degrees:.1f}\u00b0")
        self.lbl_club_moi.setText(f"{club.moment_of_inertia:.0f} g\u00b7cm\u00b2")

    def _clear_club_details(self) -> None:
        """Clear club details display."""
        self.lbl_club_length.setText("--")
        self.lbl_club_mass.setText("--")
        self.lbl_club_loft.setText("--")
        self.lbl_club_lie.setText("--")
        self.lbl_club_moi.setText("--")

    def _on_load_players(self) -> None:
        """Handle load players button click."""
        start_dir = str(self._data_dir) if self._data_dir else ""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Player Data",
            start_dir,
            "Excel Files (*.xlsx *.xls);;All Files (*)",
        )
        if file_path:
            self._load_player_file(file_path)

    def _on_load_trajectory(self) -> None:
        """Handle load trajectory button click."""
        start_dir = str(self._data_dir) if self._data_dir else ""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Trajectory Data",
            start_dir,
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)",
        )
        if file_path:
            # Ask for player name
            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Player Name",
                "Enter name for this trajectory:",
                text="Pro Player",
            )
            if ok and name:
                self._load_trajectory_file(file_path, name)

    def _load_player_file(self, file_path: str) -> None:
        """Load players from a file."""
        try:
            from .loader import ClubDataLoader

            loader = ClubDataLoader()
            self._players = loader.load_player_data_from_excel(file_path)

            self.list_players.clear()
            for player in self._players:
                item = QtWidgets.QListWidgetItem(player.player_name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, player)
                self.list_players.addItem(item)

            logger.info("Loaded %d players from %s", len(self._players), file_path)

        except ImportError as e:
            QtWidgets.QMessageBox.critical(
                self, "Load Error", f"Failed to load player data:\n{e}"
            )

    def _load_trajectory_file(self, file_path: str, player_name: str) -> None:
        """Load trajectory from a file."""
        try:
            from .loader import ClubDataLoader

            loader = ClubDataLoader()
            player = loader.load_trajectory_from_excel(file_path, player_name)

            self._players.append(player)

            item = QtWidgets.QListWidgetItem(f"{player_name} (trajectory)")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, player)
            self.list_players.addItem(item)

            logger.info("Loaded trajectory for %s from %s", player_name, file_path)

        except ImportError as e:
            QtWidgets.QMessageBox.critical(
                self, "Load Error", f"Failed to load trajectory:\n{e}"
            )

    def _on_player_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        """Handle player selection change."""
        if current is None:
            self._current_player = None
            self._clear_player_metrics()
            return

        player = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._current_player = player
        self._update_player_metrics(player)
        self.player_selected.emit(player)

    def _update_player_metrics(self, player: Any) -> None:
        """Update player metrics display."""
        m = player.metrics
        self.lbl_club_speed.setText(
            f"{m.club_head_speed_mph:.1f} mph ({m.club_head_speed_ms:.1f} m/s)"
        )
        self.lbl_ball_speed.setText(
            f"{m.ball_speed_mph:.1f} mph ({m.ball_speed_ms:.1f} m/s)"
        )
        self.lbl_launch_angle.setText(f"{m.launch_angle_degrees:.1f}\u00b0")
        self.lbl_spin_rate.setText(f"{m.spin_rate_rpm:.0f} rpm")
        self.lbl_carry.setText(
            f"{m.carry_distance_yards:.0f} yds ({m.carry_distance_meters:.0f} m)"
        )
        self.lbl_smash.setText(f"{m.smash_factor:.3f}")

    def _clear_player_metrics(self) -> None:
        """Clear player metrics display."""
        self.lbl_club_speed.setText("--")
        self.lbl_ball_speed.setText("--")
        self.lbl_launch_angle.setText("--")
        self.lbl_spin_rate.setText("--")
        self.lbl_carry.setText("--")
        self.lbl_smash.setText("--")

    def _on_target_toggled(self, enabled: bool) -> None:
        """Handle target overlay toggle."""
        self.target_enabled.emit(enabled)

    def _emit_options(self) -> None:
        """Emit current target options."""
        self.lbl_opacity.setText(f"{self.slider_opacity.value()}%")

        options = {
            "show_path": self.chk_show_path.isChecked(),
            "show_velocity": self.chk_show_velocity.isChecked(),
            "show_markers": self.chk_show_markers.isChecked(),
            "opacity": self.slider_opacity.value() / 100.0,
            "path_color": self._path_color,
        }
        self.target_options_changed.emit(options)

    def _choose_color(self) -> None:
        """Open color picker for path color."""
        current = QtGui.QColor(
            int(self._path_color[0] * 255),
            int(self._path_color[1] * 255),
            int(self._path_color[2] * 255),
        )
        color = QtWidgets.QColorDialog.getColor(current, self, "Choose Path Color")

        if color.isValid():
            self._path_color = (color.redF(), color.greenF(), color.blueF())
            self.btn_color.setStyleSheet(f"background-color: {color.name()};")
            self._emit_options()

    def update_tracking_error(
        self,
        position_error: float,
        velocity_error: float | None = None,
        phase: str | None = None,
    ) -> None:
        """Update the tracking error display.

        Args:
            position_error: Position error in meters
            velocity_error: Velocity error in m/s (optional)
            phase: Current swing phase name (optional)
        """
        self.lbl_position_error.setText(f"{position_error:.4f} m")

        if velocity_error is not None:
            self.lbl_velocity_error.setText(f"{velocity_error:.2f} m/s")
        else:
            self.lbl_velocity_error.setText("-- m/s")

        if phase:
            self.lbl_phase.setText(phase)
        else:
            self.lbl_phase.setText("--")

    def get_current_club(self) -> Any:
        """Get the currently selected club."""
        return self._current_club

    def get_current_player(self) -> Any:
        """Get the currently selected player."""
        return self._current_player

    def is_target_enabled(self) -> bool:
        """Check if target overlay is enabled."""
        return self.chk_show_target.isChecked()

    def get_target_options(self) -> dict[str, Any]:
        """Get current target visualization options."""
        return {
            "show_path": self.chk_show_path.isChecked(),
            "show_velocity": self.chk_show_velocity.isChecked(),
            "show_markers": self.chk_show_markers.isChecked(),
            "opacity": self.slider_opacity.value() / 100.0,
            "path_color": self._path_color,
        }
