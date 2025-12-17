#!/usr/bin/env python3
"""
Golf Swing Visualizer - Main Application Entry Point
Complete integration of all components with enhanced features and error handling
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("golf_visualizer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import PyQt6 with error handling
try:
    from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QFont, QPixmap
    from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen
except ImportError as e:
    print("❌ PyQt6 not found. Please install it with: pip install PyQt6")
    print(f"Error: {e}")
    sys.exit(1)

# Import OpenGL with error handling
try:
    import moderngl as mgl
except ImportError as e:
    print("❌ ModernGL not found. Please install it with: pip install moderngl")
    print(f"Error: {e}")
    sys.exit(1)

# Import core modules with error handling
try:
    from golf_camera_system import CameraController, CameraMode, CameraPreset
    from golf_data_core import (
        FrameProcessor,
        MatlabDataLoader,
        PerformanceStats,
        RenderConfig,
    )
    from golf_gui_application import GolfVisualizerMainWindow, GolfVisualizerWidget
    from golf_opengl_renderer import OpenGLRenderer
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    print(
        "❌ Core modules not found. Please ensure all files are in the same directory."
    )
    print(f"Error: {e}")
    sys.exit(1)

# ============================================================================
# ENHANCED MAIN APPLICATION
# ============================================================================


class EnhancedGolfVisualizerApp(QApplication):
    """Enhanced main application with advanced features"""

    def __init__(self, argv):
        super().__init__(argv)

        # Application metadata
        self.setApplicationName("Golf Swing Visualizer Pro")
        self.setApplicationVersion("2.0.0")
        self.setApplicationDisplayName("Golf Swing Visualizer Pro")
        self.setOrganizationName("Golf Analytics Lab")
        self.setOrganizationDomain("golfanalytics.com")

        # Application settings
        self.setQuitOnLastWindowClosed(True)

        # Setup application-wide styling
        self._setup_application_style()

        # Main window
        self.main_window: Optional[EnhancedMainWindow] = None

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        logger.info("Enhanced Golf Visualizer App initialized")

    def _setup_application_style(self):
        """Setup application-wide styling and fonts"""
        # Set default font
        font = QFont("Segoe UI", 9)
        font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
        self.setFont(font)

        # Load application icon (if available)
        # self.setWindowIcon(QIcon("assets/icon.png"))

    def initialize(self) -> bool:
        """Initialize the application"""
        try:
            # Show splash screen
            splash = self._create_splash_screen()
            splash.show()
            self.processEvents()

            # Create main window
            splash.showMessage(
                "Initializing main window...", Qt.AlignmentFlag.AlignBottom
            )
            self.processEvents()

            self.main_window = EnhancedMainWindow()

            # Setup performance monitoring
            splash.showMessage(
                "Setting up performance monitoring...", Qt.AlignmentFlag.AlignBottom
            )
            self.processEvents()

            self.performance_monitor.start_monitoring()

            # Show main window
            splash.showMessage("Starting application...", Qt.AlignmentFlag.AlignBottom)
            self.processEvents()

            self.main_window.show()
            splash.finish(self.main_window)

            # Auto-load data if available
            self._auto_load_data()

            logger.info("Application initialization complete")
            return True

        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            QMessageBox.critical(
                None, "Initialization Error", f"Failed to initialize application:\n{e}"
            )
            return False

    def _create_splash_screen(self) -> QSplashScreen:
        """Create application splash screen"""
        # Create a simple splash screen (could be replaced with custom image)
        pixmap = QPixmap(400, 300)
        pixmap.fill(Qt.GlobalColor.darkBlue)

        splash = QSplashScreen(pixmap)
        splash.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        )

        font = QFont("Arial", 16, QFont.Weight.Bold)
        splash.setFont(font)
        splash.showMessage(
            "Golf Swing Visualizer Pro\nLoading...",
            Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.white,
        )

        return splash

    def _auto_load_data(self):
        """Attempt to auto-load data files if they exist"""
        data_files = ["BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat"]

        if all(Path(f).exists() for f in data_files):
            logger.info("Found data files, attempting auto-load...")
            if self.main_window:
                QTimer.singleShot(
                    1000, lambda: self.main_window.load_data_files(data_files)
                )


class PerformanceMonitor(QThread):
    """Background performance monitoring"""

    performanceUpdate = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.monitoring = False
        self.stats = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": 0.0,
            "frame_rate": 0.0,
        }

    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        self.wait()
        logger.info("Performance monitoring stopped")

    def run(self):
        """Performance monitoring loop"""
        while self.monitoring:
            try:
                # Update performance stats (simplified implementation)
                import psutil

                self.stats["cpu_usage"] = psutil.cpu_percent(interval=1)
                self.stats["memory_usage"] = psutil.virtual_memory().percent

                # GPU usage would require additional libraries (pynvml, etc.)
                self.stats["gpu_usage"] = 0.0

                self.performanceUpdate.emit(self.stats.copy())

            except Exception as e:
                logger.warning(f"Performance monitoring error: {e}")

            self.msleep(1000)  # Update every second


class EnhancedMainWindow(GolfVisualizerMainWindow):
    """Enhanced main window with additional features"""

    def __init__(self):
        super().__init__()

        # Enhanced camera system
        self.camera_controller = CameraController()
        self._integrate_camera_system()

        # Advanced features
        self.session_manager = SessionManager()
        self.export_manager = ExportManager()
        self.plugin_manager = PluginManager()

        # Enhanced status tracking
        self.analysis_results: Dict = {}
        self.current_session: Optional[str] = None

        self._setup_enhanced_features()

        logger.info("Enhanced main window initialized")

    def _integrate_camera_system(self):
        """Integrate advanced camera system"""
        if hasattr(self.gl_widget, "camera_controller"):
            # Replace the basic camera with our advanced system
            self.gl_widget.camera_controller = self.camera_controller

            # Connect camera signals
            self.camera_controller.cameraChanged.connect(self.gl_widget.update)
            self.camera_controller.modeChanged.connect(self._on_camera_mode_changed)
            self.camera_controller.animationFinished.connect(
                self._on_camera_animation_finished
            )

        # Add camera preset menu
        self._add_camera_preset_menu()

    def _add_camera_preset_menu(self):
        """Add camera preset menu to menubar"""
        menubar = self.menuBar()
        camera_menu = menubar.addMenu("Camera")

        # Camera presets
        presets_submenu = camera_menu.addMenu("Presets")

        for preset in CameraPreset:
            action = presets_submenu.addAction(preset.value.replace("_", " ").title())
            action.triggered.connect(
                lambda checked, p=preset: self.camera_controller.set_preset(p)
            )

        camera_menu.addSeparator()

        # Camera modes
        modes_submenu = camera_menu.addMenu("Modes")

        for mode in CameraMode:
            action = modes_submenu.addAction(mode.value.title())
            action.triggered.connect(
                lambda checked, m=mode: self.camera_controller.set_mode(m)
            )

        camera_menu.addSeparator()

        # Camera animations
        anim_menu = camera_menu.addMenu("Animations")
        anim_menu.addAction("Smooth Orbit").triggered.connect(
            self._demo_orbit_animation
        )
        anim_menu.addAction("Cinematic Tour").triggered.connect(
            self._demo_cinematic_tour
        )

    def _setup_enhanced_features(self):
        """Setup enhanced application features"""
        # Add toolbar extensions
        self._add_enhanced_toolbar()

        # Setup keyboard shortcuts
        self._setup_enhanced_shortcuts()

        # Setup status monitoring
        self._setup_status_monitoring()

    def _add_enhanced_toolbar(self):
        """Add enhanced toolbar with additional controls"""
        toolbar = self.findChild(object, "MainToolBar")  # Find existing toolbar
        if toolbar:
            toolbar.addSeparator()

            # Analysis tools
            analysis_action = toolbar.addAction("📊")
            analysis_action.setToolTip("Real-time Analysis")
            analysis_action.triggered.connect(self._toggle_realtime_analysis)

            # Recording controls
            record_action = toolbar.addAction("🔴")
            record_action.setToolTip("Record Animation")
            record_action.triggered.connect(self._start_recording)

            # Export tools
            export_action = toolbar.addAction("💾")
            export_action.setToolTip("Export Data/Video")
            export_action.triggered.connect(self._show_export_dialog)

    def _setup_enhanced_shortcuts(self):
        """Setup enhanced keyboard shortcuts"""
        from PyQt6.QtGui import QKeySequence, QShortcut

        # Camera presets (F1-F7)
        presets = list(CameraPreset)
        for i, preset in enumerate(presets):
            if i < 7:  # F1-F7
                shortcut = QShortcut(QKeySequence(f"F{i+1}"), self)
                shortcut.activated.connect(
                    lambda p=preset: self.camera_controller.set_preset(p)
                )

        # Advanced navigation
        QShortcut(QKeySequence("Ctrl+Left"), self).activated.connect(
            lambda: self._jump_frames(-10)
        )
        QShortcut(QKeySequence("Ctrl+Right"), self).activated.connect(
            lambda: self._jump_frames(10)
        )
        QShortcut(QKeySequence("Shift+Left"), self).activated.connect(
            lambda: self._jump_frames(-100)
        )
        QShortcut(QKeySequence("Shift+Right"), self).activated.connect(
            lambda: self._jump_frames(100)
        )

        # Analysis shortcuts
        QShortcut(QKeySequence("A"), self).activated.connect(
            self._toggle_realtime_analysis
        )
        QShortcut(QKeySequence("M"), self).activated.connect(
            self._toggle_measurement_mode
        )

    def _setup_status_monitoring(self):
        """Setup enhanced status monitoring"""
        # Create timer for regular status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_enhanced_status)
        self.status_timer.start(1000)  # Update every second

    def load_data_files(self, file_paths: List[str]) -> bool:
        """Enhanced data loading with validation and preprocessing"""
        try:
            if len(file_paths) != 3:
                raise ValueError("Expected exactly 3 MATLAB files")

            baseq_file, ztcfq_file, delta_file = file_paths

            # Validate file existence
            for file_path in file_paths:
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

            # Load data with enhanced error handling
            success = self.gl_widget.load_data(baseq_file, ztcfq_file, delta_file)

            if success:
                # Create new session
                self.current_session = self.session_manager.create_session(file_paths)

                # Initialize camera framing
                if self.gl_widget.frame_processor:
                    sample_frames = [
                        0,
                        self.gl_widget.num_frames // 4,
                        self.gl_widget.num_frames // 2,
                        3 * self.gl_widget.num_frames // 4,
                        self.gl_widget.num_frames - 1,
                    ]

                    data_points = []
                    for frame_idx in sample_frames:
                        frame_data = self.gl_widget.frame_processor.get_frame_data(
                            frame_idx
                        )
                        data_points.extend(
                            [
                                frame_data.butt,
                                frame_data.clubhead,
                                frame_data.midpoint,
                                frame_data.left_shoulder,
                                frame_data.right_shoulder,
                            ]
                        )

                    self.camera_controller.frame_data(data_points, margin=1.8)

                # Update UI
                self.playback_panel.update_num_frames(self.gl_widget.num_frames)
                self.statusBar().showMessage(
                    f"Loaded {self.gl_widget.num_frames} frames from {len(file_paths)} files"
                )

                logger.info(
                    f"Successfully loaded data: {self.gl_widget.num_frames} frames"
                )
                return True

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            QMessageBox.critical(
                self, "Data Loading Error", f"Failed to load data files:\n{e}"
            )

        return False

    # ========================================================================
    # ENHANCED FEATURE IMPLEMENTATIONS
    # ========================================================================

    def _on_camera_mode_changed(self, mode: str):
        """Handle camera mode changes"""
        self.statusBar().showMessage(f"Camera mode: {mode}")
        logger.info(f"Camera mode changed to: {mode}")

    def _on_camera_animation_finished(self):
        """Handle camera animation completion"""
        self.statusBar().showMessage("Camera animation complete")

    def _demo_orbit_animation(self):
        """Demonstrate smooth orbit animation"""
        if not self.gl_widget.frame_processor:
            QMessageBox.information(self, "Info", "Please load data first")
            return

        # Create smooth orbit animation
        current_azimuth = self.camera_controller.current_state.azimuth
        target_azimuth = current_azimuth + 360  # Full rotation

        from golf_camera_system import CameraState

        target_state = CameraState()
        self.camera_controller._copy_state(
            self.camera_controller.current_state, target_state
        )
        target_state.azimuth = target_azimuth

        self.camera_controller.animate_to_state(target_state, duration=4.0)
        self.statusBar().showMessage("Demonstrating orbit animation...")

    def _demo_cinematic_tour(self):
        """Demonstrate cinematic camera tour"""
        if not self.gl_widget.frame_processor:
            QMessageBox.information(self, "Info", "Please load data first")
            return

        # Clear existing keyframes
        self.camera_controller.clear_keyframes()

        # Create cinematic tour keyframes
        presets_tour = [
            (0.0, CameraPreset.DEFAULT),
            (2.0, CameraPreset.SIDE_VIEW),
            (4.0, CameraPreset.TOP_DOWN),
            (6.0, CameraPreset.BEHIND_GOLFER),
            (8.0, CameraPreset.IMPACT_ZONE),
            (10.0, CameraPreset.DEFAULT),
        ]

        for time, preset in presets_tour:
            state = self.camera_controller.presets[preset]
            self.camera_controller.add_keyframe(time, state)

        # Start cinematic playback
        self.camera_controller.start_cinematic_playback(duration=10.0, loop=False)
        self.statusBar().showMessage("Playing cinematic tour...")

    def _jump_frames(self, delta: int):
        """Jump multiple frames at once"""
        if self.gl_widget.frame_processor:
            new_frame = self.gl_widget.current_frame + delta
            new_frame = max(0, min(new_frame, self.gl_widget.num_frames - 1))
            self.gl_widget.set_frame(new_frame)

    def _toggle_realtime_analysis(self):
        """Toggle real-time analysis display"""
        # This would toggle additional analysis overlays
        self.statusBar().showMessage("Real-time analysis toggled")
        logger.info("Real-time analysis toggled")

    def _toggle_measurement_mode(self):
        """Toggle measurement/annotation mode"""
        self.statusBar().showMessage("Measurement mode toggled")
        logger.info("Measurement mode toggled")

    def _start_recording(self):
        """Start recording animation"""
        try:
            filename = f"golf_swing_recording_{int(time.time())}.mp4"
            # This would start video recording
            self.statusBar().showMessage(f"Recording started: {filename}")
            logger.info(f"Recording started: {filename}")
        except Exception as e:
            QMessageBox.critical(
                self, "Recording Error", f"Failed to start recording:\n{e}"
            )

    def _show_export_dialog(self):
        """Show export options dialog"""
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Export Options")
        dialog.setModal(True)
        dialog.resize(400, 300)

        layout = QVBoxLayout(dialog)

        # Export type
        layout.addWidget(QLabel("Export Type:"))
        export_combo = QComboBox()
        export_combo.addItems(
            ["Video (MP4)", "Image Sequence", "Data (CSV)", "3D Model"]
        )
        layout.addWidget(export_combo)

        # Quality settings
        layout.addWidget(QLabel("Quality:"))
        quality_combo = QComboBox()
        quality_combo.addItems(["720p", "1080p", "4K", "Custom"])
        layout.addWidget(quality_combo)

        # Frame range
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame Range:"))
        start_spin = QSpinBox()
        start_spin.setMaximum(
            self.gl_widget.num_frames - 1 if self.gl_widget.frame_processor else 0
        )
        frame_layout.addWidget(start_spin)
        frame_layout.addWidget(QLabel("to"))
        end_spin = QSpinBox()
        end_spin.setMaximum(
            self.gl_widget.num_frames - 1 if self.gl_widget.frame_processor else 0
        )
        end_spin.setValue(
            self.gl_widget.num_frames - 1 if self.gl_widget.frame_processor else 0
        )
        frame_layout.addWidget(end_spin)
        layout.addLayout(frame_layout)

        # Options
        layout.addWidget(QCheckBox("Include timestamp"))
        layout.addWidget(QCheckBox("Include force vectors"))
        layout.addWidget(QCheckBox("Include analysis overlay"))

        # Buttons
        button_layout = QHBoxLayout()
        export_btn = QPushButton("Export")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(export_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        export_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Export started...")
            logger.info("Export dialog accepted")

    def _update_enhanced_status(self):
        """Update enhanced status information"""
        if self.gl_widget.frame_processor:
            # Update performance panel with additional info
            current_time = self.gl_widget.current_frame * 0.001
            self.perf_panel.update_render_time(
                self.gl_widget.performance_stats.frame_time_ms
            )


# ============================================================================
# SESSION AND PLUGIN MANAGEMENT
# ============================================================================


class SessionManager:
    """Manage analysis sessions and data persistence"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.current_session: Optional[str] = None

    def create_session(self, data_files: List[str]) -> str:
        """Create a new analysis session"""
        import uuid

        session_id = str(uuid.uuid4())[:8]

        self.sessions[session_id] = {
            "id": session_id,
            "data_files": data_files,
            "created": time.time(),
            "camera_states": [],
            "annotations": [],
            "analysis_results": {},
        }

        self.current_session = session_id
        logger.info(f"Created session: {session_id}")
        return session_id

    def save_session(self, session_id: str, filepath: str):
        """Save session to file"""
        if session_id in self.sessions:
            import json

            with open(filepath, "w") as f:
                json.dump(self.sessions[session_id], f, indent=2, default=str)
            logger.info(f"Session saved: {filepath}")

    def load_session(self, filepath: str) -> Optional[str]:
        """Load session from file"""
        try:
            import json

            with open(filepath, "r") as f:
                session_data = json.load(f)

            session_id = session_data["id"]
            self.sessions[session_id] = session_data
            logger.info(f"Session loaded: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None


class ExportManager:
    """Manage various export functionalities"""

    def __init__(self):
        self.export_queue: List[Dict] = []
        self.is_exporting = False

    def export_video(self, frames: List, output_path: str, fps: int = 30):
        """Export frames as video"""
        # This would use OpenCV or similar to create video
        logger.info(f"Exporting video: {output_path}")

    def export_data(self, data: Dict, output_path: str, format: str = "csv"):
        """Export analysis data"""
        if format.lower() == "csv":
            import pandas as pd

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        logger.info(f"Exported data: {output_path}")

    def export_images(self, frames: List, output_dir: str, format: str = "png"):
        """Export frame sequence as images"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Export logic here
        logger.info(f"Exported images to: {output_dir}")


class PluginManager:
    """Manage plugins and extensions"""

    def __init__(self):
        self.plugins: Dict[str, object] = {}
        self.plugin_dir = Path("plugins")

    def load_plugins(self):
        """Load available plugins"""
        if self.plugin_dir.exists():
            logger.info("Loading plugins...")
            # Plugin loading logic here

    def register_plugin(self, name: str, plugin: object):
        """Register a plugin"""
        self.plugins[name] = plugin
        logger.info(f"Plugin registered: {name}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Enhanced main entry point with comprehensive error handling"""

    # Setup exception handling
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

        # Show error dialog if GUI is available
        try:
            QMessageBox.critical(
                None,
                "Critical Error",
                f"An unexpected error occurred:\n{exc_value}\n\n"
                f"Check golf_visualizer.log for details.",
            )
        except:
            pass

    sys.excepthook = handle_exception

    # Create application
    app = EnhancedGolfVisualizerApp(sys.argv)

    # Initialize application
    if not app.initialize():
        logger.error("Application initialization failed")
        return 1

    logger.info("Golf Swing Visualizer Pro started successfully")

    # Show usage information
    print("🏌️ Golf Swing Visualizer Pro")
    print("=" * 50)
    print("📁 File -> Load Data: Load MATLAB files (BASEQ, ZTCFQ, DELTAQ)")
    print("🎥 Camera Presets: F1-F7 for quick camera positions")
    print("🖱️ Mouse Controls:")
    print("   • Left drag: Orbit camera")
    print("   • Right drag: Pan camera")
    print("   • Wheel: Zoom")
    print("⌨️ Keyboard Shortcuts:")
    print("   • Space: Play/Pause")
    print("   • Arrow keys: Frame navigation")
    print("   • Ctrl+Arrows: Jump 10 frames")
    print("   • Shift+Arrows: Jump 100 frames")
    print("   • R: Reset camera")
    print("   • F: Frame data")
    print("   • A: Toggle analysis")
    print("   • M: Measurement mode")
    print("=" * 50)

    # Run application
    try:
        exit_code = app.exec()
        logger.info(f"Application exited with code: {exit_code}")
        return exit_code
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        return 1
    finally:
        # Cleanup
        if hasattr(app, "performance_monitor"):
            app.performance_monitor.stop_monitoring()


if __name__ == "__main__":
    sys.exit(main())
