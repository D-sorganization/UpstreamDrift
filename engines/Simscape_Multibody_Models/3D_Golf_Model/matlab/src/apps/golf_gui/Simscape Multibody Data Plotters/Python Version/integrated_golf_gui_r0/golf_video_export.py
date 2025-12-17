#!/usr/bin/env python3
"""
Video Export Module for Golf Visualizer
Export 3D golf swing animations to high-quality video

Features:
- 60/120 FPS high-quality MP4 export
- Multiple resolution options (720p, 1080p, 4K)
- Progress tracking
- Background rendering (non-blocking UI)
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from golf_data_core import RenderConfig
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


@dataclass
class VideoExportConfig:
    """Configuration for video export"""

    output_path: str = "golf_swing_export.mp4"
    fps: int = 60
    resolution: tuple[int, int] = (1920, 1080)  # Width x Height
    quality: str = "high"  # 'draft', 'medium', 'high', 'lossless'
    start_frame: int = 0
    end_frame: int | None = None  # None = all frames


class VideoExporter(QObject):
    """
    Export 3D golf swing animations to video

    Usage:
        exporter = VideoExporter(renderer, frame_processor)
        exporter.export_video(config)
    """

    # Signals
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(str)  # output_path
    error = pyqtSignal(str)  # error_message

    def __init__(self, renderer, frame_processor):
        super().__init__()
        self.renderer = renderer
        self.frame_processor = frame_processor

    def export_video(self, config: VideoExportConfig):
        """
        Export animation to video file

        Args:
            config: Video export configuration
        """
        try:
            # Validate ffmpeg is available
            if not self._check_ffmpeg():
                self.error.emit(
                    "ffmpeg not found. Please install: sudo apt install ffmpeg"
                )
                return

            # Get frame range
            total_frames = len(self.frame_processor.time_vector)
            start_frame = max(0, config.start_frame)
            end_frame = min(total_frames, config.end_frame or total_frames)
            frames_to_export = range(start_frame, end_frame)

            print(
                f"🎬 Exporting {len(frames_to_export)} frames to {config.output_path}"
            )
            print(f"   Resolution: {config.resolution[0]}x{config.resolution[1]}")
            print(f"   FPS: {config.fps}")
            print(f"   Quality: {config.quality}")

            # Setup ffmpeg process
            ffmpeg_process = self._start_ffmpeg_process(config)

            # Render and write frames
            for i, frame_idx in enumerate(frames_to_export):
                # Get frame data
                frame_data = self.frame_processor.get_frame_data(frame_idx)

                # Render to buffer
                frame_buffer = self._render_frame_to_buffer(
                    frame_data, config.resolution
                )

                # Write to ffmpeg
                ffmpeg_process.stdin.write(frame_buffer.tobytes())

                # Update progress
                self.progress.emit(i + 1, len(frames_to_export))

                if (i + 1) % 10 == 0:
                    print(f"   Rendered {i + 1}/{len(frames_to_export)} frames...")

            # Finalize video
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()

            if ffmpeg_process.returncode == 0:
                print(f"✅ Video exported successfully to {config.output_path}")
                self.finished.emit(config.output_path)
            else:
                error_msg = (
                    f"ffmpeg failed with return code {ffmpeg_process.returncode}"
                )
                print(f"❌ {error_msg}")
                self.error.emit(error_msg)

        except Exception as e:
            error_msg = f"Video export failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
            self.error.emit(error_msg)

    def _start_ffmpeg_process(self, config: VideoExportConfig) -> subprocess.Popen:
        """Start ffmpeg process with appropriate settings"""

        width, height = config.resolution

        # Quality presets
        quality_settings = {
            "draft": {"preset": "ultrafast", "crf": "28"},
            "medium": {"preset": "medium", "crf": "23"},
            "high": {"preset": "slow", "crf": "18"},
            "lossless": {"preset": "slow", "crf": "0"},
        }

        settings = quality_settings.get(config.quality, quality_settings["high"])

        # Build ffmpeg command
        # Use absolute path to prevent argument injection (starting with -)
        output_abspath = str(Path(config.output_path).resolve())

        command = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(config.fps),
            "-i",
            "-",  # Read from stdin
            "-an",  # No audio
            "-vcodec",
            "libx264",
            "-preset",
            settings["preset"],
            "-crf",
            settings["crf"],
            "-pix_fmt",
            "yuv420p",  # Compatibility with most players
            output_abspath,
        ]

        print(
            f"   Running ffmpeg with preset '{settings['preset']}', CRF {settings['crf']}"
        )

        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _render_frame_to_buffer(
        self, frame_data, resolution: tuple[int, int]
    ) -> np.ndarray:
        """
        Render frame to RGB buffer

        Args:
            frame_data: Frame data to render
            resolution: (width, height) for rendering

        Returns:
            RGB buffer as numpy array (height, width, 3)
        """
        width, height = resolution

        # Setup render config
        render_config = RenderConfig()
        render_config.show_ground = True
        render_config.show_club = True
        render_config.show_body_segments = {
            "left_forearm": True,
            "left_upper_arm": True,
            "right_forearm": True,
            "right_upper_arm": True,
            "left_shoulder_neck": True,
            "right_shoulder_neck": True,
        }

        # Calculate view matrices (face-on view)
        view_matrix = self._calculate_view_matrix()
        proj_matrix = self._calculate_projection_matrix(width, height)
        view_position = np.array([0.0, 1.5, 3.0], dtype=np.float32)

        # Create offscreen framebuffer if needed
        if not hasattr(self, "_fbo") or self._fbo_size != resolution:
            self._create_offscreen_framebuffer(width, height)

        # Bind framebuffer and render
        self._fbo.use()

        self.renderer.set_viewport(width, height)
        self.renderer.render_frame(
            frame_data,
            {},  # dynamics_data
            render_config,
            view_matrix,
            proj_matrix,
            view_position,
        )

        # Read pixels from framebuffer
        raw_pixels = self._fbo.read(components=3)

        # Convert to numpy array
        pixels = np.frombuffer(raw_pixels, dtype=np.uint8)
        pixels = pixels.reshape((height, width, 3))

        # Flip vertically (OpenGL coordinates are bottom-up)
        pixels = np.flipud(pixels)

        return pixels

    def _create_offscreen_framebuffer(self, width: int, height: int):
        """Create offscreen framebuffer for rendering"""
        ctx = self.renderer.ctx

        self._fbo_texture = ctx.texture((width, height), 3)
        self._fbo_depth = ctx.depth_renderbuffer((width, height))
        self._fbo = ctx.framebuffer(
            color_attachments=[self._fbo_texture], depth_attachment=self._fbo_depth
        )
        self._fbo_size = (width, height)

        print(f"   Created offscreen framebuffer: {width}x{height}")

    def _calculate_view_matrix(self) -> np.ndarray:
        """Calculate view matrix for camera (face-on view)"""
        camera_pos = np.array([0.0, 1.5, 3.0], dtype=np.float32)
        target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = -camera_pos

        return view_matrix

    def _calculate_projection_matrix(self, width: int, height: int) -> np.ndarray:
        """Calculate projection matrix"""
        aspect = width / height
        fov = 45.0
        near = 0.1
        far = 100.0

        f = 1.0 / np.tan(np.radians(fov) / 2.0)

        proj_matrix = np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

        return proj_matrix

    @staticmethod
    def _check_ffmpeg() -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


# ============================================================================
# THREADED VIDEO EXPORT (Non-blocking UI)
# ============================================================================


class VideoExportThread(QThread):
    """
    Video export in background thread (doesn't freeze UI)

    Usage:
        thread = VideoExportThread(renderer, frame_processor, config)
        thread.progress.connect(lambda c, t: print(f"{c}/{t}"))
        thread.finished.connect(lambda p: print(f"Done: {p}"))
        thread.start()
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, renderer, frame_processor, config: VideoExportConfig):
        super().__init__()
        self.renderer = renderer
        self.frame_processor = frame_processor
        self.config = config

    def run(self):
        """Run export in background thread"""
        exporter = VideoExporter(self.renderer, self.frame_processor)

        # Connect signals
        exporter.progress.connect(self.progress.emit)
        exporter.finished.connect(self.finished.emit)
        exporter.error.connect(self.error.emit)

        # Export
        exporter.export_video(self.config)


# ============================================================================
# UI DIALOG
# ============================================================================


class VideoExportDialog(QDialog):
    """
    User-friendly video export dialog

    Integration into golf_gui_application.py:
    Simply create and show this dialog when user wants to export
    """

    def __init__(self, parent, renderer, frame_processor):
        super().__init__(parent)
        self.renderer = renderer
        self.frame_processor = frame_processor
        self.export_thread = None

        self.setWindowTitle("Export Golf Swing Video")
        self.setMinimumWidth(500)
        self._setup_ui()

    def _setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Export 3D Golf Swing Animation to Video")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Output file
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Output File:"))
        self.file_input = QLineEdit("golf_swing.mp4")
        file_layout.addWidget(self.file_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self._browse_output_file)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(
            ["1280x720 (HD)", "1920x1080 (Full HD)", "2560x1440 (2K)", "3840x2160 (4K)"]
        )
        self.res_combo.setCurrentIndex(1)  # Default to 1080p
        res_layout.addWidget(self.res_combo)
        layout.addLayout(res_layout)

        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(24, 240)
        self.fps_spin.setValue(60)
        self.fps_spin.setSuffix(" FPS")
        fps_layout.addWidget(self.fps_spin)
        layout.addLayout(fps_layout)

        # Quality
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(
            [
                "Draft (Fast, Large File)",
                "Medium (Balanced)",
                "High (Slow, Best Quality)",
                "Lossless (Very Slow, Huge File)",
            ]
        )
        self.quality_combo.setCurrentIndex(2)  # Default to high
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)

        # Info label
        info = QLabel(
            "\n💡 Tip: 60 FPS High quality recommended for smooth playback.\n"
            "Higher resolutions and frame rates will take longer to export."
        )
        info.setStyleSheet(
            "color: #666; padding: 10px; background-color: #f9f9f9; border-radius: 4px;"
        )
        layout.addWidget(info)

        # Buttons
        button_layout = QHBoxLayout()
        export_btn = QPushButton("Export Video")
        export_btn.clicked.connect(self._start_export)
        button_layout.addWidget(export_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _browse_output_file(self):
        """Browse for output file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Video As", "golf_swing.mp4", "Video Files (*.mp4 *.avi *.mov)"
        )

        if filename:
            self.file_input.setText(filename)

    def _start_export(self):
        """Start video export"""
        # Parse resolution
        res_text = self.res_combo.currentText()
        res_map = {
            "1280x720 (HD)": (1280, 720),
            "1920x1080 (Full HD)": (1920, 1080),
            "2560x1440 (2K)": (2560, 1440),
            "3840x2160 (4K)": (3840, 2160),
        }
        resolution = res_map[res_text]

        # Parse quality
        quality_map = {
            "Draft (Fast, Large File)": "draft",
            "Medium (Balanced)": "medium",
            "High (Slow, Best Quality)": "high",
            "Lossless (Very Slow, Huge File)": "lossless",
        }
        quality = quality_map[self.quality_combo.currentText()]

        # Create config
        config = VideoExportConfig(
            output_path=self.file_input.text(),
            fps=self.fps_spin.value(),
            resolution=resolution,
            quality=quality,
        )

        # Close dialog
        self.accept()

        # Show progress dialog
        progress_dialog = QProgressDialog(
            "Exporting video...", "Cancel", 0, 100, self.parent()
        )
        progress_dialog.setWindowTitle("Video Export")
        progress_dialog.setWindowModality(2)  # Application modal
        progress_dialog.setMinimumDuration(0)  # Show immediately

        # Start export thread
        self.export_thread = VideoExportThread(
            self.renderer, self.frame_processor, config
        )

        # Connect signals
        self.export_thread.progress.connect(
            lambda current, total: progress_dialog.setValue(int(100 * current / total))
        )
        self.export_thread.finished.connect(
            lambda path: self._on_export_finished(progress_dialog, path)
        )
        self.export_thread.error.connect(
            lambda err: self._on_export_error(progress_dialog, err)
        )

        # Handle cancel
        progress_dialog.canceled.connect(
            lambda: self.export_thread.terminate() if self.export_thread else None
        )

        # Start
        self.export_thread.start()
        progress_dialog.exec()

    def _on_export_finished(self, progress_dialog, output_path):
        """Handle export completion"""
        progress_dialog.close()

        QMessageBox.information(
            self.parent(),
            "Export Complete",
            f"Video exported successfully!\n\n{output_path}\n\n"
            f"You can now play the video in any media player.",
        )

    def _on_export_error(self, progress_dialog, error_msg):
        """Handle export error"""
        progress_dialog.close()

        QMessageBox.critical(
            self.parent(),
            "Export Failed",
            f"Video export failed:\n\n{error_msg}\n\n"
            f"Make sure ffmpeg is installed:\nsudo apt install ffmpeg",
        )


if __name__ == "__main__":
    print("🎥 Golf Video Export Module")
    print("\nFeatures:")
    print("  ✅ High-quality MP4 export (60/120 FPS)")
    print("  ✅ Multiple resolutions (720p to 4K)")
    print("  ✅ Background rendering (non-blocking UI)")
    print("  ✅ Progress tracking")
    print("\nRequirements:")
    print("  - ffmpeg: sudo apt install ffmpeg")
