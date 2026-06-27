# mypy: disable-error-code="attr-defined,has-type,arg-type"
"""Data-loading and playback helpers for the legacy motion-capture plotter."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from mocap_data_loader import (
    find_available_joints,
    get_simscape_joint_positions,
    parse_excel_row,
    process_excel_sheet,
    safe_float,
)
from PyQt6.QtWidgets import QFileDialog, QMessageBox

logger = logging.getLogger(__name__)


class MotionCapturePlotterDataMixin:
    _safe_float = staticmethod(safe_float)
    _parse_excel_row = staticmethod(parse_excel_row)
    _simscape_joint_position_definitions = staticmethod(get_simscape_joint_positions)
    _find_available_joints = staticmethod(find_available_joints)

    def auto_load_excel_file(self) -> None:
        """Automatically load the Excel file if it exists in the current directory."""
        excel_files = [f for f in os.listdir(".") if f.endswith((".xlsx", ".xls"))]
        if excel_files:
            # Try to load the first Excel file found
            filename = excel_files[0]
            logger.info(f"Auto-loading Excel file: {filename}")
            self.load_excel_file(filename)

    def load_file(self) -> None:
        """Load data file based on current data source."""
        if self.current_data_source == "Motion Capture (Excel)":
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Excel File", "", "Excel Files (*.xlsx *.xls)"
            )
            if filename:
                self.load_excel_file(filename)
        else:  # Simscape Multibody (CSV)
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load CSV File", "", "CSV Files (*.csv)"
            )
            if filename:
                self.load_simscape_csv(filename)

    def on_data_source_changed(self, source) -> None:
        """Handle data source change."""
        if source is None:
            raise ValueError("source must be provided")
        self.current_data_source = source
        logger.info(f"Data source changed to: {source}")

        # Update visibility flags based on data source
        if source == "Motion Capture (Excel)":
            self.show_motion_capture = True
            self.show_simscape = False
            # Try to auto-load Excel file
            self.auto_load_excel_file()
        elif source == "Simscape Multibody (CSV)":
            self.show_motion_capture = False
            self.show_simscape = True
            # Try to auto-load CSV file
            self.auto_load_simscape_csv()
        else:  # Both (Simultaneous)
            self.show_motion_capture = True
            self.show_simscape = True
            # Try to auto-load both files
            self.auto_load_excel_file()
            self.auto_load_simscape_csv()

        # Update visualization
        self.update_visualization()

    def auto_load_simscape_csv(self) -> None:
        """Automatically load the Simscape CSV file if it exists."""
        csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
        if csv_files:
            filename = csv_files[0]
            logger.info(f"Auto-loading Simscape CSV file: {filename}")
            self.load_simscape_csv(filename)

    def _process_excel_sheet(self, filename, sheet_name) -> None:
        """Process a single Excel sheet and store parsed frames in swing_data."""
        if filename is None:
            raise ValueError("filename must be provided")
        result = process_excel_sheet(filename, sheet_name)
        if result is not None:
            self.swing_data[sheet_name] = result
            self.print_data_debug(sheet_name)

    def load_excel_file(self, filename) -> None:
        """Load and process Excel file."""
        try:
            logger.info(f"Loading file: {filename}")
            excel_file = pd.ExcelFile(filename)
            logger.info(f"Available sheets: {excel_file.sheet_names}")

            for sheet_name in ["TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"]:
                if sheet_name in excel_file.sheet_names:
                    self._process_excel_sheet(filename, sheet_name)

            # Update swing selection
            self.swing_combo.clear()
            self.swing_combo.addItems(list(self.swing_data.keys()))

            if self.swing_data:
                self.current_swing = next(iter(self.swing_data.keys()))
                logger.info(f"Selected swing: {self.current_swing}")
                self.swing_combo.setCurrentText(self.current_swing)
                self.setup_frame_slider()
                self.update_visualization()
            else:
                logger.info("No valid swing data found in the file")

        except ImportError as e:
            logger.error(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def load_simscape_csv(self, filename) -> None:
        """Load and process Simscape CSV file."""
        try:
            logger.info(f"Loading Simscape CSV file: {filename}")

            df = pd.read_csv(filename)
            logger.debug(
                f"Successfully loaded CSV with {len(df)} rows "
                f"and {len(df.columns)} columns"  # noqa: E501
            )
            logger.info(
                f"Time range: {df['time'].min():.3f} to {df['time'].max():.3f} seconds"
            )  # noqa: E501

            joint_positions = self._simscape_joint_position_definitions()
            available_joints = self._find_available_joints(joint_positions, df.columns)

            if not available_joints:
                raise ValueError("No valid joint position data found in the CSV file")

            # Process the data into our standard format
            # ⚡ Bolt: Vectorized dataframe construction avoids .iterrows() overhead and is significantly faster
            out_df = pd.DataFrame({"time": df["time"]})
            for joint_name, columns in available_joints.items():
                if all(col in df.columns for col in columns):
                    out_df[f"{joint_name}_X"] = df[columns[0]]
                    out_df[f"{joint_name}_Y"] = df[columns[1]]
                    out_df[f"{joint_name}_Z"] = df[columns[2]]

            # Store the data
            swing_name = "Simscape_Swing"
            self.simscape_data[swing_name] = out_df
            logger.debug(f"Successfully loaded {len(out_df)} frames for {swing_name}")

            # Update swing selection
            self.swing_combo.clear()
            self.swing_combo.addItems(list(self.swing_data.keys()))

            if self.swing_data:
                self.current_swing = swing_name
                logger.info(f"Selected swing: {self.current_swing}")
                self.swing_combo.setCurrentText(self.current_swing)
                self.setup_frame_slider()
                self.update_visualization()
            else:
                logger.info("No valid swing data found in the file")

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Error loading Simscape CSV file: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load Simscape CSV file: {str(e)}"
            )  # noqa: E501

    def print_data_debug(self, sheet_name) -> None:
        """Print debug information about the loaded data."""
        if sheet_name in self.swing_data:
            data = self.swing_data[sheet_name]
            if not data.empty:
                logger.debug(f"\n=== Data Debug for {sheet_name} ===")
                logger.info(f"Number of frames: {len(data)}")
                logger.info(
                    f"Time range: {data['time'].min():.3f} to "
                    f"{data['time'].max():.3f} seconds"  # noqa: E501
                )
                logger.info("Mid-Hands Position ranges:")
                logger.info(
                    f"  X: {data['mid_X'].min():.3f} to {data['mid_X'].max():.3f}"
                )  # noqa: E501
                logger.info(
                    f"  Y: {data['mid_Y'].min():.3f} to {data['mid_Y'].max():.3f}"
                )  # noqa: E501
                logger.info(
                    f"  Z: {data['mid_Z'].min():.3f} to {data['mid_Z'].max():.3f}"
                )  # noqa: E501

                logger.info("Club Head Position ranges:")
                logger.info(
                    f"  X: {data['club_X'].min():.3f} to {data['club_X'].max():.3f}"
                )  # noqa: E501
                logger.info(
                    f"  Y: {data['club_Y'].min():.3f} to {data['club_Y'].max():.3f}"
                )  # noqa: E501
                logger.info(
                    f"  Z: {data['club_Z'].min():.3f} to {data['club_Z'].max():.3f}"
                )  # noqa: E501

                # Calculate total position ranges
                mid_range = np.max(
                    [
                        data["mid_X"].max() - data["mid_X"].min(),
                        data["mid_Y"].max() - data["mid_Y"].min(),
                        data["mid_Z"].max() - data["mid_Z"].min(),
                    ]
                )
                club_range = np.max(
                    [
                        data["club_X"].max() - data["club_X"].min(),
                        data["club_Y"].max() - data["club_Y"].min(),
                        data["club_Z"].max() - data["club_Z"].min(),
                    ]
                )

                logger.info(f"Mid-Hands motion range: {mid_range:.3f}")
                logger.info(f"Club Head motion range: {club_range:.3f}")

                logger.info("Data Analysis:")
                logger.info(
                    "  This data contains both mid-hands and club head positions"
                )  # noqa: E501
                logger.info(
                    "  Using actual measured positions instead of calculated ones"
                )  # noqa: E501
                logger.info(
                    "  Original data in inches, converted to meters for visualization"
                )  # noqa: E501
                logger.info(
                    "  Direction cosines (Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz) "
                    "are unitless"  # noqa: E501
                )
                logger.info("  Motion scaling applied to make visualization clearer")
                logger.info("=" * 40)

    def setup_frame_slider(self) -> None:
        """Setup the frame slider."""
        if self.current_swing in self.swing_data:
            data = self.swing_data[self.current_swing]
            max_frame = len(data) - 1
            self.frame_slider.setRange(0, max_frame)
            self.frame_slider.setValue(0)
            self.current_frame = 0

    def on_swing_change(self, swing_name) -> None:
        """Handle swing selection change."""
        if swing_name in self.swing_data:
            self.current_swing = swing_name
            self.setup_frame_slider()
            self.update_visualization()

    def on_frame_change(self, frame) -> None:
        """Handle frame slider change."""
        if frame is None:
            raise ValueError("frame must be provided")
        self.current_frame = frame
        self.frame_label.setText(str(frame))
        self.update_visualization()

    def on_speed_change(self, speed) -> None:
        """Handle speed slider change."""
        if speed is None:
            raise ValueError("speed must be provided")
        self.speed_label.setText(str(speed))
        if self.is_playing:
            self.animation_timer.setInterval(1000 // speed)

    def on_scale_change(self, scale) -> None:
        """Handle motion scale change."""
        if scale is None:
            raise ValueError("scale must be provided")
        self.motion_scale = scale
        self.scale_label.setText(f"{scale}x")
        self.update_visualization()

    def on_club_length_change(self, length_cm) -> None:
        """Handle club length change."""
        if length_cm is None:
            raise ValueError("length_cm must be provided")
        self.shaft_length = length_cm / 100.0  # Convert cm to meters
        self.club_label.setText(f"{self.shaft_length:.1f}m")
        self.update_visualization()

    def toggle_playback(self) -> None:
        """Toggle play/pause."""
        if self.is_playing:
            self.animation_timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            speed = self.speed_slider.value()
            self.animation_timer.start(1000 // speed)
            self.play_btn.setText("Pause")
            self.is_playing = True

    def next_frame(self) -> None:
        """Advance to next frame."""
        if self.current_swing in self.swing_data:
            data = self.swing_data[self.current_swing]
            if self.current_frame < len(data) - 1:
                self.current_frame += 1
                self.frame_slider.setValue(self.current_frame)
            else:
                # Loop back to start
                self.current_frame = 0
                self.frame_slider.setValue(0)
