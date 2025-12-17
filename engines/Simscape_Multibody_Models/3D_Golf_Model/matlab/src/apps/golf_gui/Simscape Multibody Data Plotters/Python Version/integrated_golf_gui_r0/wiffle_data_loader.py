#!/usr/bin/env python3
"""
Wiffle_ProV1 Data Loader for Golf Swing Visualizer
Handles Excel-based motion capture data and converts to the GUI's expected format
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_numpy(series: Any) -> np.ndarray:
    """Convert pandas Series, numpy array, or ExtensionArray to numpy array safely"""
    if hasattr(series, "to_numpy"):
        return series.to_numpy()
    elif hasattr(series, "values"):
        return series.values
    else:
        return np.asarray(series)


@dataclass
class MotionDataConfig:
    """Configuration for motion capture data processing"""

    # Excel sheet names - updated based on actual file structure
    prov1_sheet: str = "TW_ProV1"  # Top Wood ProV1 data
    wiffle_sheet: str = "TW_wiffle"  # Top Wood Wiffle data
    # Alternative sheets available: GW_ProV11, GW_wiffle (Ground Wood)

    # Column mappings for ProV1 data
    prov1_columns: dict[str, str] | None = None

    # Column mappings for Wiffle data
    wiffle_columns: dict[str, str] | None = None

    # Data processing options
    normalize_time: bool = True
    filter_noise: bool = True
    interpolate_missing: bool = True

    def __post_init__(self):
        """Set default column mappings if not provided"""
        if self.prov1_columns is None:
            self.prov1_columns = {
                "time": "Time",
                "clubhead_x": "CHx",
                "clubhead_y": "CHy",
                "clubhead_z": "CHz",
                "butt_x": "Bx",
                "butt_y": "By",
                "butt_z": "Bz",
                "midpoint_x": "MPx",
                "midpoint_y": "MPy",
                "midpoint_z": "MPz",
                "left_wrist_x": "LWx",
                "left_wrist_y": "LWy",
                "left_wrist_z": "LWz",
                "left_elbow_x": "LEx",
                "left_elbow_y": "LEy",
                "left_elbow_z": "LEz",
                "left_shoulder_x": "LSx",
                "left_shoulder_y": "LSy",
                "left_shoulder_z": "LSz",
                "right_wrist_x": "RWx",
                "right_wrist_y": "RWy",
                "right_wrist_z": "RWz",
                "right_elbow_x": "REx",
                "right_elbow_y": "REy",
                "right_elbow_z": "REz",
                "right_shoulder_x": "RSx",
                "right_shoulder_y": "RSy",
                "right_shoulder_z": "RSz",
                "hub_x": "Hx",
                "hub_y": "Hy",
                "hub_z": "Hz",
            }

        if self.wiffle_columns is None:
            self.wiffle_columns = {
                "time": "Time",
                "clubhead_x": "CHx",
                "clubhead_y": "CHy",
                "clubhead_z": "CHz",
                "butt_x": "Bx",
                "butt_y": "By",
                "butt_z": "Bz",
                "midpoint_x": "MPx",
                "midpoint_y": "MPy",
                "midpoint_z": "MPz",
                "left_wrist_x": "LWx",
                "left_wrist_y": "LWy",
                "left_wrist_z": "LWz",
                "left_elbow_x": "LEx",
                "left_elbow_y": "LEy",
                "left_elbow_z": "LEz",
                "left_shoulder_x": "LSx",
                "left_shoulder_y": "LSy",
                "left_shoulder_z": "LSz",
                "right_wrist_x": "RWx",
                "right_wrist_y": "RWy",
                "right_wrist_z": "RWz",
                "right_elbow_x": "REx",
                "right_elbow_y": "REy",
                "right_elbow_z": "REz",
                "right_shoulder_x": "RSx",
                "right_shoulder_y": "RSy",
                "right_shoulder_z": "RSz",
                "hub_x": "Hx",
                "hub_y": "Hy",
                "hub_z": "Hz",
            }


class MotionDataLoader:
    """Loader for motion capture Excel data"""

    def __init__(self, config: MotionDataConfig | None = None):
        self.config = config or MotionDataConfig()
        self.data_cache: dict[str, Any] = {}

    def load_data(self) -> dict[str, pd.DataFrame]:
        """
        Load Wiffle_ProV1 data from the default Excel file location

        Returns:
            Dictionary with 'ProV1' and 'Wiffle' DataFrames
        """
        # Try to find the Excel file in common locations
        possible_paths = [
            Path("../../../Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx"),
            Path("../Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx"),
            Path("../../Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx"),
            Path("../../../Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx"),
            Path("Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx"),
            Path("Wiffle_ProV1_club_3D_data.xlsx"),  # Current directory
        ]

        for path in possible_paths:
            if path.exists():
                return self.load_excel_data(str(path))

        raise FileNotFoundError(
            "Wiffle_ProV1 Excel file not found in any expected location"
        )

    def load_from_file(self, filepath: str) -> dict[str, pd.DataFrame]:
        """
        Load Wiffle_ProV1 Excel data from a specific file path

        Args:
            filepath: Path to the Excel file

        Returns:
            Dictionary with 'ProV1' and 'Wiffle' DataFrames
        """
        return self.load_excel_data(filepath)

    def load_excel_data(self, filepath: str) -> dict[str, pd.DataFrame]:
        """
        Load Wiffle_ProV1 Excel data and convert to GUI-compatible format

        Args:
            filepath: Path to the Excel file

        Returns:
            Dictionary with 'ProV1' and 'Wiffle' DataFrames
        """
        filepath_path = Path(filepath)
        if not filepath_path.exists():
            raise FileNotFoundError(f"Excel file not found: {filepath}")

        print(f"[INFO] Loading Wiffle_ProV1 data from: {filepath}")

        try:
            # Read both sheets
            prov1_data = pd.read_excel(
                filepath_path, sheet_name=self.config.prov1_sheet
            )
            wiffle_data = pd.read_excel(
                filepath_path, sheet_name=self.config.wiffle_sheet
            )

            print(f"[OK] Loaded ProV1 data: {prov1_data.shape}")
            print(f"[OK] Loaded Wiffle data: {wiffle_data.shape}")

            # Process and clean data
            prov1_processed = self._process_sheet_data(prov1_data, "ProV1")
            wiffle_processed = self._process_sheet_data(wiffle_data, "Wiffle")

            return {"ProV1": prov1_processed, "Wiffle": wiffle_processed}

        except Exception as e:
            raise RuntimeError(f"Error loading Excel data: {e}") from e

    def _process_sheet_data(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Process and clean sheet data"""
        print(f"[PROC] Processing {sheet_name} data...")

        # Based on the analysis, the structure is:
        # Row 0: Metadata (ball type, parameters)
        # Row 1: Point labels (Mid-hands, Center of club face)
        # Row 2: Column headers (Sample #, Time, X, Y, Z, Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz)
        # Row 3+: Actual data

        if len(df) < 3:
            print(f"[WARN] Insufficient rows in {sheet_name}, creating dummy data")
            return self._create_dummy_data(100)

        # Extract headers from row 2 (index 2)
        headers = df.iloc[2]
        data_df = df.iloc[3:].copy()
        data_df.columns = headers

        # Reset index
        data_df = data_df.reset_index(drop=True)

        # Extract time and position data
        processed_data = pd.DataFrame()

        # Time column (column 1)
        if len(data_df.columns) >= 2:
            processed_data["time"] = pd.to_numeric(data_df.iloc[:, 1], errors="coerce")
            print(f"[OK] Extracted time data from column 1 for {sheet_name}")
        else:
            print(f"[WARN] No Time column found in {sheet_name}, creating linear time")
            processed_data["time"] = np.linspace(0, 1, len(data_df))

        # Map the actual columns to our expected format
        # The data has position (X, Y, Z) and orientation (Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz) data
        # We'll use the position data for the clubhead and create reasonable estimates for other body parts

        # Clubhead position (using X, Y, Z columns)
        # The data has two sets of position data: columns 2-4 (Mid-hands) and 14-16 (Center of club face)
        # We'll use the first set (Mid-hands) as the primary position data

        # Check for position columns by index (more reliable than name matching)
        if len(data_df.columns) >= 16:
            # Use the first set of X, Y, Z (columns 2, 3, 4)
            processed_data["clubhead_x"] = pd.to_numeric(
                data_df.iloc[:, 2], errors="coerce"
            )
            processed_data["clubhead_y"] = pd.to_numeric(
                data_df.iloc[:, 3], errors="coerce"
            )
            processed_data["clubhead_z"] = pd.to_numeric(
                data_df.iloc[:, 4], errors="coerce"
            )

            print(
                f"[OK] Extracted position data from columns 2-4 (Mid-hands) for {sheet_name}"
            )
        else:
            print(
                f"[WARN] Insufficient columns in {sheet_name}, using first 3 numeric columns"
            )
            numeric_cols = data_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                processed_data["clubhead_x"] = pd.to_numeric(
                    data_df[numeric_cols[0]], errors="coerce"
                )
                processed_data["clubhead_y"] = pd.to_numeric(
                    data_df[numeric_cols[1]], errors="coerce"
                )
                processed_data["clubhead_z"] = pd.to_numeric(
                    data_df[numeric_cols[2]], errors="coerce"
                )
            else:
                print(
                    f"[WARN] Insufficient numeric columns in {sheet_name}, creating dummy data"
                )
                processed_data["clubhead_x"] = np.linspace(0, 1, len(processed_data))
                processed_data["clubhead_y"] = np.linspace(0, 1, len(processed_data))
                processed_data["clubhead_z"] = np.linspace(0, 1, len(processed_data))

        # Create reasonable estimates for other body parts based on clubhead position
        # This is a simplified model - in a real application, you'd want more sophisticated biomechanical modeling
        self._create_body_part_estimates(processed_data, sheet_name)

        # Convert to numeric and handle errors
        numeric_columns = [col for col in processed_data.columns if col != "time"]
        for col in numeric_columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors="coerce")

        # Normalize time if requested
        if self.config.normalize_time:
            processed_data["time"] = (
                processed_data["time"] - processed_data["time"].min()
            ) / (processed_data["time"].max() - processed_data["time"].min())

        # Filter noise if requested
        if self.config.filter_noise:
            processed_data = self._apply_noise_filtering(processed_data)

        # Interpolate missing values if requested
        if self.config.interpolate_missing:
            processed_data = self._interpolate_missing_values(processed_data)

        print(
            f"[OK] Processed {sheet_name}: {processed_data.shape}, "
            f"time range: [{processed_data['time'].min():.3f}, {processed_data['time'].max():.3f}]"
        )

        return processed_data

    def _create_body_part_estimates(
        self, processed_data: pd.DataFrame, sheet_name: str
    ) -> None:
        """Create reasonable estimates for body parts based on clubhead position"""
        # This is a simplified biomechanical model
        # In a real application, you'd want more sophisticated modeling based on actual motion capture data

        # Get clubhead position
        ch_x = processed_data["clubhead_x"].values
        ch_y = processed_data["clubhead_y"].values
        ch_z = processed_data["clubhead_z"].values

        # Create estimates for other body parts
        # These are simplified relationships - adjust based on your biomechanical model

        # Convert pandas Series to numpy arrays for arithmetic operations
        ch_x_array = _to_numpy(ch_x)
        ch_y_array = _to_numpy(ch_y)
        ch_z_array = _to_numpy(ch_z)

        # Club butt (opposite end of clubhead)
        processed_data["butt_x"] = ch_x_array - 0.5  # 0.5 units behind clubhead
        processed_data["butt_y"] = ch_y_array + 0.1  # Slightly above
        processed_data["butt_z"] = ch_z_array + 0.2  # Slightly to the side

        # Club midpoint
        processed_data["midpoint_x"] = ch_x_array - 0.25
        processed_data["midpoint_y"] = ch_y_array + 0.05
        processed_data["midpoint_z"] = ch_z_array + 0.1

        # Left wrist (assuming left-handed golfer or left hand on club)
        processed_data["left_wrist_x"] = ch_x_array - 0.3
        processed_data["left_wrist_y"] = ch_y_array + 0.15
        processed_data["left_wrist_z"] = ch_z_array + 0.3

        # Left elbow
        processed_data["left_elbow_x"] = ch_x_array - 0.4
        processed_data["left_elbow_y"] = ch_y_array + 0.25
        processed_data["left_elbow_z"] = ch_z_array + 0.4

        # Left shoulder
        processed_data["left_shoulder_x"] = ch_x_array - 0.5
        processed_data["left_shoulder_y"] = ch_y_array + 0.35
        processed_data["left_shoulder_z"] = ch_z_array + 0.5

        # Right wrist
        processed_data["right_wrist_x"] = ch_x_array - 0.35
        processed_data["right_wrist_y"] = ch_y_array + 0.2
        processed_data["right_wrist_z"] = ch_z_array + 0.25

        # Right elbow
        processed_data["right_elbow_x"] = ch_x_array - 0.45
        processed_data["right_elbow_y"] = ch_y_array + 0.3
        processed_data["right_elbow_z"] = ch_z_array + 0.35

        # Right shoulder
        processed_data["right_shoulder_x"] = ch_x_array - 0.55
        processed_data["right_shoulder_y"] = ch_y_array + 0.4
        processed_data["right_shoulder_z"] = ch_z_array + 0.45

        # Hub (center of rotation, roughly between shoulders)
        processed_data["hub_x"] = ch_x_array - 0.52
        processed_data["hub_y"] = ch_y_array + 0.37
        processed_data["hub_z"] = ch_z_array + 0.47

        print(
            f"[CALC] Created body part estimates for {sheet_name} based on clubhead position"
        )

    def _create_dummy_data(self, num_frames: int) -> pd.DataFrame:
        """Create dummy data for testing purposes"""
        processed_data = pd.DataFrame()
        processed_data["time"] = np.linspace(0, 1, num_frames)

        # Create simple motion pattern
        t = _to_numpy(processed_data["time"])
        processed_data["clubhead_x"] = np.sin(2 * np.pi * t) * 2
        processed_data["clubhead_y"] = np.cos(2 * np.pi * t) * 2
        processed_data["clubhead_z"] = t * 2 - 1

        # Create dummy data for other body parts
        for pos in [
            "butt",
            "midpoint",
            "left_wrist",
            "left_elbow",
            "left_shoulder",
            "right_wrist",
            "right_elbow",
            "right_shoulder",
            "hub",
        ]:
            processed_data[f"{pos}_x"] = processed_data[
                "clubhead_x"
            ] + np.random.normal(0, 0.1, num_frames)
            processed_data[f"{pos}_y"] = processed_data[
                "clubhead_y"
            ] + np.random.normal(0, 0.1, num_frames)
            processed_data[f"{pos}_z"] = processed_data[
                "clubhead_z"
            ] + np.random.normal(0, 0.1, num_frames)

        return processed_data

    def _apply_noise_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply noise filtering to position data"""
        from scipy.signal import savgol_filter

        # Apply Savitzky-Golay filter to position columns
        position_columns = [
            col for col in df.columns if any(axis in col for axis in ["_x", "_y", "_z"])
        ]

        for col in position_columns:
            if not df[col].isna().all():
                # Use window length of 5% of data length, minimum 5
                window_length = max(5, len(df) // 20)
                if window_length % 2 == 0:
                    window_length += 1  # Must be odd

                try:
                    filtered_data = savgol_filter(df[col].ffill(), window_length, 3)
                    df[col] = filtered_data
                except Exception as e:
                    print(f"[WARN] Could not filter {col}: {e}")

        return df

    def _interpolate_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values in the dataset"""
        # Interpolate missing values for position columns
        position_columns = [
            col for col in df.columns if any(axis in col for axis in ["_x", "_y", "_z"])
        ]

        for col in position_columns:
            if df[col].isna().any():
                # Use forward fill then backward fill to handle edge cases
                df[col] = df[col].interpolate(method="linear").ffill().bfill()

        return df

    def convert_to_gui_format(
        self, excel_data: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convert Excel data to the format expected by the GUI

        Args:
            excel_data: Dictionary with 'ProV1' and 'Wiffle' DataFrames

        Returns:
            Tuple of (BASEQ, ZTCFQ, DELTAQ) DataFrames for GUI compatibility
        """
        print("[CONV] Converting to GUI format...")

        # Use ProV1 data as the primary dataset (BASEQ equivalent)
        prov1_df = excel_data["ProV1"]
        wiffle_df = excel_data["Wiffle"]

        # Create BASEQ format (primary motion data)
        baseq_data = self._create_baseq_format(prov1_df)

        # Create ZTCFQ format (secondary motion data - using Wiffle)
        ztcfq_data = self._create_baseq_format(wiffle_df)

        # Create DELTAQ format (difference between ProV1 and Wiffle)
        deltaq_data = self._create_deltaq_format(prov1_df, wiffle_df)

        print("[OK] Converted to GUI format:")
        print(f"   BASEQ: {baseq_data.shape}")
        print(f"   ZTCFQ: {ztcfq_data.shape}")
        print(f"   DELTAQ: {deltaq_data.shape}")

        return baseq_data, ztcfq_data, deltaq_data

    def _create_baseq_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BASEQ format DataFrame from position data"""
        # Create a DataFrame with the structure expected by the GUI
        baseq_data = pd.DataFrame()

        # Add time column
        baseq_data["Time"] = df["time"]

        # Add position data in the expected format
        position_mappings = {
            "CHx": "clubhead_x",
            "CHy": "clubhead_y",
            "CHz": "clubhead_z",
            "Bx": "butt_x",
            "By": "butt_y",
            "Bz": "butt_z",
            "MPx": "midpoint_x",
            "MPy": "midpoint_y",
            "MPz": "midpoint_z",
            "LWx": "left_wrist_x",
            "LWy": "left_wrist_y",
            "LWz": "left_wrist_z",
            "LEx": "left_elbow_x",
            "LEy": "left_elbow_y",
            "LEz": "left_elbow_z",
            "LSx": "left_shoulder_x",
            "LSy": "left_shoulder_y",
            "LSz": "left_shoulder_z",
            "RWx": "right_wrist_x",
            "RWy": "right_wrist_y",
            "RWz": "right_wrist_z",
            "REx": "right_elbow_x",
            "REy": "right_elbow_y",
            "REz": "right_elbow_z",
            "RSx": "right_shoulder_x",
            "RSy": "right_shoulder_y",
            "RSz": "right_shoulder_z",
            "Hx": "hub_x",
            "Hy": "hub_y",
            "Hz": "hub_z",
        }

        for gui_col, data_col in position_mappings.items():
            if data_col in df.columns:
                baseq_data[gui_col] = df[data_col]
            else:
                baseq_data[gui_col] = 0.0  # Default value

        return baseq_data

    def _create_deltaq_format(
        self, prov1_df: pd.DataFrame, wiffle_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create DELTAQ format showing differences between ProV1 and Wiffle"""
        # Align the dataframes by time
        common_time = _to_numpy(prov1_df["time"])

        deltaq_data = pd.DataFrame()
        deltaq_data["Time"] = common_time

        # Calculate differences for each position component
        position_components = [
            "clubhead",
            "butt",
            "midpoint",
            "left_wrist",
            "left_elbow",
            "left_shoulder",
            "right_wrist",
            "right_elbow",
            "right_shoulder",
            "hub",
        ]

        for component in position_components:
            for axis in ["_x", "_y", "_z"]:
                prov1_col = component + axis
                wiffle_col = component + axis

                if prov1_col in prov1_df.columns and wiffle_col in wiffle_df.columns:
                    # Interpolate wiffle data to match prov1 time points
                    wiffle_interp = np.interp(
                        common_time, wiffle_df["time"], wiffle_df[wiffle_col]
                    )
                    diff = _to_numpy(prov1_df[prov1_col]) - wiffle_interp

                    # Store in DELTAQ format
                    gui_col = (
                        f"{component.upper().replace('_', '')[:2]}{axis[-1].upper()}"
                    )
                    deltaq_data[gui_col] = diff
                else:
                    deltaq_data[
                        f"{component.upper().replace('_', '')[:2]}{axis[-1].upper()}"
                    ] = 0.0

        return deltaq_data


def main():
    """Test the Wiffle data loader"""
    print("[TEST] Testing Wiffle Data Loader")

    # Find the Excel file - try multiple possible paths
    possible_paths = [
        Path("../../../Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx"),
        Path("Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx"),
        Path("../Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx"),
        Path("../../Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx"),
        Path("Wiffle_ProV1_club_3D_data.xlsx"),
    ]

    excel_file = None
    for path in possible_paths:
        if path.exists():
            excel_file = path
            break

    if excel_file is None:
        print("[ERROR] Excel file not found. Tried paths:")
        for path in possible_paths:
            print(f"   {path}")
        return

    try:
        # Create loader and load data
        loader = MotionDataLoader()
        excel_data = loader.load_excel_data(excel_file)

        # Convert to GUI format
        baseq, ztcfq, deltaq = loader.convert_to_gui_format(excel_data)

        print("\n[SUMMARY] Data Summary:")
        print(f"ProV1 data points: {len(excel_data['ProV1'])}")
        print(f"Wiffle data points: {len(excel_data['Wiffle'])}")
        print(
            f"Time range: {excel_data['ProV1']['time'].min():.3f} - {excel_data['ProV1']['time'].max():.3f}"
        )

        print("\n[OK] Wiffle data loader test completed successfully!")

    except Exception as e:
        print(f"[ERROR] Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
