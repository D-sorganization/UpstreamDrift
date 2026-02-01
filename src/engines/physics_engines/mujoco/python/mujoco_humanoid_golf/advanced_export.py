"""Advanced export formats for golf swing data.

Supports:
- MATLAB .mat files
- C3D motion capture format
- HDF5 hierarchical data
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_availability import (
    C3D_AVAILABLE,
    EZC3D_AVAILABLE,
)
from src.shared.python.engine_availability import HDF5_AVAILABLE as H5PY_AVAILABLE
from src.shared.python.engine_availability import (
    SCIPY_AVAILABLE,
)

# Conditional imports for optional dependencies
if SCIPY_AVAILABLE:
    from scipy.io import savemat

if H5PY_AVAILABLE:
    import h5py


def export_to_matlab(
    output_path: str,
    data_dict: dict[str, Any],
    compress: bool = True,
) -> bool:
    """Export recording to MATLAB .mat format.

    Args:
        output_path: Output .mat file path
        data_dict: Dictionary containing recording data
        compress: Whether to compress the file

    Returns:
        True if successful
    """
    if not SCIPY_AVAILABLE:
        msg = "scipy required for MATLAB export (pip install scipy)"
        raise ImportError(msg)

    try:
        # Convert all data to MATLAB-compatible format
        output_data: dict[str, Any] = {}

        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                # MATLAB uses Fortran (column-major) order
                output_data[key] = np.asarray(value, order="F")
            elif isinstance(value, list | tuple):
                output_data[key] = np.array(value, order="F")
            elif isinstance(value, int | float | str | bool):
                output_data[key] = value
            elif isinstance(value, dict):
                # Nested dict - flatten keys
                for subkey, subvalue in value.items():
                    flat_key = f"{key}_{subkey}".replace(" ", "_")
                    if isinstance(subvalue, np.ndarray):
                        output_data[flat_key] = np.asarray(subvalue, order="F")
                    elif isinstance(subvalue, list | tuple):
                        output_data[flat_key] = np.array(subvalue, order="F")
                    else:
                        output_data[flat_key] = subvalue

        # Save to .mat file
        savemat(
            output_path,
            output_data,
            do_compression=compress,
            format="5",  # MATLAB 5 format (compatible with most versions)
            oned_as="column",  # Save 1D arrays as column vectors
        )

        return True

    except Exception:
        return False


def export_to_hdf5(
    output_path: str,
    data_dict: dict[str, Any],
    compression: str = "gzip",
) -> bool:
    """Export recording to HDF5 format.

    Args:
        output_path: Output .h5 file path
        data_dict: Dictionary containing recording data
        compression: Compression method ('gzip', 'lzf', or None)

    Returns:
        True if successful
    """
    if not H5PY_AVAILABLE:
        msg = "h5py required for HDF5 export (pip install h5py)"
        raise ImportError(msg)

    try:
        with h5py.File(output_path, "w") as f:
            # Create groups for organization
            timeseries_group = f.create_group("timeseries")
            metadata_group = f.create_group("metadata")
            f.create_group("statistics")

            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    # Store arrays in timeseries group
                    # Only compress arrays larger than threshold
                    min_size_for_compression = 100
                    timeseries_group.create_dataset(
                        key,
                        data=value,
                        compression=(
                            compression
                            if value.size > min_size_for_compression
                            else None
                        ),
                    )
                elif isinstance(value, int | float):
                    # Store scalars as attributes
                    metadata_group.attrs[key] = value
                elif isinstance(value, str):
                    # Store strings as attributes
                    metadata_group.attrs[key] = value
                elif isinstance(value, dict):
                    # Create subgroup for nested dict
                    subgroup = f.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            # Only compress arrays larger than threshold
                            min_size_for_compression = 100
                            subgroup.create_dataset(
                                subkey,
                                data=subvalue,
                                compression=(
                                    compression
                                    if subvalue.size > min_size_for_compression
                                    else None
                                ),
                            )
                        else:
                            subgroup.attrs[subkey] = subvalue

        return True

    except Exception:
        return False


def export_to_c3d(
    output_path: str,
    times: np.ndarray,
    joint_positions: np.ndarray,
    joint_names: list,
    forces: np.ndarray | None = None,
    moments: np.ndarray | None = None,
    frame_rate: float = 60.0,
    units: dict[str, str] | None = None,
) -> bool:
    """Export recording to C3D motion capture format.

    Args:
        output_path: Output .c3d file path
        times: Time array (N,)
        joint_positions: Joint positions (N, nq)
        joint_names: Names of joints
        forces: Optional force data (N, nforces, 3)
        moments: Optional moment data (N, nforces, 3)
        frame_rate: Sampling rate in Hz
        units: Dictionary of units (position, force, moment)

    Returns:
        True if successful
    """
    if not EZC3D_AVAILABLE and not C3D_AVAILABLE:
        msg = "ezc3d or c3d required for C3D export (pip install ezc3d)"
        raise ImportError(msg)

    if units is None:
        units = {"position": "mm", "force": "N", "moment": "Nmm"}  # C3D standard is mm

    try:
        if EZC3D_AVAILABLE:
            return _export_to_c3d_ezc3d(
                output_path,
                times,
                joint_positions,
                joint_names,
                forces,
                moments,
                frame_rate,
                units,
            )
        return _export_to_c3d_py(
            output_path,
            times,
            joint_positions,
            joint_names,
            forces,
            moments,
            frame_rate,
            units,
        )

    except Exception:
        return False


def _export_to_c3d_ezc3d(
    output_path: str,
    times: np.ndarray,
    joint_positions: np.ndarray,
    joint_names: list,
    forces: np.ndarray | None,
    moments: np.ndarray | None,
    frame_rate: float,
    units: dict[str, str],
) -> bool:
    """Export using ezc3d library."""
    import ezc3d

    # Create new C3D file
    c = ezc3d.c3d()

    # Set frame rate
    c["parameters"]["POINT"]["RATE"]["value"] = [frame_rate]
    c["parameters"]["POINT"]["UNITS"]["value"] = [units["position"]]

    # Number of frames and markers
    num_frames = len(times)
    num_markers = joint_positions.shape[1]

    # Set marker labels
    c["parameters"]["POINT"]["LABELS"]["value"] = joint_names[:num_markers]

    # Prepare point data (4 x num_markers x num_frames)
    # Format: [X, Y, Z, residual] for each marker
    points = np.zeros((4, num_markers, num_frames))

    # Assume joint positions are 1D angles, create dummy 3D positions
    # In real application, would use actual 3D coordinates from MuJoCo
    for i in range(num_markers):
        # Create spiral pattern for visualization (placeholder)
        angles = joint_positions[:, i]
        radius = (i + 1) * 100  # mm
        points[0, i, :] = radius * np.cos(angles)  # X
        points[1, i, :] = radius * np.sin(angles)  # Y
        points[2, i, :] = np.arange(num_frames) * 10  # Z (progression)
        points[3, i, :] = 0  # Residual (0 = good data)

    c["data"]["points"] = points

    # Add analog data (forces/moments) if provided
    if forces is not None or moments is not None:
        analog_data = []
        analog_labels = []

        if forces is not None:
            num_force_plates = forces.shape[1]
            for fp in range(num_force_plates):
                for axis, label in enumerate(["X", "Y", "Z"]):
                    analog_data.append(forces[:, fp, axis])
                    analog_labels.append(f"Force{fp + 1}_{label}")

        if moments is not None:
            num_moment_plates = moments.shape[1]
            for mp in range(num_moment_plates):
                for axis, label in enumerate(["X", "Y", "Z"]):
                    analog_data.append(moments[:, mp, axis])
                    analog_labels.append(f"Moment{mp + 1}_{label}")

        if analog_data:
            # Analog data format: (num_channels, num_samples)
            c["data"]["analogs"] = np.array(analog_data)
            c["parameters"]["ANALOG"]["LABELS"]["value"] = analog_labels
            c["parameters"]["ANALOG"]["RATE"]["value"] = [frame_rate]
            c["parameters"]["FORCE_PLATFORM"]["UNITS"]["value"] = [units["force"]]

    # Write file
    c.write(output_path)
    return True


def _export_to_c3d_py(
    output_path: str,
    times: np.ndarray,
    joint_positions: np.ndarray,
    joint_names: list,
    forces: np.ndarray | None,
    moments: np.ndarray | None,
    frame_rate: float,
    units: dict[str, str],
) -> bool:
    """Export using c3d library (fallback)."""
    import c3d

    writer = c3d.Writer(point_rate=frame_rate)

    num_frames = len(times)
    num_markers = joint_positions.shape[1]

    # Add point data
    for frame_idx in range(num_frames):
        points = []
        for marker_idx in range(num_markers):
            angle = joint_positions[frame_idx, marker_idx]
            radius = (marker_idx + 1) * 100  # mm

            # Create 3D position from angle (placeholder)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = frame_idx * 10

            points.append([x, y, z, 0.0, 0.0])  # [x, y, z, residual, cameras]

        writer.add_frames([(np.array(points), np.array([]))])

    # Write file
    with open(output_path, "wb") as f:
        writer.write(f)

    return True


def export_recording_all_formats(
    base_path: str,
    data_dict: dict[str, Any],
    formats: list | None = None,
) -> dict[str, bool]:
    """Export recording in multiple formats.

    Args:
        base_path: Base path without extension
        data_dict: Recording data dictionary
        formats: List of formats to export

    Returns:
        Dictionary mapping format to success status
    """
    from .telemetry import export_telemetry_csv, export_telemetry_json

    if formats is None:
        formats = ["json", "csv", "mat", "hdf5"]
    base_path_obj = Path(base_path)
    results = {}

    for fmt in formats:
        try:
            output_path = base_path_obj.with_suffix(f".{fmt}")

            if fmt == "json":
                success = export_telemetry_json(str(output_path), data_dict)
            elif fmt == "csv":
                success = export_telemetry_csv(str(output_path), data_dict)
            elif fmt == "mat":
                success = export_to_matlab(str(output_path), data_dict)
            elif fmt in ["hdf5", "h5"]:
                output_path = base_path_obj.with_suffix(".h5")
                success = export_to_hdf5(str(output_path), data_dict)
            elif fmt == "c3d":
                # C3D needs special handling
                times = data_dict.get("times", np.array([]))
                positions = data_dict.get("joint_positions", np.array([]))

                # Handle 1D positions array
                num_joints = 0
                if positions.size > 0:
                    if positions.ndim > 1:
                        num_joints = positions.shape[1]
                    else:
                        num_joints = 1
                        positions = positions.reshape(-1, 1)

                joint_names = data_dict.get(
                    "joint_names",
                    [f"Joint{i}" for i in range(num_joints)],
                )
                forces = data_dict.get("ground_reaction_forces")

                success = export_to_c3d(
                    str(output_path),
                    times,
                    positions,
                    joint_names,
                    forces=forces,
                )
            else:
                success = False

            results[fmt] = success

        except Exception:
            results[fmt] = False

    return results


def create_matlab_script(
    output_path: str,
    mat_file: str,
    script_type: str = "plot",
) -> None:
    """Create a MATLAB script to load and visualize exported data.

    Args:
        output_path: Output .m script path
        mat_file: Path to .mat file (relative or absolute)
        script_type: Type of script ('plot', 'analyze', 'animate')
    """
    mat_file = Path(mat_file).name  # Use just filename

    if script_type == "plot":
        script = f"""% MATLAB Script to plot golf swing data
% Auto-generated by Golf Swing Analysis Suite

% Load data
data = load('{mat_file}');

% Extract time series
t = data.times;

% Create figure with subplots
figure('Name', 'Golf Swing Analysis', 'Position', [100, 100, 1200, 800]);

% Subplot 1: Joint Angles
subplot(2, 2, 1);
plot(t, rad2deg(data.joint_positions));
xlabel('Time (s)');
ylabel('Joint Angles (deg)');
title('Joint Angles');
grid on;
legend('show');

% Subplot 2: Joint Velocities
subplot(2, 2, 2);
plot(t, rad2deg(data.joint_velocities));
xlabel('Time (s)');
ylabel('Joint Velocities (deg/s)');
title('Joint Velocities');
grid on;

% Subplot 3: Joint Torques
subplot(2, 2, 3);
plot(t, data.joint_torques);
xlabel('Time (s)');
ylabel('Torques (Nm)');
title('Applied Torques');
grid on;

% Subplot 4: Club Head Speed
if isfield(data, 'club_head_speed')
    subplot(2, 2, 4);
    plot(t, data.club_head_speed);
    xlabel('Time (s)');
    ylabel('Club Head Speed (mph)');
    title('Club Head Speed');
    grid on;

    % Mark peak
    [peak_speed, peak_idx] = max(data.club_head_speed);
    hold on;
    plot(t(peak_idx), peak_speed, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    text(t(peak_idx), peak_speed, sprintf('  Peak: %.1f mph', peak_speed), ...
        'VerticalAlignment', 'bottom');
    hold off;
end

% Overall title
sgtitle('Golf Swing Biomechanical Analysis');
"""

    elif script_type == "analyze":
        script = f"""% MATLAB Script to analyze golf swing data
% Auto-generated by Golf Swing Analysis Suite

% Load data
data = load('{mat_file}');

% Extract time series
t = data.times;

fprintf('=== Golf Swing Analysis ===\\n\\n');

% Basic metrics
fprintf('Duration: %.2f s\\n', t(end) - t(1));
fprintf('Number of samples: %d\\n', length(t));
fprintf('Sample rate: %.1f Hz\\n', 1/mean(diff(t)));
fprintf('\\n');

% Club head speed analysis
if isfield(data, 'club_head_speed')
    [peak_speed, peak_idx] = max(data.club_head_speed);
    fprintf('Peak club head speed: %.1f mph at t=%.3f s\\n', ...
        peak_speed, t(peak_idx));
    fprintf('\\n');
end

% Joint ROM analysis
fprintf('Range of Motion (degrees):\\n');
joint_angles_deg = rad2deg(data.joint_positions);
for i = 1:size(joint_angles_deg, 2)
    rom = max(joint_angles_deg(:, i)) - min(joint_angles_deg(:, i));
    fprintf('  Joint %d: %.1f deg\\n', i, rom);
end
fprintf('\\n');

% Energy analysis
if isfield(data, 'kinetic_energy') && isfield(data, 'potential_energy')
    max_ke = max(data.kinetic_energy);
    max_pe = max(data.potential_energy);
    fprintf('Maximum kinetic energy: %.2f J\\n', max_ke);
    fprintf('Maximum potential energy: %.2f J\\n', max_pe);
end
"""

    else:  # animate
        script = f"""% MATLAB Script to animate golf swing
% Auto-generated by Golf Swing Analysis Suite

% Load data
data = load('{mat_file}');

% Extract time series
t = data.times;
positions = data.joint_positions;

% Create figure
figure('Name', 'Golf Swing Animation', 'Position', [100, 100, 800, 600]);

% Animation loop
for i = 1:length(t)
    clf;

    % Plot current joint configuration
    % (Customize based on your model structure)
    plot(positions(i, :), 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Joint Index');
    ylabel('Angle (rad)');
    title(sprintf('Time: %.3f s', t(i)));
    grid on;
    ylim([min(positions(:)), max(positions(:))]);

    drawnow;
    pause(0.016);  % 60 FPS
end
"""

    with open(output_path, "w") as f:
        f.write(script)


def get_available_export_formats() -> dict[str, dict[str, Any]]:
    """Get information about available export formats.

    Returns:
        Dictionary mapping format name to info dict
    """
    return {
        "json": {
            "name": "JSON",
            "extension": ".json",
            "available": True,
            "description": "JavaScript Object Notation - universal format",
        },
        "csv": {
            "name": "CSV",
            "extension": ".csv",
            "available": True,
            "description": "Comma-Separated Values - spreadsheet compatible",
        },
        "mat": {
            "name": "MATLAB",
            "extension": ".mat",
            "available": SCIPY_AVAILABLE,
            "description": "MATLAB MAT-File - for MATLAB/Simulink analysis",
        },
        "hdf5": {
            "name": "HDF5",
            "extension": ".h5",
            "available": H5PY_AVAILABLE,
            "description": "Hierarchical Data Format - efficient for large datasets",
        },
        "c3d": {
            "name": "C3D",
            "extension": ".c3d",
            "available": EZC3D_AVAILABLE or C3D_AVAILABLE,
            "description": "Motion Capture Standard - compatible with Vicon, etc.",
        },
    }
