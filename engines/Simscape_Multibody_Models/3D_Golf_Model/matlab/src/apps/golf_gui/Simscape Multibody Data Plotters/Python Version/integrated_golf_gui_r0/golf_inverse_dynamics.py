import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Apply a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def savitzky_golay_filter(data, window_length=9, polyorder=3):
    """Apply a Savitzky-Golay filter."""
    if window_length % 2 == 0:
        window_length += 1  # Must be odd
    return signal.savgol_filter(data, window_length, polyorder)


def moving_average_filter(data, window_size=5):
    """Apply a moving average filter."""
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def calculate_derivatives(data, time):
    """Calculate velocity and acceleration using splines for accuracy."""
    spline = UnivariateSpline(time, data, s=0)
    velocity = spline.derivative(n=1)(time)
    acceleration = spline.derivative(n=2)(time)
    return velocity, acceleration


def calculate_inverse_dynamics(
    position_data, orientation_data, time_vector, club_mass=0.2, eval_offset=0.0
):
    """
    Calculate inverse dynamics (forces and torques).

    Args:
        position_data (np.array): Array of shape (N, 3) for X, Y, Z position.
        orientation_data (np.array): Array of shape (N, 3, 3) for rotation matrices.
        time_vector (np.array): Array of shape (N,) for time.
        club_mass (float): Mass of the club.
        eval_offset (float): Evaluation point offset in inches.

    Returns:
        dict: A dictionary containing forces and torques.
    """
    num_frames = position_data.shape[0]

    # Convert offset from inches to meters
    offset_m = eval_offset * 0.0254

    # Calculate derivatives for the club head position
    vx, ax = calculate_derivatives(position_data[:, 0], time_vector)
    vy, ay = calculate_derivatives(position_data[:, 1], time_vector)
    vz, az = calculate_derivatives(position_data[:, 2], time_vector)

    velocity = np.vstack([vx, vy, vz]).T
    acceleration = np.vstack([ax, ay, az]).T

    # Apply offset along the shaft's Z-axis (assuming Z is the shaft direction)
    offset_vec = np.array([0, 0, offset_m])

    # Transform offset vector by club orientation at each frame
    adjusted_acceleration = np.zeros_like(acceleration)
    for i in range(num_frames):
        # This is a simplified assumption. A proper implementation would need
        # to calculate angular velocity and acceleration to get the tangential
        # and centripetal acceleration at the offset point.
        # For now, we just apply the linear acceleration.
        # rotation_matrix = orientation_data[i]
        # rotated_offset = rotation_matrix @ offset_vec
        # adjusted_pos = position_data[i] + rotated_offset
        # Re-calculating derivatives for adjusted_pos would be more accurate
        # but computationally expensive. We'll stick to a simpler model here.
        adjusted_acceleration[i] = acceleration[i]

    # Calculate Force (F = ma)
    force = adjusted_acceleration * club_mass

    # Calculate Torque (simplified)
    # T = r x F, where r is the lever arm from a pivot (e.g., hands)
    # This is a highly simplified model. A full model would require
    # moment of inertia and angular acceleration.
    # For visualization, we can represent torque proportional to force.
    torque = np.cross(offset_vec, force)

    return {"force": force, "torque": torque, "velocity": velocity}
