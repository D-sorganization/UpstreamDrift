"""
Wrench extraction and processing on the clubhead.
"""

import numpy as np
from scipy.signal import butter, filtfilt


class WrenchTrace:
    """Processes force and torque applied by the sand on the clubhead."""

    def __init__(
        self, time: np.ndarray, force_world: np.ndarray, torque_world: np.ndarray
    ) -> None:
        """
        Initialize the WrenchTrace.
        Args:
            time: (N,) array of time points
            force_world: (N, 3) array of forces in the world frame
            torque_world: (N, 3) array of torques in the world frame
        """
        self.time = time
        self.force_world = force_world
        self.torque_world = torque_world

    def filter(self, cutoff_freq: float = 2000.0, order: int = 4) -> "WrenchTrace":
        """
        Apply a low-pass Butterworth filter.
        Args:
            cutoff_freq: Cutoff frequency in Hz.
            order: Filter order.
        Returns:
            A new WrenchTrace with filtered forces and torques.
        """
        dt = np.mean(np.diff(self.time))
        fs = 1.0 / dt
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq

        # In case the sampling frequency is too low for the requested cutoff
        if normal_cutoff >= 1.0:
            return WrenchTrace(
                self.time.copy(), self.force_world.copy(), self.torque_world.copy()
            )

        b, a = butter(order, normal_cutoff, btype="low", analog=False)

        f_filt = filtfilt(b, a, self.force_world, axis=0)
        t_filt = filtfilt(b, a, self.torque_world, axis=0)

        return WrenchTrace(self.time, f_filt, t_filt)

    def resample(self, target_times: np.ndarray) -> "WrenchTrace":
        """Resample wrench onto a common time grid using linear interpolation."""
        f_res = np.zeros((len(target_times), 3))
        t_res = np.zeros((len(target_times), 3))

        for i in range(3):
            f_res[:, i] = np.interp(target_times, self.time, self.force_world[:, i])
            t_res[:, i] = np.interp(target_times, self.time, self.torque_world[:, i])

        return WrenchTrace(target_times, f_res, t_res)

    def get_impulses(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate force and torque over time to get linear and angular impulses.
        Returns:
            linear_impulse (3,), angular_impulse (3,)
        """
        lin_imp = np.trapezoid(self.force_world, x=self.time, axis=0)
        ang_imp = np.trapezoid(self.torque_world, x=self.time, axis=0)
        return lin_imp, ang_imp
