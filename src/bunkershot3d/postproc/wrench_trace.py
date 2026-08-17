"""
Wrench extraction and processing on the clubhead.
"""

import numpy as np
from scipy.signal import butter, filtfilt

#: Relative tolerance on the spread of sample intervals before a trace is
#: considered non-uniformly sampled (issue #8617, finding B25).
_UNIFORM_RTOL = 1e-6


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

    def _uniform_dt(self) -> float:
        """Return the sample interval, proving the trace is uniformly sampled.

        Returns:
            The sample interval [s].

        Raises:
            ValueError: If there are fewer than 2 samples, time is not strictly
                increasing, or the sampling is not uniform. A mean interval is
                meaningless for non-uniform sampling, so deriving a sample rate
                from it would silently mis-scale the filter cutoff (B25).
        """
        time = np.asarray(self.time, dtype=float).reshape(-1)
        if time.size < 2:
            raise ValueError(f"filter() needs at least 2 samples, got {time.size}")
        intervals = np.diff(time)
        if np.any(intervals <= 0.0):
            raise ValueError("filter() requires strictly increasing sample times")
        dt = float(intervals.mean())
        if not np.allclose(intervals, dt, rtol=_UNIFORM_RTOL, atol=0.0):
            spread = float(intervals.max() - intervals.min())
            raise ValueError(
                "filter() requires uniformly sampled data; sample intervals span "
                f"{spread:.3e} s around a mean of {dt:.3e} s. "
                "Call resample() onto a uniform grid first."
            )
        return dt

    def filter(self, cutoff_freq: float = 2000.0, order: int = 4) -> "WrenchTrace":
        """
        Apply a zero-phase low-pass Butterworth filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz.
            order: Filter order.
        Returns:
            A new WrenchTrace with filtered forces and torques. When
            ``cutoff_freq`` is at or above the Nyquist frequency there is
            nothing to remove, so an unfiltered copy is returned.
        Raises:
            ValueError: If the trace is not uniformly sampled, or is too short
                for ``filtfilt`` at this order (it needs more than
                ``3 * (order + 1)`` samples for its default padding).
        """
        dt = self._uniform_dt()
        fs = 1.0 / dt
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq

        # In case the sampling frequency is too low for the requested cutoff
        if normal_cutoff >= 1.0:
            return WrenchTrace(
                self.time.copy(), self.force_world.copy(), self.torque_world.copy()
            )

        n_samples = int(np.asarray(self.time).size)
        min_samples = 3 * (order + 1) + 1
        if n_samples < min_samples:
            raise ValueError(
                f"filtfilt at order {order} needs at least {min_samples} samples "
                f"(3 * (order + 1) of default padding), got {n_samples}. "
                "Lower the order or resample onto a longer grid."
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
