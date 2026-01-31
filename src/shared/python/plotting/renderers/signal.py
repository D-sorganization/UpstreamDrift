"""Signal analysis plotting renderer."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from src.shared.python.plotting.renderers.base import BaseRenderer


class SignalRenderer(BaseRenderer):
    """Renderer for signal processing and nonlinear analysis plots."""

    def plot_jerk_trajectory(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot jerk (rate of change of acceleration) over time."""
        times, velocities = self.data.get_series("joint_velocities")
        _, accelerations = self.data.get_series("joint_accelerations")

        velocities = np.asarray(velocities)
        accelerations = np.asarray(accelerations)

        if len(times) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.01
        fs = 1.0 / dt

        try:
            from src.shared.python import signal_processing

            use_sp = True
        except ImportError:
            use_sp = False

        ax = fig.add_subplot(111)

        if joint_indices is None:
            limit = min(velocities.shape[1], 3)
            joint_indices = list(range(limit))

        for idx in joint_indices:
            if idx >= velocities.shape[1]:
                continue

            if len(accelerations) > 0 and idx < accelerations.shape[1]:
                acc = accelerations[:, idx]
            else:
                acc = np.gradient(velocities[:, idx], dt)

            if use_sp:
                jerk = signal_processing.compute_jerk(acc, float(fs))
            else:
                jerk = np.gradient(acc, dt)

            label = self.data.get_joint_name(idx)
            ax.plot(times, jerk, label=label, linewidth=1.5)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Jerk (rad/s³)", fontsize=12, fontweight="bold")
        ax.set_title("Joint Jerk Trajectory", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_frequency_analysis(
        self,
        fig: Figure,
        joint_idx: int = 0,
        signal_type: str = "velocity",
    ) -> None:
        """Plot frequency content (PSD) of a joint signal."""
        if signal_type == "position":
            _, data = self.data.get_series("joint_positions")
            ylabel = "PSD (rad²/Hz)"
            title = "Joint Position PSD"
        elif signal_type == "torque":
            _, data = self.data.get_series("joint_torques")
            ylabel = "PSD (Nm²/Hz)"
            title = "Joint Torque PSD"
        else:  # velocity
            _, data = self.data.get_series("joint_velocities")
            ylabel = "PSD ((rad/s)²/Hz)"
            title = "Joint Velocity PSD"

        data = np.asarray(data)
        if data.ndim < 2 or joint_idx >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        signal_data = data[:, joint_idx]

        times, _ = self.data.get_series("joint_positions")
        if len(times) < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return

        dt = float(np.mean(np.diff(times)))
        fs = 1.0 / dt

        try:
            from src.shared.python import signal_processing

            freqs, psd = signal_processing.compute_psd(signal_data, fs)
        except ImportError:
            from scipy import signal

            freqs, psd = signal.welch(signal_data, fs=fs)

        ax = fig.add_subplot(111)
        ax.semilogy(freqs, psd, color=self.colors["primary"], linewidth=2)

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_title(f"{title}: {joint_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        fig.tight_layout()

    def plot_spectrogram(
        self,
        fig: Figure,
        joint_idx: int = 0,
        signal_type: str = "velocity",
    ) -> None:
        """Plot spectrogram of a joint signal."""
        if signal_type == "position":
            _, data = self.data.get_series("joint_positions")
            title = "Joint Position Spectrogram"
        elif signal_type == "torque":
            _, data = self.data.get_series("joint_torques")
            title = "Joint Torque Spectrogram"
        else:  # velocity
            _, data = self.data.get_series("joint_velocities")
            title = "Joint Velocity Spectrogram"

        data = np.asarray(data)
        if data.ndim < 2 or joint_idx >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        signal_data = data[:, joint_idx]
        times, _ = self.data.get_series("joint_positions")
        if len(times) < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return

        dt = float(np.mean(np.diff(times)))
        fs = 1.0 / dt

        try:
            from src.shared.python import signal_processing

            f, t, Sxx = signal_processing.compute_spectrogram(signal_data, fs)
        except ImportError:
            from scipy import signal

            f, t, Sxx = signal.spectrogram(signal_data, fs=fs)

        ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(
            t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="inferno"
        )

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_title(f"{title}: {joint_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Power Spectral Density (dB)", rotation=270, labelpad=15)
        fig.tight_layout()

    def plot_multiscale_entropy(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
        max_scale: int = 20,
    ) -> None:
        """Plot Multiscale Entropy (MSE) curves."""
        try:
            from src.shared.python.statistical_analysis import StatisticalAnalyzer
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Analysis module missing", ha="center", va="center")
            return

        times, velocities = self.data.get_series("joint_velocities")
        velocities = np.asarray(velocities)

        if len(times) == 0 or velocities.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        analyzer = StatisticalAnalyzer(
            times=np.asarray(times),
            joint_positions=np.zeros_like(velocities),  # Dummy
            joint_velocities=velocities,
            joint_torques=np.zeros_like(velocities),  # Dummy
        )

        ax = fig.add_subplot(111)

        if joint_indices is None:
            limit = min(velocities.shape[1], 3)
            joint_indices = list(range(limit))

        for idx in joint_indices:
            if idx >= velocities.shape[1]:
                continue
            data = velocities[:, idx]
            scales, mse = analyzer.compute_multiscale_entropy(data, max_scale=max_scale)

            label = self.data.get_joint_name(idx)
            ax.plot(scales, mse, marker="o", label=label, linewidth=2)

        ax.set_xlabel("Scale Factor (Time)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sample Entropy", fontsize=12, fontweight="bold")
        ax.set_title("Multiscale Entropy Analysis", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_lyapunov_exponent(
        self,
        fig: Figure,
        joint_idx: int = 0,
        tau: int = 5,
        dim: int = 3,
    ) -> None:
        """Plot divergence of nearest neighbors over time to estimate Lyapunov Exponent."""
        try:
            from src.shared.python.statistical_analysis import StatisticalAnalyzer
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Analysis module missing", ha="center", va="center")
            return

        times, positions = self.data.get_series("joint_positions")
        _, velocities = self.data.get_series("joint_velocities")

        if len(times) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        analyzer = StatisticalAnalyzer(
            times=np.asarray(times),
            joint_positions=np.asarray(positions),
            joint_velocities=np.asarray(velocities),
            joint_torques=np.zeros_like(positions),
        )

        velocities = np.asarray(velocities)
        if joint_idx >= velocities.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Joint index out of bounds", ha="center", va="center")
            return

        data_1d = velocities[:, joint_idx]

        try:
            time_div, divergence, slope = analyzer.compute_lyapunov_divergence(
                data_1d,
                tau=tau,
                dim=dim,
            )
        except AttributeError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Method not implemented", ha="center", va="center")
            return

        if len(time_div) == 0:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Insufficient data for Lyapunov analysis",
                ha="center",
                va="center",
            )
            return

        ax = fig.add_subplot(111)
        ax.plot(
            time_div,
            divergence,
            "o-",
            color=self.colors["primary"],
            markersize=3,
            label="Divergence",
        )

        limit = len(time_div) // 2 if len(time_div) > 10 else len(time_div)
        if limit > 1:
            fit_slope, intercept = np.polyfit(time_div[:limit], divergence[:limit], 1)
            fit_line = fit_slope * time_div + intercept
            ax.plot(
                time_div, fit_line, "r--", linewidth=2, label=f"MLE = {fit_slope:.3f}"
            )

        name = self.data.get_joint_name(joint_idx)
        ax.set_title(
            f"Lyapunov Exponent Estimation: {name}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("ln(Divergence)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

    def plot_wavelet_scalogram(
        self,
        fig: Figure,
        joint_idx: int,
        signal_type: str = "velocity",
        freq_range: tuple[float, float] = (1.0, 50.0),
        title_prefix: str = "",
    ) -> None:
        """Plot Continuous Wavelet Transform (CWT) scalogram."""
        try:
            from src.shared.python import signal_processing
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Signal Processing module missing", ha="center", va="center"
            )
            return

        if signal_type == "position":
            times, data = self.data.get_series("joint_positions")
            title_prefix = title_prefix or "Position"
        elif signal_type == "torque":
            times, data = self.data.get_series("joint_torques")
            title_prefix = title_prefix or "Torque"
        else:
            times, data = self.data.get_series("joint_velocities")
            title_prefix = title_prefix or "Velocity"

        data = np.asarray(data)

        if len(times) == 0 or data.ndim < 2 or joint_idx >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        signal_data = data[:, joint_idx]
        dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.01
        fs = 1.0 / dt

        try:
            freqs, _, cwt_matrix = signal_processing.compute_cwt(
                signal_data, fs, freq_range=freq_range
            )
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"CWT Error: {e}", ha="center", va="center")
            return

        power = np.abs(cwt_matrix) ** 2

        ax = fig.add_subplot(111)
        T, F = np.meshgrid(times, freqs)

        pcm = ax.pcolormesh(T, F, power, shading="auto", cmap="jet")

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_title(
            f"Wavelet Scalogram ({title_prefix}): {joint_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.yaxis.set_major_formatter(ScalarFormatter())

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Power", rotation=270, labelpad=15)
        fig.tight_layout()

    def plot_cross_wavelet(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        signal_type: str = "velocity",
        freq_range: tuple[float, float] = (1.0, 50.0),
    ) -> None:
        """Plot Cross Wavelet Transform (XWT) between two signals."""
        try:
            from src.shared.python import signal_processing
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Signal Processing module missing", ha="center", va="center"
            )
            return

        if signal_type == "position":
            times, data = self.data.get_series("joint_positions")
        elif signal_type == "torque":
            times, data = self.data.get_series("joint_torques")
        else:
            times, data = self.data.get_series("joint_velocities")

        data = np.asarray(data)
        if len(times) == 0 or data.ndim < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        if joint_idx_1 >= data.shape[1] or joint_idx_2 >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Joint index out of bounds", ha="center", va="center")
            return

        s1 = data[:, joint_idx_1]
        s2 = data[:, joint_idx_2]
        dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.01
        fs = 1.0 / dt

        try:
            freqs, _, xwt_matrix = signal_processing.compute_xwt(
                s1, s2, fs, freq_range=freq_range
            )
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"XWT Error: {e}", ha="center", va="center")
            return

        power = np.abs(xwt_matrix)
        phase = np.angle(xwt_matrix)

        ax = fig.add_subplot(111)
        T, F = np.meshgrid(times, freqs)
        pcm = ax.pcolormesh(T, F, power, shading="auto", cmap="jet")

        t_skip = max(1, len(times) // 30)
        f_skip = max(1, len(freqs) // 20)

        ax.quiver(
            T[::f_skip, ::t_skip],
            F[::f_skip, ::t_skip],
            np.cos(phase[::f_skip, ::t_skip]),
            np.sin(phase[::f_skip, ::t_skip]),
            units="width",
            pivot="mid",
            width=0.005,
            headwidth=3,
            color="black",
            alpha=0.6,
        )

        name1 = self.data.get_joint_name(joint_idx_1)
        name2 = self.data.get_joint_name(joint_idx_2)
        ax.set_title(
            f"Cross Wavelet: {name1} vs {name2}", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.yaxis.set_major_formatter(ScalarFormatter())

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Cross Power", rotation=270, labelpad=15)
        fig.tight_layout()
