"""Kinetics plotting renderer."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class KineticsRenderer(BaseRenderer):
    """Renderer for kinetics plots (torques, powers, stiffness)."""

    def plot_joint_torques(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot applied joint torques over time."""
        times, torques = self.data.get_series("joint_torques")

        if len(times) == 0 or len(torques) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        # Ensure torques is a numpy array
        if not isinstance(torques, np.ndarray):
            torques = np.array(torques)

        ax = fig.add_subplot(111)

        if joint_indices is None:
            joint_indices = list(range(torques.shape[1]))

        for idx in joint_indices:
            if idx < torques.shape[1]:
                label = self.data.get_aligned_label(idx, torques.shape[1])
                ax.plot(times, torques[:, idx], label=label, linewidth=2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Torque (Nm)", fontsize=12, fontweight="bold")
        ax.set_title("Applied Joint Torques vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        fig.tight_layout()

    def plot_actuator_powers(self, fig: Figure) -> None:
        """Plot actuator mechanical powers over time."""
        times, powers = self.data.get_series("actuator_powers")

        if len(times) == 0 or len(powers) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        # Ensure powers is a numpy array
        if not isinstance(powers, np.ndarray):
            powers = np.array(powers)

        ax = fig.add_subplot(111)

        for idx in range(powers.shape[1]):
            label = self.data.get_joint_name(idx)
            ax.plot(times, powers[:, idx], label=label, linewidth=2, alpha=0.7)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=12, fontweight="bold")
        ax.set_title("Actuator Powers vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        fig.tight_layout()

    def plot_torque_comparison(self, fig: Figure) -> None:
        """Plot comparison of all joint torques (stacked area or grouped bars)."""
        times, torques = self.data.get_series("joint_torques")
        torques = np.asarray(torques)

        if len(times) == 0 or len(torques) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        torques_pos = np.maximum(torques, 0)
        torques_neg = np.minimum(torques, 0)

        if torques.ndim < 2:
            labels = [self.data.get_joint_name(0)]
        else:
            labels = [self.data.get_joint_name(i) for i in range(torques.shape[1])]

        ax.stackplot(times, torques_pos.T, labels=labels, alpha=0.7)
        ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
        ax.stackplot(times, torques_neg.T, alpha=0.7)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Torque (Nm)", fontsize=12, fontweight="bold")
        ax.set_title("Joint Torque Contributions", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.5)
        fig.tight_layout()

    def plot_work_loop(
        self,
        fig: Figure,
        joint_idx: int = 0,
        title: str | None = None,
    ) -> None:
        """Plot Work Loop (Torque vs Angle) for a joint."""
        times, positions = self.data.get_series("joint_positions")
        _, torques = self.data.get_series("joint_torques")

        positions = np.asarray(positions)
        torques = np.asarray(torques)

        if (
            len(times) == 0
            or positions.ndim < 2
            or joint_idx >= positions.shape[1]
            or torques.ndim < 2
            or joint_idx >= torques.shape[1]
        ):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        angle = np.rad2deg(positions[:, joint_idx])
        torque = torques[:, joint_idx]

        sc = ax.scatter(angle, torque, c=times, cmap="viridis", s=30, alpha=0.6)
        ax.plot(angle, torque, alpha=0.3, color="gray", linewidth=1)
        ax.fill(angle, torque, alpha=0.1, color=self.colors["primary"])

        ax.scatter(
            angle[0],
            torque[0],
            c="green",
            s=100,
            label="Start",
            edgecolor="black",
            zorder=5,
        )
        ax.scatter(
            angle[-1],
            torque[-1],
            c="red",
            s=100,
            marker="s",
            label="End",
            edgecolor="black",
            zorder=5,
        )

        name = self.data.get_joint_name(joint_idx)
        ax.set_xlabel(f"{name} Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{name} Torque (Nm)", fontsize=12, fontweight="bold")
        ax.set_title(title or f"Work Loop: {name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="k", linestyle="-", alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_power_flow(self, fig: Figure) -> None:
        """Plot power flow (stacked bar) over time."""
        times, powers = self.data.get_series("actuator_powers")
        powers = np.asarray(powers)

        if len(times) == 0 or powers.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Power Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        pos_powers = np.maximum(powers, 0)
        neg_powers = np.minimum(powers, 0)

        labels = [self.data.get_joint_name(i) for i in range(powers.shape[1])]

        ax.stackplot(times, pos_powers.T, labels=labels, alpha=0.7)

        from cycler import cycler

        default_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        ax.set_prop_cycle(cycler("color", default_colors))
        ax.stackplot(times, neg_powers.T, alpha=0.7)

        ax.axhline(0, color="k", linewidth=1)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Power Flow (Generation/Absorption)", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
        fig.tight_layout()

    def plot_joint_power_curves(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot joint power curves with generation/absorption regions."""
        times, torques = self.data.get_series("joint_torques")
        _, velocities = self.data.get_series("joint_velocities")

        torques = np.asarray(torques)
        velocities = np.asarray(velocities)

        if len(times) == 0 or torques.size == 0 or velocities.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        if joint_indices is None:
            joint_indices = list(range(min(torques.shape[1], velocities.shape[1])))

        for idx in joint_indices:
            if idx < torques.shape[1] and idx < velocities.shape[1]:
                power = torques[:, idx] * velocities[:, idx]
                label = self.data.get_aligned_label(idx, torques.shape[1])

                line = ax.plot(times, power, label=label, linewidth=2)
                color = line[0].get_color()

                ax.fill_between(
                    times,
                    power,
                    0,
                    where=(power >= 0),
                    alpha=0.2,
                    color=color,
                    interpolate=True,
                )
                ax.fill_between(
                    times,
                    power,
                    0,
                    where=(power < 0),
                    alpha=0.1,
                    color=color,
                    hatch="///",
                    interpolate=True,
                )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Joint Power: Generation (+) vs Absorption (-)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="k", linewidth=1)
        fig.tight_layout()

    def plot_impulse_accumulation(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot cumulative impulse (integrated torque) over time."""
        times, torques = self.data.get_series("joint_torques")
        torques = np.asarray(torques)

        if len(times) == 0 or torques.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        if joint_indices is None:
            joint_indices = list(range(torques.shape[1]))

        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.0

        if dt > 0:
            from scipy.integrate import cumulative_trapezoid

            for idx in joint_indices:
                if idx < torques.shape[1]:
                    impulse = cumulative_trapezoid(torques[:, idx], dx=dt, initial=0)
                    label = self.data.get_aligned_label(idx, torques.shape[1])
                    ax.plot(times, impulse, label=label, linewidth=2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Impulse (Nms)", fontsize=12, fontweight="bold")
        ax.set_title("Angular Impulse Accumulation", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="k", linewidth=1)
        fig.tight_layout()

    def plot_joint_stiffness(
        self,
        fig: Figure,
        joint_idx: int = 0,
        ax: plt.Axes | None = None,
    ) -> None:
        """Plot joint stiffness (moment-angle relationship)."""
        times, positions = self.data.get_series("joint_positions")
        _, torques = self.data.get_series("joint_torques")

        positions = np.asarray(positions)
        torques = np.asarray(torques)

        if ax is None:
            ax = fig.add_subplot(111)

        if (
            len(times) == 0
            or positions.ndim < 2
            or torques.ndim < 2
            or joint_idx >= positions.shape[1]
            or joint_idx >= torques.shape[1]
        ):
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return

        theta = np.rad2deg(positions[:, joint_idx])
        tau = torques[:, joint_idx]

        sc = ax.scatter(theta, tau, c=times, cmap="viridis", s=30, alpha=0.7)
        ax.plot(theta, tau, color="gray", alpha=0.3, linewidth=1)

        if len(theta) > 2:
            from scipy.stats import linregress

            slope, intercept, r_value, _, _ = linregress(theta, tau)
            theta_line = np.array([theta.min(), theta.max()])
            tau_line = slope * theta_line + intercept
            ax.plot(
                theta_line,
                tau_line,
                "r--",
                linewidth=2,
                label=f"K={slope:.2f} Nm/deg, R²={r_value**2:.3f}",
            )
            ax.legend(loc="best")

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_xlabel(f"{joint_name} Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Torque (Nm)", fontsize=12, fontweight="bold")
        ax.set_title(f"Joint Stiffness: {joint_name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_dynamic_stiffness(
        self,
        fig: Figure,
        joint_idx: int = 0,
        window_size: int = 20,
    ) -> None:
        """Plot dynamic (time-varying) stiffness with R² quality metric."""
        times, positions = self.data.get_series("joint_positions")
        _, torques = self.data.get_series("joint_torques")

        positions = np.asarray(positions)
        torques = np.asarray(torques)

        if (
            len(times) < window_size
            or positions.ndim < 2
            or torques.ndim < 2
            or joint_idx >= positions.shape[1]
            or joint_idx >= torques.shape[1]
        ):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return

        theta = np.rad2deg(positions[:, joint_idx])
        tau = torques[:, joint_idx]

        from scipy.stats import linregress

        n_windows = len(theta) - window_size + 1
        t_centers = np.zeros(n_windows)
        k_values = np.zeros(n_windows)
        r2_values = np.zeros(n_windows)

        for i in range(n_windows):
            window_theta = theta[i : i + window_size]
            window_tau = tau[i : i + window_size]
            t_centers[i] = times[i + window_size // 2]

            if np.std(window_theta) > 1e-6:
                slope, _, r_value, _, _ = linregress(window_theta, window_tau)
                k_values[i] = slope
                r2_values[i] = r_value**2
            else:
                k_values[i] = 0.0
                r2_values[i] = 0.0

        ax1 = fig.add_subplot(111)

        line1 = ax1.plot(
            t_centers,
            k_values,
            color=self.colors["primary"],
            linewidth=2,
            label="Stiffness K",
        )
        ax1.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax1.set_ylabel(
            "Stiffness (Nm/deg)",
            fontsize=12,
            fontweight="bold",
            color=self.colors["primary"],
        )
        ax1.tick_params(axis="y", labelcolor=self.colors["primary"])

        ax2 = ax1.twinx()
        line2 = ax2.plot(
            t_centers,
            r2_values,
            color=self.colors["quaternary"],
            linewidth=2,
            linestyle="--",
            label="R² Quality",
        )
        ax2.set_ylabel(
            "R² Quality",
            fontsize=12,
            fontweight="bold",
            color=self.colors["quaternary"],
        )
        ax2.tick_params(axis="y", labelcolor=self.colors["quaternary"])
        ax2.set_ylim(0, 1.05)

        lns = line1 + line2
        labs = [str(ln.get_label()) for ln in lns]
        ax1.legend(lns, labs, loc="best")

        joint_name = self.data.get_joint_name(joint_idx)
        ax1.set_title(
            f"Dynamic Stiffness: {joint_name}", fontsize=14, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        fig.tight_layout()

    def plot_activation_heatmap(
        self,
        fig: Figure,
        data_type: str = "torque",
    ) -> None:
        """Plot activation heatmap (Joints vs Time)."""
        if data_type == "power":
            times, data = self.data.get_series("actuator_powers")
            title = "Actuator Power Activation"
            cbar_label = "Power (W)"
        else:
            times, data = self.data.get_series("joint_torques")
            title = "Joint Torque Activation"
            cbar_label = "Torque (Nm)"

        data = np.asarray(data)

        if len(times) == 0 or data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)
        heatmap_data = data.T
        max_val = np.max(np.abs(heatmap_data))
        if max_val < 1e-6:
            max_val = 1.0

        if len(times) > 1:
            dt = times[1] - times[0]
            time_edges = np.concatenate(
                (
                    [times[0] - dt / 2],
                    times[:-1] + np.diff(times) / 2,
                    [times[-1] + dt / 2],
                )
            )
        else:
            time_edges = np.array([times[0] - 0.5, times[0] + 0.5])

        joint_edges = np.arange(heatmap_data.shape[0] + 1)

        im = ax.pcolormesh(
            time_edges,
            joint_edges,
            heatmap_data,
            cmap="RdBu_r",
            vmin=-max_val,
            vmax=max_val,
            shading="flat",
        )

        ax.set_yticks(np.arange(heatmap_data.shape[0]) + 0.5)
        labels = [self.data.get_joint_name(i) for i in range(heatmap_data.shape[0])]
        ax.set_yticklabels(labels)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(False)

        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()

    def plot_angular_momentum(self, fig: Figure) -> None:
        """Plot Angular Momentum over time (Magnitude and Components)."""
        times, am_data = self.data.get_series("angular_momentum")
        am_data = np.asarray(am_data)

        if len(times) == 0 or am_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Angular Momentum Data", ha="center", va="center")
            return

        am_mag = np.sqrt(np.sum(am_data**2, axis=1))

        ax = fig.add_subplot(111)

        ax.plot(
            times, am_data[:, 0], label="Lx", color=self.colors["secondary"], alpha=0.7
        )
        ax.plot(
            times, am_data[:, 1], label="Ly", color=self.colors["tertiary"], alpha=0.7
        )
        ax.plot(
            times, am_data[:, 2], label="Lz", color=self.colors["quaternary"], alpha=0.7
        )

        ax.plot(
            times,
            am_mag,
            label="Magnitude",
            color=self.colors["primary"],
            linewidth=2.5,
        )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Angular Momentum (kg m²/s)", fontsize=12, fontweight="bold")
        ax.set_title("System Angular Momentum", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_angular_momentum_3d(self, fig: Figure) -> None:
        """Plot 3D trajectory of the Angular Momentum vector."""
        times, am_data = self.data.get_series("angular_momentum")
        am_data = np.asarray(am_data)

        if len(times) == 0 or am_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Angular Momentum Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111, projection="3d")

        lx = am_data[:, 0]
        ly = am_data[:, 1]
        lz = am_data[:, 2]

        sc = ax.scatter(lx, ly, lz, c=times, cmap="viridis", s=20)
        ax.plot(lx, ly, lz, color="gray", alpha=0.3)

        max_idx = np.argmax(np.sum(am_data**2, axis=1))

        ax.plot(
            [0, lx[max_idx]],
            [0, ly[max_idx]],
            [0, lz[max_idx]],
            color="red",
            linewidth=2,
            label="Peak L",
        )

        ax.scatter([0], [0], zs=[0], color="black", s=50, marker="o")

        ax.set_xlabel("Lx (kg m²/s)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Ly (kg m²/s)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Lz (kg m²/s)", fontsize=10, fontweight="bold")
        ax.set_title("3D Angular Momentum Trajectory", fontsize=14, fontweight="bold")
        ax.legend()

        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_induced_acceleration(
        self,
        fig: Figure,
        source_name: str | int,
        joint_idx: int | None = None,
        breakdown_mode: bool = False,
    ) -> None:
        """Plot induced accelerations."""
        ax = fig.add_subplot(111)

        if breakdown_mode:
            self._plot_induced_breakdown(ax, joint_idx if joint_idx is not None else 0)
        else:
            self._plot_induced_single(ax, source_name, joint_idx)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def _plot_induced_breakdown(self, ax: plt.Axes, joint_idx: int) -> None:
        """Plot induced acceleration breakdown for a joint."""
        components = ["gravity", "velocity", "total"]
        linestyles = ["--", "-.", "-"]
        labels = ["Gravity", "Velocity (Coriolis)", "Total (Passive)"]
        colors = [self.colors["secondary"], self.colors["tertiary"], "black"]

        has_data = False
        for comp, ls, lbl, clr in zip(
            components, linestyles, labels, colors, strict=True
        ):
            try:
                times, acc = self.data.get_induced_acceleration_series(comp)
                if len(times) > 0 and acc.size > 0 and joint_idx < acc.shape[1]:
                    ax.plot(
                        times,
                        acc[:, joint_idx],
                        label=lbl,
                        linestyle=ls,
                        color=clr,
                        linewidth=2 if comp == "total" else 1.5,
                    )
                    has_data = True
            except (AttributeError, KeyError):
                continue

        try:
            times_c, acc_c = self.data.get_induced_acceleration_series("control")
            if len(times_c) > 0 and acc_c.size > 0 and joint_idx < acc_c.shape[1]:
                ax.plot(
                    times_c,
                    acc_c[:, joint_idx],
                    label="Control",
                    linestyle=":",
                    color=self.colors["quaternary"],
                    linewidth=1.5,
                )
        except (AttributeError, KeyError):
            pass

        if not has_data:
            ax.text(
                0.5,
                0.5,
                "No induced acceleration breakdown data",
                ha="center",
                va="center",
            )
            return

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_title(
            f"Induced Accelerations Breakdown: {joint_name}",
            fontsize=14,
            fontweight="bold",
        )

    def _plot_induced_single(
        self, ax: plt.Axes, source_name: str | int, joint_idx: int | None
    ) -> None:
        """Plot induced acceleration for a single source."""
        try:
            times, acc = self.data.get_induced_acceleration_series(source_name)
        except (AttributeError, KeyError):
            ax.text(
                0.5,
                0.5,
                f"No induced acceleration data for {source_name}",
                ha="center",
                va="center",
            )
            return

        if len(times) == 0 or acc.size == 0:
            ax.text(0.5, 0.5, f"No data for {source_name}", ha="center", va="center")
            return

        if joint_idx is not None:
            if joint_idx < acc.shape[1]:
                ax.plot(
                    times,
                    acc[:, joint_idx],
                    label=self.data.get_joint_name(joint_idx),
                    linewidth=2,
                    color=self.colors["primary"],
                )
                ax.set_ylabel(
                    f"Joint {joint_idx} Acceleration (rad/s²)",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Joint index {joint_idx} out of bounds",
                    ha="center",
                    va="center",
                )
                return
        else:
            norm = np.sqrt(np.sum(acc**2, axis=1))
            ax.plot(
                times,
                norm,
                label="L2 Norm",
                linewidth=2,
                color=self.colors["primary"],
            )
            ax.set_ylabel(
                "Acceleration Magnitude (rad/s²)", fontsize=12, fontweight="bold"
            )

        ax.set_title(
            f"Induced Acceleration: {source_name}", fontsize=14, fontweight="bold"
        )
