"""Energy plotting renderer."""

from __future__ import annotations

from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class EnergyRenderer(BaseRenderer):
    """Renderer for energy analysis plots."""

    def plot_energy_analysis(self, fig: Figure) -> None:
        """Plot kinetic, potential, and total energy over time."""
        times_ke, ke = self.data.get_series("kinetic_energy")
        times_pe, pe = self.data.get_series("potential_energy")
        times_te, te = self.data.get_series("total_energy")

        if len(times_ke) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        ax.plot(
            times_ke,
            ke,
            label="Kinetic Energy",
            linewidth=2.5,
            color=self.colors["primary"],
        )
        ax.plot(
            times_pe,
            pe,
            label="Potential Energy",
            linewidth=2.5,
            color=self.colors["secondary"],
        )
        ax.plot(
            times_te,
            te,
            label="Total Energy",
            linewidth=2.5,
            color=self.colors["quaternary"],
            linestyle="--",
        )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy (J)", fontsize=12, fontweight="bold")
        ax.set_title("Energy Analysis", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
