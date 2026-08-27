"""Render the reviewer-facing phase/event finite-time stability figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REPORT_PATH = ARTICLE / "data/phase_event_stability.json"
ARRAY_PATH = ARTICLE / "data/phase_event_stability.npz"
FIGURE_PATH = ARTICLE / "figures/fig_phase_event_stability.pdf"


def main() -> None:
    """Render finite-time spectra, exponents, and event derivatives."""

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    with np.load(ARRAY_PATH, allow_pickle=False) as arrays:
        time_s = arrays["time_s"]
        singular = arrays["singular_values"]
        exponents = arrays["finite_time_exponents_per_s"]
        direct_event = arrays["direct_event_derivatives_s_per_scaled_state"]

    event_time_s = float(report["reference_event"]["time_s"])
    phase = time_s / event_time_s
    implicit = np.asarray(
        report["event_time_sensitivity"]["implicit"]["derivative_s_per_scaled_state"],
        dtype=float,
    )
    labels = (r"$\theta_s$", r"$\theta_w$", r"$\dot\theta_s$", r"$\dot\theta_w$")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.25), constrained_layout=True)

    axes[0].plot(phase, singular[:, 0], color="#7b2cbf", label="Largest")
    axes[0].plot(phase, singular[:, -1], color="#2a9d8f", label="Smallest")
    axes[0].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Fraction of Delivery-Event Time")
    axes[0].set_ylabel("Scaled Perturbation Gain")
    axes[0].set_title("Finite-Time Singular Gains")
    axes[0].legend(frameon=False)

    axes[1].plot(phase[1:], exponents[1:, 0], color="#e76f51", label="Largest")
    axes[1].plot(phase[1:], exponents[1:, -1], color="#457b9d", label="Smallest")
    axes[1].axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Fraction of Delivery-Event Time")
    axes[1].set_ylabel(r"Finite-Time Exponent [s$^{-1}$]")
    axes[1].set_title("Local Finite-Window Rates")
    axes[1].legend(frameon=False)

    x = np.arange(4)
    direct_mean = np.mean(direct_event, axis=0)
    direct_span = np.max(np.abs(direct_event - direct_mean), axis=0)
    width = 0.37
    axes[2].bar(x - width / 2, implicit, width, color="#264653", label="Implicit")
    axes[2].bar(
        x + width / 2,
        direct_mean,
        width,
        yerr=direct_span,
        capsize=2,
        color="#f4a261",
        label="Direct Rollout",
    )
    axes[2].axhline(0.0, color="#555555", linewidth=0.8)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Event-Time Derivative [s]")
    axes[2].set_title("Transverse-Event Sensitivity")
    axes[2].legend(frameon=False)

    figure.suptitle(
        "Finite-Time Amplification and Event Sensitivity Remain Distinct Estimands",
        fontsize=11,
    )
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
