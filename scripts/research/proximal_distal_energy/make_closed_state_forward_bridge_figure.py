"""Render the closed-state forward-contact bridge falsification summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
STEM = STUDY_DIR / "figures/fig_closed_state_forward_bridge"


def main() -> int:
    """Write one vector-first four-panel scientific summary."""

    record = json.loads(
        (STUDY_DIR / "data/closed_state_forward_bridge.json").read_text(
            encoding="utf-8"
        )
    )
    with np.load(STUDY_DIR / "data/closed_state_forward_bridge.npz") as archive:
        position = archive["position_closure_error_m"] * 1.0e6
        velocity = archive["velocity_closure_error_m_s"] * 1.0e3
    cases = record["forward_subset"]["cases"]
    x = np.arange(len(cases))
    position_max = [
        case["observed_metrics"]["club_position_max_m"] * 1.0e6 for case in cases
    ]
    wrench = [
        case["observed_metrics"]["contact_wrench_relative_rms"] * 100.0
        for case in cases
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), constrained_layout=True)
    images = (
        (axes[0, 0], position.max(axis=2), "Position Closure Error", "Error (µm)"),
        (axes[0, 1], velocity.max(axis=2), "Velocity Closure Error", "Error (mm/s)"),
    )
    for axis, values, title, label in images:
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Phase Sample")
        axis.set_ylabel("Profile–Span Case")
        fig.colorbar(image, ax=axis, label=label)
    axes[1, 0].plot(x, position_max, color="#1f77b4", linewidth=1.6)
    axes[1, 0].set(
        title="Cross-Engine Club-Position Difference",
        xlabel="Spanning Case",
        ylabel="Maximum Difference (µm)",
    )
    axes[1, 1].plot(x, wrench, color="#d95f02", linewidth=1.6)
    axes[1, 1].set(
        title="Cross-Engine Contact-Wrench Difference",
        xlabel="Spanning Case",
        ylabel="Relative RMS (%)",
    )
    for axis in axes[1]:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Closed Subject States Enter Independent Forward Solvers Without Preload",
        fontsize=15,
    )
    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        STEM.with_suffix(".pdf"),
        metadata={"Title": "Closed-State Forward-Contact Bridge"},
    )
    fig.savefig(
        STEM.with_suffix(".svg"),
        metadata={"Title": "Closed-State Forward-Contact Bridge"},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
