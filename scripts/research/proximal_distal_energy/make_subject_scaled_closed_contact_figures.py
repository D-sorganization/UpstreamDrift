"""Render publication figures for closed-contact feasibility."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
DATA_DIR = ARTICLE_DIR / "data"
FIGURE_DIR = ARTICLE_DIR / "figures"
STEM = "fig_subject_scaled_closed_contact"
COLORS = ("#2C7FB8", "#D95F0E", "#238B45")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "pdf.use14corefonts": True,
            "savefig.bbox": "tight",
        }
    )


def _load() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    record = json.loads(
        (DATA_DIR / "subject_scaled_closed_contact.json").read_text(encoding="utf-8")
    )
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return record, arrays


def _save(fig: Figure) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{STEM}.pdf"
    svg_path = FIGURE_DIR / f"{STEM}.svg"
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return pdf_path, svg_path


def make_figure() -> tuple[Path, Path]:
    """Show closure, limit, collision, and displacement diagnostics."""

    _style()
    record, data = _load()
    profiles = record["design"]["profiles"]  # type: ignore[index]
    labels = [profile["profile"]["profile_id"] for profile in profiles]
    time = data["time_s"]
    spans = data["grip_spans_m"]
    profile_count = len(labels)
    span_count = spans.size
    distances = data["hand_to_grip_distance_m"].reshape(
        profile_count, span_count, time.size, 2
    )
    limit_margin = data["minimum_joint_limit_margin_rad"].reshape(
        profile_count, span_count, time.size
    )
    collision = data["minimum_collision_clearance_m"].reshape(
        profile_count, span_count, time.size
    )
    q = data["solution_q"].reshape(profile_count, span_count, time.size, 20)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax = axes[0, 0]
    for span_index, span in enumerate(spans):
        ax.plot(
            time,
            np.max(distances[:, span_index], axis=(0, 2)) * 1000.0,
            marker="o",
            ms=3,
            color=COLORS[span_index],
            label=f"{span:.2f} m Grip Span",
        )
    ax.axhline(0.5, color="#B2182B", ls="--", label="0.5 mm Gate")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-9, 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Worst Bilateral Closure Error (mm)")
    ax.set_title("A. Closed-Contact Residual (Log Scale)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    x = np.arange(profile_count)
    for span_index, span in enumerate(spans):
        ax.plot(
            x,
            np.min(limit_margin[:, span_index], axis=1),
            marker="o",
            color=COLORS[span_index],
            label=f"{span:.2f} m",
        )
    ax.axhline(0.0, color="#B2182B", ls="--")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Minimum Joint-Limit Margin (rad)")
    ax.set_title("B. Declared Engineering-Limit Margin")

    ax = axes[1, 0]
    for span_index, span in enumerate(spans):
        ax.plot(
            x,
            np.min(collision[:, span_index], axis=1) * 1000.0,
            marker="o",
            color=COLORS[span_index],
            label=f"{span:.2f} m",
        )
    ax.axhline(0.0, color="#B2182B", ls="--")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Minimum Bounding-Sphere Clearance (mm)")
    ax.set_title("C. Coarse Nonadjacent-Body Collision Screen")

    ax = axes[1, 1]
    displacement = np.linalg.norm(np.diff(q[:, 1, :, :14], axis=1), axis=2)
    for profile_index, label in enumerate(labels):
        ax.plot(
            time[1:],
            displacement[profile_index],
            alpha=0.75,
            label=label,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Adjacent-Sample Configuration Change (rad)")
    ax.set_title("D. Continuation-Path Smoothness at 0.18 m")
    ax.legend(frameon=False, ncol=2, fontsize=7)

    fig.suptitle(
        "Subject-Scaled Bilateral Closed-Contact Feasibility",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Reduced-tree inverse kinematics with a fixed club pose. Engineering limits and "
        "bounding spheres are screening guards, not anatomical validation.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    return _save(fig)


def main() -> int:
    """Render the deterministic PDF and SVG outputs."""

    for path in make_figure():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
