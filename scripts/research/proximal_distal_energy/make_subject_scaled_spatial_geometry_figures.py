"""Render publication figures for the subject-scaled contact-geometry atlas."""

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
STEM = "fig_subject_scaled_spatial_geometry"
COLORS = ("#2C7FB8", "#D95F0E", "#238B45", "#6A51A3", "#B2182B")


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
        (DATA_DIR / "subject_scaled_spatial_geometry.json").read_text(encoding="utf-8")
    )
    with np.load(DATA_DIR / "subject_scaled_spatial_geometry.npz") as archive:
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
    """Show closure, conditioning, grip-span scaling, and load pathways."""

    _style()
    record, data = _load()
    profiles = record["design"]["profiles"]  # type: ignore[index]
    labels = [profile["profile"]["profile_id"] for profile in profiles]
    time = data["time_s"]
    spans = data["grip_spans_m"]
    distances = data["hand_to_grip_distance_m"].reshape(6, 3, time.size, 2)
    condition = data["constraint_jacobian_condition_number"].reshape(6, 3, time.size)
    couple = data["force_generated_couple_nm"].reshape(6, 3, time.size)
    regional = data["regional_generalized_load_norm"].reshape(6, 3, time.size, 5)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    ax = axes[0, 0]
    closure_median = np.median(distances, axis=(1, 2, 3)) * 1000.0
    closure_min = np.min(distances, axis=(1, 2, 3)) * 1000.0
    closure_max = np.max(distances, axis=(1, 2, 3)) * 1000.0
    x = np.arange(len(labels))
    ax.errorbar(
        x,
        closure_median,
        yerr=(closure_median - closure_min, closure_max - closure_median),
        fmt="o",
        color=COLORS[0],
        capsize=3,
    )
    ax.axhline(5.0, color=COLORS[4], ls="--", label="5 mm Closure Tolerance")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Anatomical Hand-to-Grip Distance (mm)")
    ax.set_title("A. Prescribed States Do Not Close the Contacts")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    for span_index, span in enumerate(spans):
        ax.plot(
            time,
            np.median(condition[:, span_index], axis=0),
            color=COLORS[span_index],
            label=f"{span:.2f} m Grip Span",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Constraint Jacobian Condition Number")
    ax.set_title("B. Local Rank Does Not Establish Contact Feasibility")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    sample_index = int(np.argmax(np.median(np.abs(couple), axis=(0, 1))))
    peak = np.median(np.abs(couple[:, :, sample_index]), axis=0)
    ax.plot(spans, peak, "o-", color=COLORS[1], label="Prescribed Force Couple")
    slope = float(np.dot(spans, peak) / np.dot(spans, spans))
    ax.plot(
        spans, slope * spans, ls="--", color="#657786", label="Linear Through Origin"
    )
    ax.set_xlabel("Grip Span (m)")
    ax.set_ylabel("Force-Generated Couple Magnitude (N m)")
    ax.set_title("C. Grip Span Scales the Force Couple Linearly")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    region_names = ("Pelvis", "Torso", "Lead Arm", "Trail Arm", "Club")
    load_p95 = np.percentile(regional, 95.0, axis=(0, 1, 2))
    ax.bar(region_names, load_p95, color=COLORS)
    ax.set_ylabel("95th-Percentile Generalized Load Norm (N m)")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("D. Prescribed Contact Loads Reach Multiple Regions")

    fig.suptitle(
        "Subject-Scaled Spatial Contact-Geometry Audit",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Six deterministic de Leva design profiles; these are not human participants. "
        "Full local rank at an open contact state is not evidence of anatomical feasibility.",
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
