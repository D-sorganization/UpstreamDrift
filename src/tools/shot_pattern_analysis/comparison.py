"""Compare matched saved experiments without recomputing their physics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def validate_comparison(summaries: list[dict[str, Any]]) -> None:
    """Require common random pairing and target before comparing experiments."""
    if not summaries:
        raise ValueError("At least one summary is required")
    reference = summaries[0]
    for summary in summaries:
        for key in ("seed", "n_shots"):
            if summary["config"][key] != reference["config"][key]:
                raise ValueError(f"Experiments must be matched on {key}")
        if not math.isclose(
            summary["target_x_m"], reference["target_x_m"], abs_tol=1e-6
        ):
            raise ValueError("Experiments must share a matched target")
        if set(summary["patterns"]) != {"Straight", "Draw", "Fade"}:
            raise ValueError("All three shot patterns are required")


def export_comparison(bundles: list[Path], output: Path) -> Path:
    """Create a shareable overview from validated, paired result bundles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summaries = [json.loads((p / "summary.json").read_text()) for p in bundles]
    validate_comparison(summaries)
    labels = [
        f"Face SD {s['config']['face_sd_deg']:g}°\nCurve {s['config'].get('curve_scale', 1):g}×"
        for s in summaries
    ]
    navy, foreground = "#101c30", "#eff5fc"
    with plt.rc_context(
        {
            "figure.facecolor": navy,
            "axes.facecolor": navy,
            "text.color": foreground,
            "axes.labelcolor": foreground,
            "xtick.color": foreground,
            "ytick.color": foreground,
            "font.size": 13,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), dpi=100)
        measures = [
            ("aimed_lateral_sd_m", "Lateral Standard Deviation", "m", 1),
            ("aimed_target_hit_fraction", "Landings Within 15 m", "%", 100),
            ("mean_carry_m", "Mean Carry", "m", 1),
        ]
        x = np.arange(len(summaries))
        for ax, (key, title, unit, factor) in zip(axes, measures, strict=True):
            for index, (pattern, color) in enumerate(
                zip(
                    ("Straight", "Draw", "Fade"),
                    ("#64d9ff", "#ffbf69", "#d69aff"),
                    strict=True,
                )
            ):
                values = [s["patterns"][pattern][key] * factor for s in summaries]
                ax.bar(
                    x + (index - 1) * 0.24,
                    values,
                    width=0.23,
                    color=color,
                    label=pattern,
                )
            ax.set(title=title, ylabel=unit, xticks=x, xticklabels=labels)
            ax.grid(axis="y", alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
            if key == "mean_carry_m":
                ax.set_ylim(
                    min(s["patterns"][p][key] for s in summaries for p in s["patterns"])
                    - 1,
                    max(s["patterns"][p][key] for s in summaries for p in s["patterns"])
                    + 0.5,
                )
        axes[0].legend(frameon=False)
        fig.suptitle("Does More Curve Improve Shot Outcomes?", fontsize=29, y=0.92)
        fig.text(
            0.5,
            0.84,
            "10,000 Shots per Pattern • Paired Face Errors • Fixed Path • Nominal Aiming",
            ha="center",
        )
        fig.text(
            0.5,
            0.09,
            "1× Draw: Face +1.5°, Path +3°   |   2× Draw: Face +3°, Path +6°   |   Fade Mirrored",
            ha="center",
        )
        fig.text(
            0.5,
            0.045,
            "Conditional Model Study • Carry Only • Simplified Impact Model • No Player Validation",
            ha="center",
            fontsize=12,
        )
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.23, top=0.74, wspace=0.35)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)
    return output
