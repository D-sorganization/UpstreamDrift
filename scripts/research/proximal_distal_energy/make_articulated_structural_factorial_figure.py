"""Render reviewer-facing structural-factorial effects and support."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCHEMA = "articulated-structural-factorial-summary/1.2.0"
OUTCOME_LABELS = {
    "final_club_translation_speed_m_s": "Club Translation Speed Effect (m/s)",
    "club_linear_momentum_change_kg_m_s": "Club Momentum-Change Effect (kg m/s)",
    "signed_contact_impulse_n_s": "Signed Contact-Impulse Effect (N s)",
    "signed_contact_work_j": "Signed Contact-Work Effect (J)",
    "terminal_total_dissipation_j": "Terminal Dissipation Effect (J)",
}
SIGN_COLORS = {"negative": "#3b6fb6", "zero": "#b8b8b8", "positive": "#c84a4a"}


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_summary(path: Path) -> dict[str, Any]:
    """Load only the support-complete structural summary schema."""

    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("schema_version") != SCHEMA:
        raise ValueError("unsupported structural-factorial summary schema")
    if not isinstance(summary.get("contrast_aggregates"), list):
        raise ValueError("structural summary lacks contrast aggregates")
    if not isinstance(summary.get("factorial_contrasts"), list):
        raise ValueError("structural summary lacks paired-block contrasts")
    return summary


def _primary_aggregates(
    summary: Mapping[str, Any], outcome: str
) -> list[Mapping[str, Any]]:
    records = [
        _mapping(row, name="contrast aggregate")
        for row in summary["contrast_aggregates"]
        if _mapping(row, name="contrast aggregate").get("estimand_class") == "primary"
        and _mapping(row, name="contrast aggregate").get("outcome") == outcome
    ]
    if not records:
        raise ValueError(f"summary contains no primary contrasts for {outcome}")
    return records


def _short_label(contrast_id: str) -> str:
    return (
        contrast_id.replace("ground_", "G: ")
        .replace("shaft_", "S: ")
        .replace("x", " × ")
    )


def _effect_panel(
    axis: plt.Axes, aggregates: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> None:
    for index, row in enumerate(aggregates):
        effect = _mapping(row["high_minus_low_effect"], name="effect range")
        median = effect.get("median")
        if median is None:
            axis.scatter(0.0, index, marker="x", color="#777777")
            axis.text(0.0, index, "  Missing Support", va="center", fontsize=8)
            continue
        low, high = float(effect["minimum"]), float(effect["maximum"])
        center = float(median)
        axis.errorbar(
            center,
            index,
            xerr=np.asarray([[center - low], [high - center]]),
            fmt="o",
            color="#1f4e79",
            capsize=3,
        )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Median and Full Block Range")
    axis.set_title("A. High-Minus-Low Effects")
    axis.grid(axis="x", alpha=0.25)


def _sign_panel(
    axis: plt.Axes, aggregates: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> None:
    left = np.zeros(len(aggregates), dtype=float)
    for sign in ("negative", "zero", "positive"):
        fractions = []
        for row in aggregates:
            eligible = int(row["eligible_block_count"])
            counts = _mapping(row["exact_sign_counts"], name="sign counts")
            fractions.append(float(counts[sign]) / eligible if eligible else 0.0)
        values = np.asarray(fractions)
        axis.barh(
            labels, values, left=left, color=SIGN_COLORS[sign], label=sign.title()
        )
        left += values
    missing = [int(row["eligible_block_count"]) == 0 for row in aggregates]
    for index, is_missing in enumerate(missing):
        if is_missing:
            axis.barh(index, 1.0, color="#eeeeee", edgecolor="#777777", hatch="//")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Fraction of Eligible Blocks")
    axis.set_title("B. Exact Sign Distribution")
    axis.legend(fontsize=8, ncol=3)


def _support_panel(
    axis: plt.Axes, aggregates: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> None:
    support = np.asarray([float(row["support_fraction"]) for row in aggregates])
    axis.barh(labels, support, color="#487a55")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Eligible / Registered Blocks")
    axis.set_title("C. Common-Support Adequacy")
    for index, (value, row) in enumerate(zip(support, aggregates, strict=True)):
        axis.text(
            min(value + 0.01, 0.97),
            index,
            f"{row['eligible_block_count']}/{row['expected_block_count']}",
            va="center",
            fontsize=8,
        )


def _block_heatmap(
    axis: plt.Axes,
    summary: Mapping[str, Any],
    aggregates: Sequence[Mapping[str, Any]],
    outcome: str,
    labels: Sequence[str],
) -> None:
    contrast_ids = [str(row["contrast_id"]) for row in aggregates]
    rows = [
        _mapping(row, name="factorial contrast")
        for row in summary["factorial_contrasts"]
        if _mapping(row, name="factorial contrast").get("outcome") == outcome
        and _mapping(row, name="factorial contrast").get("contrast_id") in contrast_ids
    ]
    blocks = sorted({tuple(row["block"]) for row in rows}, key=str)
    matrix = np.full((len(contrast_ids), max(1, len(blocks))), np.nan)
    block_index = {block: index for index, block in enumerate(blocks)}
    contrast_index = {name: index for index, name in enumerate(contrast_ids)}
    for row in rows:
        matrix[
            contrast_index[str(row["contrast_id"])], block_index[tuple(row["block"])]
        ] = float(row["high_minus_low_effect"])
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(np.max(finite)) if finite.size else 0.0, np.finfo(float).eps)
    color_map = plt.get_cmap("coolwarm").copy()
    color_map.set_bad("#e6e6e6")
    image = axis.imshow(matrix, aspect="auto", cmap=color_map, vmin=-limit, vmax=limit)
    axis.invert_yaxis()
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Registered Paired Blocks (Stable Order)")
    axis.set_title("D. All-State Effect Matrix; Gray Is Missing")
    axis.set_xticks([])
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label(OUTCOME_LABELS[outcome])


def render_figure(summary: Mapping[str, Any], *, outcome: str, output: Path) -> None:
    """Render PDF and SVG without combining outcomes that have unlike units."""

    if outcome not in OUTCOME_LABELS:
        raise ValueError(f"unsupported structural outcome: {outcome}")
    if summary.get("schema_version") != SCHEMA:
        raise ValueError("unsupported structural-factorial summary schema")
    aggregates = _primary_aggregates(summary, outcome)
    labels = [_short_label(str(row["contrast_id"])) for row in aggregates]
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True)
    _effect_panel(axes[0, 0], aggregates, labels)
    _sign_panel(axes[0, 1], aggregates, labels)
    _support_panel(axes[1, 0], aggregates, labels)
    _block_heatmap(axes[1, 1], summary, aggregates, outcome, labels)
    figure.suptitle(
        "Structural Pathway Factorial: Effects, Direction, and Support\n"
        + OUTCOME_LABELS[outcome],
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = output.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def main() -> None:
    """Render one registered outcome from a completed summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outcome", choices=tuple(OUTCOME_LABELS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_figure(load_summary(args.summary), outcome=args.outcome, output=args.output)


if __name__ == "__main__":
    main()


__all__ = ["load_summary", "render_figure"]
