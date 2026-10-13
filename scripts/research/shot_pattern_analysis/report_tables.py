"""Compact publication tables with adjacent, lossless source-identity keys."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.tools.shot_pattern_analysis.matched_formatting import format_sg_interval_record


def resolved_selection_lines(
    cells: list[dict], *, club_id: str | None = None
) -> list[str]:
    """Keep precise intervals in wide columns, and populations in their own table."""
    indexed_cells = list(enumerate(cells, 1))
    if club_id is not None:
        indexed_cells = [
            (index, cell)
            for index, cell in indexed_cells
            if cell["config"]["club"]["club_id"] == club_id
        ]
        if not indexed_cells:
            raise ValueError(f"no resolved cells for club {club_id!r}")
    lines = [
        "### Resolved Zero-Lean Curve and Variability Comparison",
        "",
        "This resolved selection retains original primary accounting and explicitly "
        "identifies the four PW strong-curve face-bounds supplements. Each condition "
        "key below maps to one exact cell and protocol. Positive paired SG favors "
        "the named curve over Straight; intervals describe Monte Carlo sampling. "
        "SG is per identical hypothetical opportunity, not per round.",
        "",
        "**Condition Key**",
        "",
    ]
    for index, cell in indexed_cells:
        cfg = cell["config"]
        lines += [
            f"**C{index:02d}**: `{cell['cell']}`; protocol `{cell['source_protocol']}`. "
            f"Shaft twist SD {cfg['noise']['standard_deviations'][0]:g}°; "
            f"curve ×{cfg['curve_scale']:g}.",
            "",
        ]
    lines += [
        "**Paired Course Scoring**",
        "",
        "| Condition | Common-Three Scored / Attempted | Draw Δ SG [95%] | Fade Δ SG [95%] |",
        "| --- | ---: | --- | --- |",
    ]
    for index, cell in indexed_cells:
        score = cell["course_scoring"]
        sg = score["paired_benefit_vs_straight"]
        lines.append(
            f"| C{index:02d} | {score['paired_n']:,} / {score['total_n']:,} | "
            f"{format_sg_interval_record(sg['Draw'])} | {format_sg_interval_record(sg['Fade'])} |"
        )
    lines += [
        "",
        "**Individual Supported Endpoints**",
        "",
        "These individual populations can exceed the common-three scored intersection. "
        "Each entry is supported / attempted; no unsupported shots are retried.",
        "",
        "| Condition | Straight | Draw | Fade |",
        "| --- | ---: | ---: | ---: |",
    ]
    for index, cell in indexed_cells:
        counts = [
            f"{cell['summary'][name]['supported_n']:,} / {cell['summary'][name]['total_n']:,}"
            for name in ("Straight", "Draw", "Fade")
        ]
        lines.append(f"| C{index:02d} | " + " | ".join(counts) + " |")
    return lines + [""]


def keyed_effect_table(effects: Iterable[Mapping[str, Any]]) -> str:
    """Retain ordered protocols/cells outside the narrow numerical columns."""
    selected = list(effects)
    lines = ["**Ordered Contrast Key**", ""]
    for index, effect in enumerate(selected, 1):
        sources = list(
            zip(effect["source_protocols"], effect["source_cells"], strict=True)
        )
        if len(sources) != 2:
            raise ValueError(
                "paired table requires exactly two ordered source identities"
            )
        lines += [f"**E{index:02d}** — {effect['comparison_kind']}: ", ""]
        for label, (protocol, cell) in zip(("A", "B"), sources, strict=True):
            lines += [f"{label}: `{protocol}`; `{cell}`.", ""]
    lines += [
        "| Contrast | Curved Pattern | B − A Relative SG [95% MC] | Common Six / Attempted |",
        "| --- | --- | ---: | ---: |",
    ]
    for index, effect in enumerate(selected, 1):
        for name, key in (
            ("Draw", "draw_minus_straight_change"),
            ("Fade", "fade_minus_straight_change"),
        ):
            value = effect[key]
            interval = format_sg_interval_record(value)
            lines.append(
                f"| E{index:02d} | {name} | {interval} | "
                f"{effect['common_six_n']:,} / {effect['total_n_per_population']:,} |"
            )
    lines += [
        "",
        "B − A is the change in (curved − Straight) SG on the same six-population "
        "supported-index intersection, with A and B identified above. It is not a causal "
        "distance contribution. Intervals are pointwise Monte Carlo sampling intervals; "
        "shared seeded errors correlate cells and do not validate player performance.",
    ]
    return "\n".join(lines)
