import sys
import json
from pathlib import Path
from typing import Any, TypedDict

from scripts.research.shot_pattern_analysis.build_final_figures import _supported_points
from src.tools.shot_pattern_analysis import academic_plots as p

root = Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift"
    "/docs/research/shot_pattern_analysis/phase2/final_v1"
    "/driver_lean0_curve1_rotation1_combined2"
)
points, total = _supported_points(root)


class TickGapRow(TypedDict):
    axis: int
    labels: list[str]
    gap_points: float
    axis_width_points: float


class CaptureResult(TypedDict):
    presentation: str
    tick_gaps: list[TickGapRow]


results: list[CaptureResult] = []


def capture(figure: Any, output_stem: Path, *, dpi: int) -> dict[str, Path]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    rows: list[TickGapRow] = []
    for i, ax in enumerate(figure.axes):
        ticks = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        ticks.sort(key=lambda t: t.get_window_extent(renderer).x0)
        for a, b in zip(ticks, ticks[1:], strict=False):
            gap = b.get_window_extent(renderer).x0 - a.get_window_extent(renderer).x1
            rows.append(
                {
                    "axis": i,
                    "labels": [a.get_text(), b.get_text()],
                    "gap_points": gap * 72 / figure.dpi,
                    "axis_width_points": ax.get_window_extent(renderer).width
                    * 72
                    / figure.dpi,
                }
            )
    results.append({"presentation": str(output_stem), "tick_gaps": rows})
    return {}


p._save_formats = capture
for mode in ("academic", "social"):
    p.write_academic_dispersion_figure(
        points,
        mode,
        scenario="Driver · Lean 0° · Curve ×1 · Shaft Twist SD 1°",
        metadata={
            "social_club": "Driver",
            "valid_shots_by_pattern": {k: len(v) for k, v in points.items()},
            "total_shots_by_pattern": total,
        },
        target_point_m=(178.94875782337786, 0),
        presentation=mode,
        dpi=180,
    )
Path("/tmp/p2-marginal-gutter-review.json").write_text(json.dumps(results, indent=2))  # nosec B108
sys.stdout.write(
    str(
        json.dumps(
            [
                {
                    "presentation": r["presentation"],
                    "right_marginals": [
                        x for x in r["tick_gaps"] if x["axis"] in (2, 5, 8)
                    ],
                }
                for r in results
            ],
            indent=2,
        )
    )
    + "\n"
)
