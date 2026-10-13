"""Small helpers for the saved-data footer review runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict, cast
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from src.tools.shot_pattern_analysis import matched_supplement_figures as figures
from src.tools.shot_pattern_analysis.matched_pair_social import (
    PairSocialContext,
    write_pair_social_density,
)
from src.tools.shot_pattern_analysis.matched_report import load_validated_cell


class FigureGeometry(TypedDict):
    path: str
    issues: list[str]
    minimum_font_em_px_at_1080: float
    minimum_footer_clearance_pt: float


class FooterPlan(TypedDict):
    cells: dict[str, dict[str, object]]
    execution: dict[str, object]


def inspect_figure(figure: Figure, path: str | Path) -> FigureGeometry:
    """Return the existing decorated-text clearance diagnostics for one PNG."""
    figure.canvas.draw()
    canvas = cast(FigureCanvasAgg, figure.canvas)
    renderer = canvas.get_renderer()
    footer = figure.texts[-1].get_window_extent(renderer)
    issues = []
    min_clearance = float("inf")
    for axis in figure.axes[:-1]:
        decoration = axis.get_tightbbox(renderer)
        assert decoration is not None
        min_clearance = min(min_clearance, decoration.y0 - footer.y1)
        if decoration.overlaps(footer):
            issues.append("footer overlaps decorated axes")
        for legend in figure.legends:
            if legend.get_window_extent(renderer).overlaps(
                axis.title.get_window_extent(renderer)
            ):
                issues.append("legend overlaps panel title")
    for panel in range(2):
        main_axis, right_axis = figure.axes[3 * panel + 1 : 3 * panel + 3]
        for left in main_axis.get_xticklabels():
            for right in right_axis.get_xticklabels():
                if (
                    left.get_visible()
                    and right.get_visible()
                    and left.get_text()
                    and right.get_text()
                    and left.get_window_extent(renderer).overlaps(
                        right.get_window_extent(renderer)
                    )
                ):
                    issues.append("main and marginal tick labels overlap")
    font_pixels = [
        cast(float, text.get_fontsize()) * 1080 / (72 * figure.get_figwidth())
        for text in figure.findobj(Text)
        if text.get_visible() and text.get_text().strip()
    ]
    if min(font_pixels) < 11:
        issues.append("per-line font size below 11 em-pixels at 1080")
    for text in figure.texts:
        box = text.get_window_extent(renderer)
        if (
            box.x0 < 0
            or box.y0 < 0
            or box.x1 > figure.bbox.width
            or box.y1 > figure.bbox.height
        ):
            issues.append("figure caption outside canvas")
    return {
        "path": str(path),
        "issues": issues,
        "minimum_font_em_px_at_1080": min(font_pixels),
        "minimum_footer_clearance_pt": min_clearance * 72 / figure.dpi,
    }


def render_club(
    club: str,
    data: Path,
    output: Path,
    plan: FooterPlan,
    inputs: dict[str, dict[str, str]],
    artifacts: dict[str, str],
    previews: dict[str, str],
    sha: Callable[[Path], str],
) -> None:
    """Render one club's pair and record original and social preview hashes."""
    refs, loaded = [], []
    for sd in (1, 2):
        name = f"{club}_lean0_curve1_rotation{sd}_combined2"
        directory = data / name
        inputs[name] = {
            part: sha(directory / part)
            for part in ("manifest.json", "summary.json", "shots.csv.gz")
        }
        summary = load_validated_cell(directory, plan["cells"][name], plan["execution"])
        refs.append(
            {
                "cell": name,
                "source_root": str(data),
                "source_protocol": "original_primary/1",
                "status": "completed",
                "config": summary["config"],
                "summary": summary,
            }
        )
        loaded.append(
            figures._supported_endpoints(directory / "shots.csv.gz", 10000)[0]
        )
    common = sorted(
        set.intersection(
            *(
                {index for index, _, _ in endpoints[pattern]}
                for endpoints in loaded
                for pattern in ("Straight", "Draw", "Fade")
            )
        )
    )
    labels = tuple(
        figures._pair_condition_label("shaft_rotation_sd", reference)
        for reference in refs
    )
    points = {}
    for label, endpoints in zip(labels, loaded, strict=True):
        for pattern, rows in endpoints.items():
            by_index = {index: (x, y) for index, x, y in rows}
            points[f"{pattern} · {label}"] = [by_index[index] for index in common]
    context = PairSocialContext(
        condition_labels=(labels[0], labels[1]),
        scenario=figures._pair_social_scenario("shaft_rotation_sd", *refs),
        social_club=club.replace("_", " ").title(),
        target_point_m=(refs[0]["summary"]["target_x_m"], 0),
        total_n=10000,
        delivery_noise=figures._noise_profile_description(
            "combined2", refs[0]["config"]["noise"]
        ),
    )
    paths = write_pair_social_density(points, output / club, context=context)
    for path in paths.values():
        artifacts[str(path.relative_to(output))] = sha(path)
        if path.suffix == ".png":
            with Image.open(path) as image:
                preview = image.resize(
                    (1080, round(image.height * 1080 / image.width)),
                    Image.Resampling.LANCZOS,
                )
                preview_path = path.with_name(path.stem + "_1080px.png")
                preview.save(preview_path)
            previews[str(preview_path.relative_to(output))] = sha(preview_path)
