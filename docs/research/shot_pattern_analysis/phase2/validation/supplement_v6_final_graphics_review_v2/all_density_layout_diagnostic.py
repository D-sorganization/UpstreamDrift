import datetime
import hashlib
import importlib.util
import json
import sys
import pathlib
import re

ROOT = pathlib.Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2"
)
BASE = ROOT / "validation/supplement_v6_final_graphics_review_v2"
FIGURES = ROOT / "supplement_report_v1/figures/comparisons"
SOURCE = BASE / "curve_strength_144_layout_diagnostic.py"
OUTPUT = BASE / "all_density_layout_diagnostic.json"

spec = importlib.util.spec_from_file_location("layout_diagnostic", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load layout diagnostic: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = sorted(FIGURES.glob("*_density.svg"))
    if len(files) != 384:
        raise ValueError(f"expected 384 density SVGs, found {len(files)}")
    results = []
    for path in files:
        svg = path.read_text()
        collisions = []
        titles = []
        for label in ("Straight", "Draw", "Fade"):
            match = re.search(r"<!-- (" + label + r" · [^\n]*?) -->", svg)
            if match is None:
                raise ValueError(f"missing first-row title in {path}: {label}")
            titles.append(match.group(1))
        legends = [
            "50% Simulated-Sample HDR",
            "80% Simulated-Sample HDR",
            "95% Simulated-Sample HDR",
            "Common Target",
        ]
        for title in titles:
            p1, b1, a1 = module.glyph(title, svg)
            for legend in legends:
                p2, b2, a2 = module.glyph(legend, svg)
                intersection = [
                    max(b1[0], b2[0]),
                    max(b1[1], b2[1]),
                    min(b1[2], b2[2]),
                    min(b1[3], b2[3]),
                ]
                if (
                    intersection[0] < intersection[2]
                    and intersection[1] < intersection[3]
                    and p1.intersects_path(p2, filled=True)
                ):
                    collisions.append(
                        {
                            "panel_title": title,
                            "legend_label": legend,
                            "panel_title_anchor_pt": a1,
                            "legend_anchor_pt": a2,
                            "panel_title_glyph_bounds_pt": b1,
                            "legend_glyph_bounds_pt": b2,
                            "bounding_box_intersection_pt": intersection,
                        }
                    )
        if collisions:
            results.append(
                {
                    "svg_path": str(path),
                    "svg_sha256": sha(path),
                    "matching_png_path": str(path.with_suffix(".png")),
                    "matching_pdf_path": str(path.with_suffix(".pdf")),
                    "collisions": collisions,
                }
            )
    result = {
        "schema": "supplement-all-density-svg-title-legend-clearance/1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": "/tmp/shot-pattern-validation-venv-20261009/bin/python "
        + str(pathlib.Path(__file__).resolve()),
        "method": "All 384 saved density SVGs; exact saved text anchors with Matplotlib DejaVu Sans 10 pt TextPath outline-intersection calculation.",
        "svg_count": len(files),
        "collision_svg_count": len(results),
        "source_script_path": str(SOURCE),
        "source_script_sha256": sha(SOURCE),
        "collisions": results,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha(OUTPUT),
                "svg_count": len(files),
                "collision_svg_count": len(results),
                "paths": [r["svg_path"] for r in results],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
