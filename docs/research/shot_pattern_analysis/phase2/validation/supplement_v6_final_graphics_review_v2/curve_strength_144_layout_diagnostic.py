import datetime
import hashlib
import json
import pathlib
import re
import sys

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

ROOT = pathlib.Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2"
)
BATCH = ROOT / "validation/supplement_v6_final_graphics_review_v2/curve_strength_144"
REPORT = ROOT / "supplement_report_v1"
OUTPUT = (
    ROOT
    / "validation/supplement_v6_final_graphics_review_v2/curve_strength_144_layout_diagnostic.json"
)
INVENTORY = BATCH / "batch_inventory.json"
PAIRS = [
    ("Straight · Curve ×1", "50% Simulated-Sample HDR"),
    ("Draw · Curve ×1", "80% Simulated-Sample HDR"),
    ("Fade · Curve ×1", "Common Target"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def glyph(text, svg):
    match = re.search(
        r"<!-- "
        + re.escape(text)
        + r' -->\s*<g transform="translate\(([\d.]+) ([\d.]+)\) scale\(0\.1 -0\.1\)"',
        svg,
    )
    if match is None:
        raise ValueError(f"missing anchor: {text}")
    x, y = (float(value) for value in match.groups())
    path = TextPath(
        (0, 0), text, prop=FontProperties(family="DejaVu Sans", size=10)
    ).transformed(Affine2D().scale(1, -1).translate(x, y))
    bounds = path.get_extents()
    return (
        path,
        [float(bounds.x0), float(bounds.y0), float(bounds.x1), float(bounds.y1)],
        [x, y],
    )


def main():
    inventory = json.loads(INVENTORY.read_text())
    analyses = []
    for selected in inventory["selected"]:
        rel = next(
            path for path in selected["asset_sha256"] if path.endswith("density.svg")
        )
        path = REPORT / rel
        assert sha(path) == selected["asset_sha256"][rel]
        svg = path.read_text()
        tests = []
        for title, legend in PAIRS:
            p1, b1, a1 = glyph(title, svg)
            p2, b2, a2 = glyph(legend, svg)
            intersection = [
                max(b1[0], b2[0]),
                max(b1[1], b2[1]),
                min(b1[2], b2[2]),
                min(b1[3], b2[3]),
            ]
            tests.append(
                {
                    "panel_title": title,
                    "legend_label": legend,
                    "panel_title_anchor_pt": a1,
                    "legend_anchor_pt": a2,
                    "panel_title_glyph_bounds_pt": b1,
                    "legend_glyph_bounds_pt": b2,
                    "bounding_box_intersection_pt": intersection
                    if intersection[0] < intersection[2]
                    and intersection[1] < intersection[3]
                    else None,
                    "glyph_path_intersection": bool(
                        p1.intersects_path(p2, filled=True)
                    ),
                }
            )
        analyses.append(
            {
                "index": selected["index"],
                "svg_path": str(path),
                "svg_sha256": sha(path),
                "tests": tests,
            }
        )
    result = {
        "schema": "supplement-density-svg-title-legend-clearance/1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": "/tmp/shot-pattern-validation-venv-20261009/bin/python "
        + str(pathlib.Path(__file__).resolve()),
        "inventory_path": str(INVENTORY),
        "inventory_sha256": sha(INVENTORY),
        "method": "Matplotlib TextPath with saved SVG DejaVu Sans 10 pt text and exact SVG anchor coordinates; outline intersection is evaluated in SVG point coordinates.",
        "analyses": analyses,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha(OUTPUT),
                "intersections": [
                    (a["index"], t["panel_title"], t["legend_label"])
                    for a in analyses
                    for t in a["tests"]
                    if t["glyph_path_intersection"]
                ],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
