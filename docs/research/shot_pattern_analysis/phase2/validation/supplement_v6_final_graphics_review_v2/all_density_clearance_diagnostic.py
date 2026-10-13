import datetime
import hashlib
import importlib.util
import json
import sys
import pathlib
import re

BASE = pathlib.Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2/validation/supplement_v6_final_graphics_review_v2"
)
FIGURES = BASE.parent.parent / "supplement_report_v1/figures/comparisons"
OUTPUT = BASE / "all_density_clearance_diagnostic.json"
SOURCE = BASE / "curve_strength_144_layout_diagnostic.py"
spec = importlib.util.spec_from_file_location("layout_diagnostic", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load layout diagnostic: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = sorted(FIGURES.glob("*_density.svg"))
    assert len(files) == 384
    crowded = []
    for path in files:
        svg = path.read_text()
        metrics = []
        for label, legend in [
            ("Straight", "50% Simulated-Sample HDR"),
            ("Draw", "80% Simulated-Sample HDR"),
            ("Fade", "Common Target"),
        ]:
            match = re.search(r"<!-- (" + label + r" · [^\n]*?) -->", svg)
            assert match, path
            title = match.group(1)
            p1, b1, _ = module.glyph(title, svg)
            p2, b2, _ = module.glyph(legend, svg)
            clearance = b1[1] - b2[3]
            metrics.append(
                {
                    "title": title,
                    "legend": legend,
                    "bbox_vertical_clearance_pt": clearance,
                    "glyph_path_intersection": bool(p1.intersects_path(p2, filled=True))
                    if clearance < 2
                    else False,
                }
            )
        if any(v["bbox_vertical_clearance_pt"] < 2 for v in metrics):
            crowded.append(
                {
                    "svg_path": str(path),
                    "svg_sha256": sha(path),
                    "matching_png_path": str(path.with_suffix(".png")),
                    "matching_pdf_path": str(path.with_suffix(".pdf")),
                    "metrics": metrics,
                }
            )
    result = {
        "schema": "supplement-all-density-clearance/1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": "/tmp/shot-pattern-validation-venv-20261009/bin/python "  # nosec B108
        + str(pathlib.Path(__file__).resolve()),
        "threshold_pt": 2.0,
        "svg_count": len(files),
        "crowded_count": len(crowded),
        "source_sha256": sha(SOURCE),
        "crowded": crowded,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha(OUTPUT),
                "crowded_count": len(crowded),
                "colliding_count": sum(
                    any(m["glyph_path_intersection"] for m in item["metrics"])
                    for item in crowded
                ),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
