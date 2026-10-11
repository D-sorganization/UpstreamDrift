"""Apply the saved glyph and colorbar diagnostics to every child SVG."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHILD_FIGURES = (
    HERE.parents[1] / "supplement_report_v2_layout_repair/figures/comparisons"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    outputs = []
    for name, suffix in (
        ("all_density_clearance_diagnostic.py", "top"),
        ("all_density_colorbar_diagnostic.py", "right"),
    ):
        source = HERE / name
        spec = importlib.util.spec_from_file_location("child_" + suffix, source)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__dict__["FIGURES"] = CHILD_FIGURES
        target = HERE / ("layout_child_" + suffix + "_clearance.json")
        module.__dict__["OUTPUT"] = target
        module.main()
        result = json.loads(target.read_text())
        if result["svg_count"] != 384 or result["crowded_count"] != 0:
            raise ValueError(f"{suffix} clearance failed: {result['crowded_count']}")
        outputs.append(
            {
                "diagnostic": name,
                "source_sha256": sha(source),
                "receipt": str(target),
                "receipt_sha256": sha(target),
            }
        )
    receipt = {
        "schema": "shot-pattern-layout-child-clearance/1",
        "child_figures": str(CHILD_FIGURES),
        "threshold_pt": 2.0,
        "svg_count": 384,
        "checks": outputs,
        "verdict": "pass",
    }
    target = HERE / "layout_child_clearance_receipt.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(
        json.dumps({"receipt": str(target), "sha256": sha(target), "verdict": "pass"})
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
