"""Direct public-API scoring of nonzero-lean common-distance pilot controls."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.tools.launch_monitor_model import (
    CourseStateColumnsV1,
    StrokesGainedRequestV1,
    analyze_source_backed_strokes_gained,
)
from src.tools.shot_pattern_analysis.scoring import (
    YARDS_PER_METRE,
    build_broadie_approx_baseline,
)

root = Path(sys.argv[1])
output = Path(sys.argv[2])
baseline = build_broadie_approx_baseline()
results = []
columns = {
    "context_column": "context",
    "target_column": "target",
    "distance_unit": "yd",
}
request = StrokesGainedRequestV1(
    start=CourseStateColumnsV1(
        lie_column="start_lie", distance_column="start_distance", **columns
    ),
    finish=CourseStateColumnsV1(
        lie_column="finish_lie", distance_column="finish_distance", **columns
    ),
    min_samples=1,
)
for club in ("driver", "seven_iron", "pitching_wedge"):
    for lean in (5, 10):
        name = f"{club}_lean{lean}_curve1_rotation1_combined2"
        path = root / name
        summary = json.loads((path / "summary.json").read_text())
        item = {
            "cell": name,
            "summary_sha256": hashlib.sha256(
                (path / "summary.json").read_bytes()
            ).hexdigest(),
        }
        if summary["status"] != "completed":
            item.update(status="unavailable", reason=summary["reason"])
            results.append(item)
            continue
        with gzip.open(path / "shots.csv.gz", "rt") as handle:
            shots = list(csv.DictReader(handle))
        rows = []
        for shot in shots:
            assert shot["status"] == "supported"
            x, y = float(shot["carry_x_m"]), float(shot["carry_y_m"])
            driver = club == "driver"
            distance = (
                math.hypot(400.0 - x, y)
                if driver
                else math.hypot(x - summary["target_x_m"], y)
            )
            primary = (
                abs(y) <= 15.0
                if driver
                else distance <= summary["config"]["target_radius_m"]
            )
            rows.append(
                {
                    "start_lie": "tee" if driver else "fairway",
                    "finish_lie": ("fairway" if driver else "green")
                    if primary
                    else "rough",
                    "start_distance": (400.0 if driver else summary["target_x_m"])
                    * YARDS_PER_METRE,
                    "finish_distance": distance * YARDS_PER_METRE,
                    "context": "standard",
                    "target": "hole",
                }
            )
        scored = analyze_source_backed_strokes_gained(
            pd.DataFrame(rows), baseline, request
        )
        assert scored.exclusions.total_excluded == 0 and len(scored.row_results) == len(
            shots
        )
        errors = {
            pattern: abs(
                float(
                    np.mean(
                        [
                            r.strokes_gained
                            for r in scored.row_results
                            if shots[r.source_index]["pattern"] == pattern
                        ]
                    )
                )
                - summary["course_scoring"]["patterns"][pattern]["mean_strokes_gained"]
            )
            for pattern in ("Straight", "Draw", "Fade")
        }
        item.update(
            status="checked",
            rows=len(shots),
            mean_sg_errors=errors,
            pass_check=max(errors.values()) <= 1e-12,
        )
        results.append(item)
receipt = {
    "scope": "Nonzero-lean direct public-API pilot check, no final inference",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "baseline_sha256": baseline.table_sha256,
    "cells": results,
    "pass": all(r.get("pass_check", True) for r in results),
}
output.write_text(json.dumps(receipt, indent=2) + "\n")
sys.stdout.write(json.dumps(receipt, indent=2) + "\n")
