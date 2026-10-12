"""Compare historical Git scorer with refactored public API, without flights."""

from pathlib import Path
import hashlib
import json
import subprocess
import types
import sys
from typing import Any
import numpy as np
from src.tools.shot_pattern_analysis.scenario_scoring import score_course_endpoints

old = subprocess.check_output(
    ["git", "show", "6cea552a^:src/tools/shot_pattern_analysis/scenario_scoring.py"]
)
module = types.ModuleType("src.tools.shot_pattern_analysis._independent_old_scenario")
module.__package__ = "src.tools.shot_pattern_analysis"
exec(compile(old, "git:6cea552a^:scenario_scoring.py", "exec"), module.__dict__)  # nosec B102
fixture = json.loads(
    Path(
        "tests/tools/shot_pattern_analysis/data/legacy_course_reports.json"
    ).read_text()
)
assert hashlib.sha256(old).hexdigest() == fixture["pre_refactor_source_sha256"]
for case in fixture["cases"]:
    a = module.score_course_endpoints(case["endpoints"], **case["options"])
    assert (
        a
        == case["report"]
        == score_course_endpoints(case["endpoints"], **case["options"])
    )
rng = np.random.default_rng(12006)
for club, target in [
    ("driver", 180.0),
    ("seven_iron", 140.0),
    ("pitching_wedge", 80.0),
]:
    for k in range(10):
        endpoints = {
            name: [
                (i, float(target + rng.uniform(-30, 30)), float(rng.uniform(-30, 30)))
                for i in range(20)
            ]
            for name in ("Straight", "Draw", "Fade")
        }
        opts: dict[str, Any] = {
            "club_id": club,
            "target_x_m": target,
            "hole_distance_m": 350.0 + 10 * k,
            "fairway_half_width_m": 10.0 + k,
            "green_radius_m": 10.0 + k,
            "seed": 20261009,
        }
        assert module.score_course_endpoints(
            endpoints, **opts
        ) == score_course_endpoints(endpoints, **opts)
for case in fixture["errors"]:
    errors = []
    for fn in (module.score_course_endpoints, score_course_endpoints):
        try:
            fn(case["endpoints"], **case["options"])
        except ValueError as exc:
            errors.append(str(exc))
    assert errors == [case["error"]] * 2
paths = [
    "src/tools/shot_pattern_analysis/scenario_scoring.py",
    "src/tools/shot_pattern_analysis/course_score_vectors.py",
    "src/tools/shot_pattern_analysis/provenance.py",
    "tests/tools/shot_pattern_analysis/data/legacy_course_reports.json",
]
sys.stdout.write(
    json.dumps(
        {
            "status": "exact_saved_and_randomized_report_parity_pass",
            "old_source_sha256": hashlib.sha256(old).hexdigest(),
            "current_sha256": {
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
            },
            "saved_reports": 6,
            "randomized_reports": 30,
            "exact_error_cases": 3,
            "seed": 12006,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Course scoring refactor, no flight execution",
        },
        indent=2,
    )
)
