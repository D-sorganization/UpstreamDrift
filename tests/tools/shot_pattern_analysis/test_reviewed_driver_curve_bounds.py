"""Explicit-pair contract for the predeclared Driver5° curve control extension."""

import hashlib
import json
from pathlib import Path

import pytest

from tests.tools.shot_pattern_analysis.test_matched_explicit_source_pair import (
    _write_cell,
)
from src.tools.shot_pattern_analysis.matched_lean_control import (
    analyze_explicit_source_pair,
)

pytestmark = pytest.mark.unit


def _rewrite(reference: dict, changes: dict, *, protocol: str | None = None) -> None:
    root = Path(reference["source_root"])
    directory = root / reference["cell"]
    plan = json.loads((root / "matrix_plan.json").read_text())
    config = plan["cells"][reference["cell"]]
    config.update(changes)
    if protocol is not None:
        reference["source_protocol"] = protocol
        plan["supplement_protocol"] = (
            None if protocol.startswith("original_") else protocol
        )
    for name in ("summary.json", "run_start.json"):
        path = directory / name
        data = json.loads(path.read_text())
        data["config"] = config
        path.write_text(json.dumps(data))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["files_sha256"]:
        manifest["files_sha256"][name] = hashlib.sha256(
            (directory / name).read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    (root / "matrix_plan.json").write_text(json.dumps(plan))
    reference["source_plan_sha256"] = hashlib.sha256(
        (root / "matrix_plan.json").read_bytes()
    ).hexdigest()
    reference["source_cell_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def _pair(tmp_path: Path) -> tuple[dict, dict]:
    first = _write_cell(
        tmp_path / "original",
        lean=5,
        target_request=77.40247724866299,
        protocol="original_primary/1",
        club_id="driver",
        curve_scale=1,
        speed_factors=(0.75, 1.25),
    )
    _rewrite(first, {"match_face_bounds_deg": None}, protocol="original_control/1")
    second = _write_cell(
        tmp_path / "supplement",
        lean=5,
        target_request=77.40247724866299,
        protocol="driver-control-speed-bounds/1",
        club_id="driver",
        curve_scale=2,
        speed_factors=(0.75, 1.30),
    )
    return first, second


def test_reviewed_driver_curve_bounds_pass_full_pair_api(tmp_path: Path) -> None:
    first, second = _pair(tmp_path)
    result = analyze_explicit_source_pair(
        first, second, comparison_kind="curve_strength", target_x_m=77.40247724866299
    )
    assert result["allowed_config_differences"] == [
        "curve_scale",
        "match_face_bounds_deg",
        "match_speed_factors",
    ]
    assert result["common_supported_n"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_protocol",
        "wrong_original_protocol",
        "face_widening",
        "speed_widening",
        "lower_speed_change",
        "no_fixed_target",
        "target_change",
        "wrong_lean",
        "wrong_club",
        "reverse_curve",
    ],
)
def test_unreviewed_driver_curve_bound_changes_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    first, second = _pair(tmp_path)
    if mutation == "wrong_protocol":
        _rewrite(second, {}, protocol="unreviewed-speed-bounds/1")
    elif mutation == "wrong_original_protocol":
        _rewrite(first, {}, protocol="original_primary/1")
    elif mutation == "face_widening":
        _rewrite(second, {"match_face_bounds_deg": [-7, 7]})
    elif mutation == "speed_widening":
        _rewrite(second, {"match_speed_factors": [0.75, 1.35]})
    elif mutation == "lower_speed_change":
        _rewrite(second, {"match_speed_factors": [0.70, 1.30]})
    elif mutation == "no_fixed_target":
        _rewrite(first, {"target_x_m": None})
    elif mutation == "target_change":
        _rewrite(second, {"target_x_m": 78.0})
    elif mutation in ("wrong_lean", "wrong_club"):
        for ref in (first, second):
            plan = json.loads(
                (Path(ref["source_root"]) / "matrix_plan.json").read_text()
            )
            club = plan["cells"][ref["cell"]]["club"]
            club["shaft_lean_deg" if mutation == "wrong_lean" else "club_id"] = (
                0 if mutation == "wrong_lean" else "seven_iron"
            )
            _rewrite(ref, {"club": club})
    else:
        _rewrite(first, {"curve_scale": 2})
        _rewrite(second, {"curve_scale": 1})
    with pytest.raises(ValueError):
        analyze_explicit_source_pair(
            first,
            second,
            comparison_kind="curve_strength",
            target_x_m=77.40247724866299,
        )
