"""Fixed synthetic saved-data receipts; source and input hashes checked separately."""

import hashlib
import json
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import matched_lean_control as lean
from src.tools.shot_pattern_analysis import matched_sensitivity as sensitivity
from tests.tools.shot_pattern_analysis.test_matched_explicit_source_pair import (
    _write_cell,
)
from tests.tools.shot_pattern_analysis.test_matched_lean_control import saved_lean_plan
from tests.tools.shot_pattern_analysis.test_matched_sensitivity import saved_cell

pytestmark = pytest.mark.unit
REFERENCE = Path(__file__).with_name("fixtures") / "saved_comparison_reference.json"


def _without_run_hashes(report):
    """Exclude only changing receipt digests, retaining their inventory/order."""
    for key in ("input_sha256", "source_sha256", "analysis_source_sha256"):
        if key in report:
            report[key] = _digest_inventory(report[key])
    for source in report.get("sources", []):
        for key in (
            "source_plan_sha256",
            "source_cell_manifest_sha256",
            "source_csv_sha256",
        ):
            source[key] = "<run-specific SHA-256>"
    return report


def _digest_inventory(value):
    if isinstance(value, dict):
        return {key: _digest_inventory(item) for key, item in value.items()}
    assert isinstance(value, str) and len(value) == 64
    return "<run-specific SHA-256>"


def _assert_saved_hashes(report, root):
    for name, digest in report["input_sha256"].items():
        if name == "cell_manifests":
            for cell, value in digest.items():
                assert (
                    value
                    == hashlib.sha256(
                        (root / cell / "manifest.json").read_bytes()
                    ).hexdigest()
                )
        else:
            path = (
                root / name
                if name == "matrix_plan.json"
                else root / report["cell"] / name
            )
            assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_receipts(root):
    reports = {}
    for club in ("driver", "seven_iron", "pitching_wedge"):
        directory = root / club
        directory.mkdir()
        reports[club] = sensitivity.analyze_saved_sensitivity(
            *saved_cell(directory, club)
        )
        _assert_saved_hashes(reports[club], directory)
    for unavailable in (True, False):
        directory = root / f"lean_{unavailable}"
        directory.mkdir()
        saved_lean_plan(directory, unavailable_ten=unavailable)
        reports[directory.name] = lean.analyze_across_lean(directory)
        _assert_saved_hashes(reports[directory.name], directory)
    for kind, changes in (
        ("across_lean", {"lean": 5}),
        ("curve_strength", {"curve_scale": 2.0}),
        ("shaft_rotation_sd", {"rotation_sd": 2.0}),
    ):
        options = {
            "lean": 0,
            "target_request": 77.40247724866299,
            "protocol": "original_primary/1",
            "curve_scale": 1.0,
        }
        first = _write_cell(root / f"{kind}_a", **options)
        second = _write_cell(
            root / f"{kind}_b", **{**options, **changes}, invalid=("Draw", 3)
        )
        reports[kind] = lean.analyze_explicit_source_pair(
            first, second, comparison_kind=kind, target_x_m=77.40247724866299
        )
        for source, receipt in zip(
            (first, second), reports[kind]["sources"], strict=True
        ):
            for key in (
                "source_plan_sha256",
                "source_cell_manifest_sha256",
                "source_csv_sha256",
            ):
                assert receipt[key] == source[key]
    return {name: _without_run_hashes(report) for name, report in reports.items()}


def test_saved_comparison_receipts_remain_exact(tmp_path):
    expected = json.loads(REFERENCE.read_text())["receipts"]
    actual = synthetic_receipts(tmp_path)
    assert json.dumps(actual, allow_nan=False) == json.dumps(expected, allow_nan=False)


def test_sensitivity_grid_order_and_rejection():
    cases = list(sensitivity._sensitivity_cases("driver"))
    assert [name for name, _ in cases] == [
        f"width{width}_hole{hole}" for width in (10, 15, 20) for hole in (350, 400, 450)
    ]
    assert list(sensitivity._sensitivity_cases("seven_iron")) == [
        (f"radius{radius}", {"green_radius_m": float(radius)})
        for radius in (10, 15, 20)
    ]
    with pytest.raises(
        ValueError, match="^saved club is not a supported scoring preset$"
    ):
        sensitivity._sensitivity_cases("unknown")


def test_future_saved_comparison_sources_are_current():
    for module, hashes in (
        (sensitivity, sensitivity._source_hashes()),
        (lean, lean._paired_scoring_source_hashes()),
    ):
        path = Path(module.__file__)
        assert hashes[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
