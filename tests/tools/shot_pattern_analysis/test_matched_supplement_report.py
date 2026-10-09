"""Contracts for loading the separately predeclared bounds supplement."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_matrix import noise_profile

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = (
    ROOT / "docs/research/shot_pattern_analysis/phase2/review_followups/"
    "SUPPLEMENTAL_EXECUTION_INVENTORY.json"
)
PROTOCOLS = {
    "pw_within_lean_face": "pw-primary-face-bounds/1",
    "pw_fixed_target_face": "pw-control-face-bounds/1",
    "driver_fixed_target_speed": "driver-control-speed-bounds/1",
}
TARGETS = {"pitching_wedge": 77.40247724866299, "driver": 178.94875782337786}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(entry: dict) -> dict:
    name = entry["cell"]
    club_id = next(
        name
        for name in ("pitching_wedge", "driver")
        if entry["cell"].startswith(name + "_")
    )
    lean_text, curve_text, rotation_text, *profile_words = entry["cell"][
        len(club_id) + 1 :
    ].split("_")
    profile = "_".join(profile_words)
    club = replace(
        matched_preset(club_id), shaft_lean_deg=float(lean_text.removeprefix("lean"))
    )
    speed = float(entry["speed_bounds_mps"][0]) / club.club_speed_mps
    speed_hi = float(entry["speed_bounds_mps"][1]) / club.club_speed_mps
    return {
        "club": asdict(club),
        "noise": asdict(
            noise_profile(profile, float(rotation_text.removeprefix("rotation")))
        ),
        "n_shots": entry["n_shots"],
        "seed": entry["seed"],
        "curve_scale": float(curve_text.removeprefix("curve")),
        "target_x_m": entry["target_x_m"],
        "match_tolerance_m": entry["match_tolerance_m"],
        "target_radius_m": 15.0,
        "match_face_bounds_deg": entry["face_bounds_deg"],
        "match_speed_factors": [speed, speed_hi],
    }


def _write_cell(root: Path, name: str, config: dict, target: float) -> dict:
    directory = root / name
    directory.mkdir(parents=True)
    execution = {"source_sha256": {"frozen.py": "fixture"}}
    (directory / "run_start.json").write_text(
        json.dumps({"config": config, "execution": execution})
    )
    nominals = {
        pattern: {
            "face_deg": 0.0,
            "path_deg": 0.0,
            "speed_mps": config["club"]["club_speed_mps"],
            "residual_m": 0.0,
            "outcome": {"carry_x_m": target, "carry_y_m": 0.0},
        }
        for pattern in ("Straight", "Draw", "Fade")
    }
    summary = {
        "config": config,
        "status": "completed",
        "target_x_m": target,
        "nominals": nominals,
        "summary": {
            pattern: {"total_n": config["n_shots"], "supported_n": config["n_shots"]}
            for pattern in ("Straight", "Draw", "Fade")
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    with gzip.open(directory / "shots.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pattern", "shot_index", "status"))
    hashes = {
        filename: _sha(directory / filename)
        for filename in ("run_start.json", "summary.json", "shots.csv.gz")
    }
    (directory / "manifest.json").write_text(
        json.dumps({"schema": "matched-shot-study/1", "files_sha256": hashes})
    )
    return summary


def _make_roots(tmp_path: Path, *, mutations: dict | None = None) -> tuple[dict, dict]:
    inventory = json.loads(INVENTORY.read_text())
    entries_by_kind = {
        kind: [entry for entry in inventory["entries"] if entry["protocol"] == kind]
        for kind in PROTOCOLS
    }
    roots = {}
    for kind, protocol_id in PROTOCOLS.items():
        root = tmp_path / protocol_id.replace("/", "_")
        root.mkdir()
        cells = {}
        summaries = {}
        entries = entries_by_kind[kind]
        for entry in entries:
            config = _config(entry)
            if mutations and entry["cell"] in mutations:
                mutations[entry["cell"]](config)
            cells[entry["cell"]] = config
            target = entry["target_x_m"] or TARGETS["pitching_wedge"]
            summaries[entry["cell"]] = _write_cell(root, entry["cell"], config, target)
        plan = {
            "schema": "matched-shot-matrix/1",
            "stage": "final",
            "execution": {"source_sha256": {"frozen.py": "fixture"}},
            "supplement_protocol": protocol_id,
            "selected_cell_names": list(cells),
            "cells": cells,
        }
        (root / "matrix_plan.json").write_text(json.dumps(plan))
        roots[protocol_id] = root
    return roots, inventory


def test_loader_requires_exact_protocol_inventory_and_marks_four_source_aliases(
    tmp_path,
) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_report import (
        load_supplement_sources,
        write_supplement_artifacts,
    )

    roots, inventory = _make_roots(tmp_path)
    report = load_supplement_sources(roots, INVENTORY)

    assert report["schema"] == "matched-supplement-report/1"
    assert len(report["unique_cells"]) == 26
    assert report["protocol_counts"] == {
        "pw-primary-face-bounds/1": 18,
        "pw-control-face-bounds/1": 4,
        "driver-control-speed-bounds/1": 4,
    }
    assert len(report["aliases"]) == 4
    assert all(
        alias["execution_kind"] == "provenance_alias" for alias in report["aliases"]
    )
    assert all(
        "summary" not in alias and "receipt" not in alias for alias in report["aliases"]
    )
    assert all(alias["proof_status"] == "verified" for alias in report["aliases"])
    assert report["inventory_sha256"] == _sha(INVENTORY)
    paths = write_supplement_artifacts(report, tmp_path / "artifacts")
    assert paths["inventory"].name == "REPORT_SUPPLEMENT.csv"
    aliases = json.loads(paths["control_aliases"].read_text())
    assert aliases["schema"] == "matched-study-control-aliases/1"
    assert len(aliases["references"]) == 4
    assert all(
        "summary" not in ref and "receipt" not in ref for ref in aliases["references"]
    )


def test_loader_rejects_missing_extra_duplicate_or_mislabelled_plan_cells(
    tmp_path,
) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_report import (
        load_supplement_sources,
    )

    roots, _ = _make_roots(tmp_path)
    protocol, root = next(iter(roots.items()))
    plan_path = root / "matrix_plan.json"
    original = json.loads(plan_path.read_text())
    plan = dict(original)
    plan["selected_cell_names"] = plan["selected_cell_names"][:-1]
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="selected cell inventory"):
        load_supplement_sources(roots, INVENTORY)
    plan = dict(original)
    plan["supplement_protocol"] = "wrong/1"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="protocol ID"):
        load_supplement_sources(roots, INVENTORY)
    plan = dict(original)
    plan["selected_cell_names"] = [
        *plan["selected_cell_names"],
        plan["selected_cell_names"][0],
    ]
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="duplicate selected"):
        load_supplement_sources(roots, INVENTORY)
    roots[protocol] = root


def test_loader_rejects_wrong_bounds_even_if_cell_names_match(tmp_path) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_report import (
        load_supplement_sources,
    )

    def mutate(config: dict) -> None:
        config["match_face_bounds_deg"] = [-5.0, 5.0]

    inventory = json.loads(INVENTORY.read_text())
    target = inventory["entries"][0]["cell"]
    roots, _ = _make_roots(tmp_path, mutations={target: mutate})
    with pytest.raises(ValueError, match="face bounds differ"):
        load_supplement_sources(roots, INVENTORY)


def test_loader_rejects_unknown_config_fields_and_nonradial_nominal_miss(
    tmp_path,
) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_report import (
        load_supplement_sources,
    )

    inventory = json.loads(INVENTORY.read_text())
    target_name = next(
        entry["cell"]
        for entry in inventory["entries"]
        if entry["protocol"] == "pw_within_lean_face"
    )
    roots, _ = _make_roots(tmp_path)
    protocol = PROTOCOLS["pw_within_lean_face"]
    root = roots[protocol]
    plan_path = root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["cells"][target_name]["target_radius_m"] = 99.0
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="unknown fields|differs from the declared"):
        load_supplement_sources(roots, INVENTORY)

    (tmp_path / "radial").mkdir()
    roots, _ = _make_roots(tmp_path / "radial")
    control_name = next(iter(inventory["control_aliases_to_primary"].values()))
    root = roots[PROTOCOLS["pw_within_lean_face"]]
    summary_path = root / control_name / "summary.json"
    summary = json.loads(summary_path.read_text())
    target = summary["target_x_m"]
    for pattern in ("Straight", "Draw", "Fade"):
        summary["nominals"][pattern]["outcome"]["carry_x_m"] = target + 0.04
        summary["nominals"][pattern]["outcome"]["carry_y_m"] = 0.04
    summary_path.write_text(json.dumps(summary))
    manifest_path = root / control_name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["summary.json"] = _sha(summary_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nominal misses"):
        load_supplement_sources(roots, INVENTORY)


def test_loader_rejects_nonfinite_nominal_coordinates(tmp_path) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_report import (
        load_supplement_sources,
    )

    inventory = json.loads(INVENTORY.read_text())
    control_name = next(iter(inventory["control_aliases_to_primary"].values()))
    roots, _ = _make_roots(tmp_path)
    root = roots[PROTOCOLS["pw_within_lean_face"]]
    summary_path = root / control_name / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["nominals"]["Draw"]["outcome"]["carry_x_m"] = float("nan")
    summary_path.write_text(json.dumps(summary))
    manifest_path = root / control_name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["summary.json"] = _sha(summary_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonfinite"):
        load_supplement_sources(roots, INVENTORY)


def test_loader_requires_one_shared_execution_snapshot_across_protocol_roots(
    tmp_path,
) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_report import (
        load_supplement_sources,
    )

    roots, _ = _make_roots(tmp_path)
    root = roots[PROTOCOLS["pw_fixed_target_face"]]
    plan_path = root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    changed_execution = {"source_sha256": {"frozen.py": "other-snapshot"}}
    plan["execution"] = changed_execution
    for name in plan["cells"]:
        cell_dir = root / name
        run_start_path = cell_dir / "run_start.json"
        run_start = json.loads(run_start_path.read_text())
        run_start["execution"] = changed_execution
        run_start_path.write_text(json.dumps(run_start))
        manifest_path = cell_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"]["run_start.json"] = _sha(run_start_path)
        manifest_path.write_text(json.dumps(manifest))
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="do not share one execution snapshot"):
        load_supplement_sources(roots, INVENTORY)


def test_comparison_report_exports_each_actual_pattern_and_refuses_overwrite(
    tmp_path,
) -> None:
    from src.tools.shot_pattern_analysis.matched_supplement_report import (
        write_supplement_comparisons,
    )

    result = {
        "comparison_kind": "curve_strength",
        "status": "conditional_only",
        "common_supported_n": 8,
        "total_n_per_population": 10,
        "conditioning": "same_indices_supported_in_all_six_populations",
        "input_error_max_abs_difference": 0.0,
        "limitations": "Matched conditions do not isolate causal curvature.",
        "sources": [
            {"protocol_id": "original-primary", "cell": "curve1"},
            {"protocol_id": "pw-primary-face-bounds/1", "cell": "curve2"},
        ],
        "patterns": {
            "Straight": {
                "physical_pattern": "Straight",
                "radial_error_difference_m": {"estimate": 0.0},
            },
            "Draw": {
                "physical_pattern": "Draw",
                "radial_error_difference_m": {"estimate": -0.2},
            },
            "Fade": {
                "physical_pattern": "Fade",
                "radial_error_difference_m": {"estimate": 0.3},
            },
        },
    }
    paths = write_supplement_comparisons([result], tmp_path / "report")
    rows = list(csv.DictReader(paths["csv"].open(newline="")))
    assert [row["physical_pattern"] for row in rows] == ["Straight", "Draw", "Fade"]
    assert rows[1]["second_protocol"] == "pw-primary-face-bounds/1"
    with pytest.raises(FileExistsError):
        write_supplement_comparisons([result], tmp_path / "report")
