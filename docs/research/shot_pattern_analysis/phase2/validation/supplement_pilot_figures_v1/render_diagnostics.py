"""Render source-matched supplemental pilot diagnostics from saved data only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.tools.shot_pattern_analysis.academic_plots import (
    write_academic_dispersion_figure,
    write_overhead_trajectory_figure,
)
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
from src.tools.shot_pattern_analysis.matched_report import (
    load_individually_supported_endpoints,
    load_validated_cell,
    verify_nominal_deliveries_hit_target,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    phase = Path("docs/research/shot_pattern_analysis/phase2")
    output = Path(__file__).resolve().parent
    receipt_path = output / "render_receipt.json"
    if receipt_path.exists():
        raise ValueError("existing diagnostic receipt must be preserved")
    execution = matched_source_snapshot()
    transition_path = phase / "source_transition_supplement.json"
    transition_sha = sha(transition_path)
    assert execution == json.loads(transition_path.read_text())["new_execution"]
    sources = {
        str(p): sha(p)
        for p in sorted(Path("src/tools/shot_pattern_analysis").glob("*.py"))
    }
    sources[str(Path(__file__).resolve())] = sha(Path(__file__))
    inputs, cells, artifacts = {}, {}, {}
    for name in ("pw_primary_face", "pw_control_face", "driver_control_speed"):
        root = phase / "supplement_pilot_v1" / name
        plan_path = root / "matrix_plan.json"
        inputs[str(plan_path)] = sha(plan_path)
        plan = json.loads(plan_path.read_text())
        assert plan["stage"] == "pilot" and plan["execution"] == execution
        candidates = [
            cell for cell in plan["cells"] if cell.endswith("rotation2_combined2")
        ]
        assert len(candidates) == 1
        cell = candidates[0]
        config = plan["cells"][cell]
        assert config["n_shots"] == 32
        directory = root / cell
        for filename in (
            "manifest.json",
            "summary.json",
            "shots.csv.gz",
            "flight_samples.json",
        ):
            path = directory / filename
            inputs[str(path)] = sha(path)
        summary = load_validated_cell(directory, config, execution)
        assert summary["status"] == "completed"
        target = summary["target_x_m"]
        verify_nominal_deliveries_hit_target(
            summary, target, config["match_tolerance_m"], context="pilot diagnostic"
        )
        points, totals = load_individually_supported_endpoints(
            directory / "shots.csv.gz", 32
        )
        counts = {pattern: len(values) for pattern, values in points.items()}
        assert all(n >= 2 for n in counts.values())
        scenario = f"Pilot: {config['club']['club_id'].replace('_', ' ').title()} · Lean {config['club']['shaft_lean_deg']:g}° · Curve ×{config['curve_scale']:g} · Shaft Twist SD 2°"
        metadata = {
            "Protocol": plan["supplement_protocol"],
            "Stage": "32-Shot Pilot; No Final Inference",
            "Population": "Individually Supported Endpoints",
            "Path SD": "0.5°",
            "Speed SD": "2 mph",
            "Strike SD": "0.25 in on Each Axis",
            "valid_shots_by_pattern": counts,
            "total_shots_by_pattern": totals,
        }
        destination = output / name
        destination.mkdir()
        density = write_academic_dispersion_figure(
            points,
            destination / "density",
            scenario=scenario,
            metadata=metadata,
            target_point_m=(target, 0),
            bins=48,
        )
        flight = json.loads((directory / "flight_samples.json").read_text())
        trajectories = flight["trajectories"]
        overhead = write_overhead_trajectory_figure(
            trajectories,
            destination / "overhead",
            target_x_m=target,
            scenario=scenario,
            metadata={
                "Protocol": plan["supplement_protocol"],
                "Stage": "32-Shot Pilot; No Final Inference",
                "Population": "Saved Raw Trajectories; Not Paired by Shot Index",
            },
        )
        for path in (*density.values(), *overhead.values()):
            artifacts[str(path.relative_to(output))] = sha(path)
        cells[name] = {
            "cell": cell,
            "protocol": plan["supplement_protocol"],
            "config": config,
            "nominals": summary["nominals"],
            "target_x_m": target,
            "supported_counts": counts,
            "total_counts": totals,
            "saved_trajectory_counts": {p: len(v) for p, v in trajectories.items()},
        }
    assert matched_source_snapshot() == execution
    assert sha(transition_path) == transition_sha
    assert all(sha(Path(p)) == digest for p, digest in inputs.items())
    assert all(sha(Path(p)) == digest for p, digest in sources.items())
    assert all(sha(output / p) == digest for p, digest in artifacts.items())
    receipt = {
        "schema": "supplement-pilot-render-diagnostics/1",
        "status": "visual_review_pending",
        "scope": "32-Shot Source-Matched Pilot Diagnostics; No Final Inference or MC Authorization",
        "execution": execution,
        "transition_receipt_sha256": transition_sha,
        "source_sha256": sources,
        "input_sha256": inputs,
        "cells": cells,
        "artifact_sha256": artifacts,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
