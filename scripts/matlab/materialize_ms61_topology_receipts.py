#!/usr/bin/env python3
"""Materialize MS-61 (#10348) topology + terminal disclosure receipts for run-103.

Derives fail-closed software receipts from the committed run-102 returned package.
Does not invent native R2025b fit success: native_gate.json stays blocked until a
licensed DeskComputer fit under two_window_fit_9967_103 produces a G1-passing
full-marker terminal (or MS-104 lands neck/full-body model work).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.motion_matching.full_marker_terminal import (  # noqa: E402
    TerminalMarkerBreakdown,
    compute_terminal_marker_breakdown,
)
from src.shared.python.motion_matching.simscape_topology import (  # noqa: E402
    SimscapeTopologyReport,
    classify_simscape_topology,
)

RUN102 = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_102"
)
RUN103 = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_103"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_run102() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate = json.loads(
        (RUN102 / "returned-candidate.json").read_text(encoding="utf-8")
    )
    qualified = json.loads(
        (RUN102 / "qualified_candidate_replay.json").read_text(encoding="utf-8")
    )
    run_manifest = json.loads(
        (RUN102 / "run_manifest.json").read_text(encoding="utf-8")
    )
    release = str(qualified.get("matlab_release", "")).strip().lower().lstrip("r")
    if release != "2025b":
        raise ValueError(
            "MS-61 requires R2025b identity from run-102 qualified replay; "
            f"got {qualified.get('matlab_release')!r}"
        )
    return candidate, qualified, run_manifest


def _classify_and_breakdown(
    candidate: dict[str, Any],
) -> tuple[SimscapeTopologyReport, TerminalMarkerBreakdown]:
    coords = list(candidate["coordinate_names"])
    labels = list(candidate["marker_labels"])
    bodies = list(candidate["marker_bodies"])
    topology = classify_simscape_topology(
        coordinate_names=coords,
        marker_labels=labels,
        marker_bodies=bodies,
    )
    with np.load(RUN102 / "returned-replay.npz") as raw:
        breakdown = compute_terminal_marker_breakdown(
            pred_markers_m=raw["markers_m"],
            target_markers_m=raw["target_m"],
            valid=raw["valid"],
            marker_labels=labels,
            marker_bodies=bodies,
        )
    return topology, breakdown


def _native_gate_payload(
    *,
    qualified: dict[str, Any],
    run_manifest: dict[str, Any],
    full_m: float,
) -> dict[str, Any]:
    return {
        "schema_version": "simscape-native-gate/1",
        "issue": "#10348",
        "run_id": "two_window_fit_9967_103",
        "status": "blocked",
        "is_physically_accepted": False,
        "g1_full_marker_terminal_passed": False,
        "full_marker_terminal_rms_m": full_m,
        "g1_terminal_ceiling_m": 0.035,
        "matlab_release_required": "2025b",
        "matlab_version": qualified.get("matlab_version"),
        "host": run_manifest.get("host"),
        "blocked_reasons": [
            "Native two_window_fit_9967_103 Fit has not been executed on a licensed "
            "R2025b host in this worktree.",
            "Source run-102 full-marker terminal "
            f"({full_m * 1000:.3f} mm) exceeds G1 ceiling (35 mm).",
            "Reduced 27-DOF topology lacks an independent neck; Hub head/back "
            "coupling remains the documented rigidity floor (see topology_report).",
        ],
        "follow_up": {
            "model_work": (
                "MS-104 (#10378) full-body Simscape flagship / neck capability"
            ),
            "runtime_inventory": (
                "MS-102 (#10376) soft dependency — does not block software contracts"
            ),
            "native_command": (
                "powershell scripts/matlab/run_simscape_candidate.ps1 "
                "-Run two_window_fit_9967_103 -Fit"
            ),
        },
        "artifacts": {
            "topology_report": "topology_report.json",
            "terminal_breakdown": "terminal_breakdown.json",
            "runtime_license_receipt": "runtime_license_receipt.json",
            "parity_receipt": "parity_receipt.json",
            "source_run102_replay": "../two_window_fit_9967_102/returned-replay.npz",
        },
    }


def _build_receipts(
    *,
    candidate: dict[str, Any],
    qualified: dict[str, Any],
    run_manifest: dict[str, Any],
    topology: SimscapeTopologyReport,
    breakdown: TerminalMarkerBreakdown,
) -> dict[str, dict[str, Any]]:
    full_m = float(breakdown.full_marker_terminal_rms_m)
    parity = qualified.get("cross_engine_parity") or {}
    return {
        "topology_report.json": {
            "schema_version": "simscape-topology-report/1",
            "issue": "#10348",
            "source_run_id": "two_window_fit_9967_102",
            "model_sha256": candidate["model_sha256"],
            **topology.as_dict(),
            "rigidity_floor_ref": (
                "docs/development/simscape_tour_matching/native_evidence/"
                "fixed_attachment_rigidity_floor.json"
            ),
            "note": (
                "Software topology classification from the committed 27-DOF candidate. "
                "R2025b runtime identity is recorded in runtime_license_receipt.json "
                "(no inferred 1000-block license cap)."
            ),
        },
        "terminal_breakdown.json": {
            "schema_version": "simscape-terminal-breakdown/1",
            "issue": "#10348",
            "source_run_id": "two_window_fit_9967_102",
            "acceptance_terminal_source": "full_marker",
            "g1_terminal_ceiling_m": 0.035,
            **breakdown.as_dict(),
            "note": (
                "Full-marker terminal remains above the G1 35 mm ceiling. "
                "body_excluding_head_terminal_rms_m is a reduced-model diagnostic only "
                "and must not be used for full-body acceptance."
            ),
        },
        "native_gate.json": _native_gate_payload(
            qualified=qualified, run_manifest=run_manifest, full_m=full_m
        ),
        "runtime_license_receipt.json": {
            "schema_version": "simscape-runtime-license/1",
            "issue": "#10348",
            "run_id": "two_window_fit_9967_103",
            "matlab_release": "2025b",
            "matlab_version": qualified.get("matlab_version"),
            "host": run_manifest.get("host"),
            "model_sha256": candidate["model_sha256"],
            "candidate_sha256": run_manifest.get("candidate_sha256"),
            "source_run_manifest": (
                "docs/development/simscape_tour_matching/native_evidence/"
                "two_window_fit_9967_102/run_manifest.json"
            ),
            "license_cap_assumed": False,
            "note": (
                "R2025b identity reused from committed MS-60 run-102 receipt. "
                "No inferred 1000-block license cap."
            ),
        },
        "parity_receipt.json": {
            "schema_version": "simscape-parity-receipt/1",
            "issue": "#10348",
            "run_id": "two_window_fit_9967_103",
            "parent_run_id": "two_window_fit_9967_102",
            "max_marker_euclidean_discrepancy_m": parity.get(
                "max_marker_euclidean_discrepancy_m"
            ),
            "mean_marker_euclidean_discrepancy_m": parity.get(
                "mean_marker_euclidean_discrepancy_m"
            ),
            "source": (
                "docs/development/simscape_tour_matching/native_evidence/"
                "two_window_fit_9967_102/qualified_candidate_replay.json"
            ),
            "note": (
                "Pinocchio↔Simscape parity retained from MS-60 run-102 R2025b cold "
                "replay; MS-61 does not re-claim native physical success."
            ),
        },
    }


def _write_handoff(
    output_dir: Path,
    *,
    qualified: dict[str, Any],
    run_manifest: dict[str, Any],
    breakdown: TerminalMarkerBreakdown,
    parity: dict[str, Any],
) -> None:
    full_m = float(breakdown.full_marker_terminal_rms_m)
    (output_dir / "HANDOFF.md").write_text(
        "\n".join(
            [
                "# MS-61 Topology + Full-Marker Terminal Qualification (#10348)",
                "",
                "Software contracts for Simscape R2025b topology classification and",
                "full-marker terminal disclosure. **Native G1 is blocked** — this",
                "directory does not invent a physical pass.",
                "",
                "## R2025b Runtime / License",
                "",
                f"- Release: **R2025b** (`{qualified.get('matlab_version')}`)",
                f"- Host: **{run_manifest.get('host')}** (from run-102 manifest)",
                "- License cap assumed: **false** (no inferred 1000-block ceiling)",
                "",
                "## Derived From Run-102",
                "",
                f"- Full-marker terminal: **{full_m * 1000:.3f} mm** (G1 ceiling 35 mm)",
                (
                    "- Head-cluster terminal: "
                    f"**{(breakdown.head_cluster_terminal_rms_m or 0) * 1000:.3f} mm**"
                ),
                (
                    "- Body-excluding-head (diagnostic only): "
                    f"**{(breakdown.body_excluding_head_terminal_rms_m or 0) * 1000:.3f} mm**"
                ),
                "- Topology profile: `reduced_27_no_neck` (no independent neck DOFs)",
                (
                    "- Pinocchio↔Simscape max Euclidean: "
                    f"**{(parity.get('max_marker_euclidean_discrepancy_m') or 0) * 1000:.3f} mm**"
                ),
                "",
                "## Receipts",
                "",
                "- `topology_report.json`",
                "- `terminal_breakdown.json`",
                "- `native_gate.json` (status=`blocked`)",
                "- `runtime_license_receipt.json`",
                "- `parity_receipt.json`",
                "",
                "## Next Native Action (DeskComputer / R2025b Only)",
                "",
                "```powershell",
                "PYTHONPATH=. python scripts/matlab/materialize_ms61_topology_receipts.py",
                "powershell scripts/matlab/run_simscape_candidate.ps1 "
                "-Run two_window_fit_9967_103 -Fit",
                "```",
                "",
                "If neck/model development is required, track under MS-104 (#10378).",
                "MS-102 (#10376) inventory is a soft dependency and does not block",
                "these software contracts.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def materialize(output_dir: Path = RUN103) -> None:
    candidate, qualified, run_manifest = _load_run102()
    topology, breakdown = _classify_and_breakdown(candidate)
    receipts = _build_receipts(
        candidate=candidate,
        qualified=qualified,
        run_manifest=run_manifest,
        topology=topology,
        breakdown=breakdown,
    )
    for name, payload in receipts.items():
        _write_json(output_dir / name, payload)
    _write_handoff(
        output_dir,
        qualified=qualified,
        run_manifest=run_manifest,
        breakdown=breakdown,
        parity=qualified.get("cross_engine_parity") or {},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN103,
        help="Evidence directory for run-103 receipts",
    )
    args = parser.parse_args()
    materialize(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
