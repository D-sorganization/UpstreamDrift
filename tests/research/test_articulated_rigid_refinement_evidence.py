"""Published evidence checks for the #9153 rigid-refinement extension."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "articulated_rigid_refinement"
)


def test_committed_refinement_retains_all_cases_and_failed_states() -> None:
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))

    assert summary["counts"] == {
        "completed": 108,
        "failed": 0,
        "registered": 216,
        "unavailable": 108,
    }
    assert summary["checkpoint_inventory"]["count"] == 216
    assert summary["checkpoint_inventory"]["checkpoint_set_sha256"] == (
        "743999ac8afd516b2cea4e8f7ac43e56f87f66d9cdba1a2b5bad95081421e7c7"
    )
    assert summary["promotion"] == {
        "eligible": False,
        "failure_codes": [
            "native_engine_unavailable",
            "cross_engine_parity_unavailable",
            "refinement_failure",
        ],
    }
    completed = [
        group
        for group in summary["groups"]
        if group["engine"] == "mujoco" and group["status"] == "completed"
    ]
    assert len(completed) == 36
    assert all(
        max(group["momentum_relative_residuals"]) <= 0.02
        and max(group["work_relative_residuals"]) <= 0.01
        for group in completed
    )
    failed = [
        (group["source_case_index"], group["source_sample_index"], group["variant"])
        for group in completed
        if not group["passes"]
    ]
    assert failed == [(4, 0, "nominal"), (13, 0, "nominal"), (13, 12, "nominal")]


def test_checkpoint_inventory_matches_published_files() -> None:
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    expected = {
        row["name"]: (row["sha256"], row["size_bytes"])
        for row in summary["checkpoint_inventory"]["files"]
    }
    actual = sorted((EVIDENCE / "checkpoints").glob("case-*.json"))

    assert len(actual) == len(expected) == 216
    import hashlib

    for path in actual:
        content = path.read_bytes()
        assert expected[path.name] == (
            hashlib.sha256(content).hexdigest(),
            len(content),
        )
