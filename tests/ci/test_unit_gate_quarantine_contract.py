"""Contract tests for the removal-only unit-gate quarantine ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci import check_unit_gate_quarantine as quarantine


REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def _ledger(*node_ids: str) -> dict[str, object]:
    return {
        "issue": "#8766",
        "ratchet": "Entries may only be removed, never added.",
        "node_ids": list(node_ids),
    }


def _clusters(*selectors: tuple[str, list[str]]) -> dict[str, object]:
    return {
        "schema_version": "unit-gate-quarantine-clusters/v1",
        "issue": "#8766",
        "clusters": [
            {
                "cluster_id": cluster_id,
                "owner": "Named subsystem maintainers",
                "rationale": "One deterministic failure family.",
                "test_path_prefixes": prefixes,
                "reproduction_command": (
                    "python3 scripts/ci/check_unit_gate_quarantine.py "
                    f"--cluster {cluster_id} --run"
                ),
                "exit_criteria": "The selected tests pass without quarantine.",
                "blocking_status": "Implementation repair required.",
            }
            for cluster_id, prefixes in selectors
        ],
    }


def test_repository_ledger_has_unique_exact_cluster_coverage() -> None:
    ledger = quarantine.load_json(
        REPO_ROOT / "scripts" / "config" / "unit_gate_quarantine.json"
    )
    clusters = quarantine.load_json(
        REPO_ROOT / "scripts" / "config" / "unit_gate_quarantine_clusters.json"
    )

    report = quarantine.validate_contract(ledger, clusters)

    assert report.node_count == 520
    assert report.cluster_count >= 8
    assert report.errors == ()


def test_duplicate_node_ids_fail_closed() -> None:
    node_id = "tests/unit/test_widget.py::test_one"
    report = quarantine.validate_contract(
        _ledger(node_id, node_id),
        _clusters(("widgets", ["tests/unit/test_widget.py"])),
    )

    assert report.errors == (f"duplicate quarantine node ID: {node_id}",)


def test_unassigned_node_id_fails_closed() -> None:
    node_id = "tests/unit/test_widget.py::test_one"
    report = quarantine.validate_contract(
        _ledger(node_id),
        _clusters(("other", ["tests/unit/test_other.py"])),
    )

    assert report.errors == (f"unassigned quarantine node ID: {node_id}",)


def test_ambiguous_cluster_assignment_fails_closed() -> None:
    node_id = "tests/unit/test_widget.py::test_one"
    report = quarantine.validate_contract(
        _ledger(node_id),
        _clusters(
            ("unit", ["tests/unit/"]),
            ("widgets", ["tests/unit/test_widget.py"]),
        ),
    )

    assert report.errors == (
        f"quarantine node ID assigned to multiple clusters (unit, widgets): {node_id}",
    )


def test_cluster_metadata_and_reproduction_command_are_required() -> None:
    node_id = "tests/unit/test_widget.py::test_one"
    clusters = _clusters(("widgets", ["tests/unit/test_widget.py"]))
    del clusters["clusters"][0]["exit_criteria"]  # type: ignore[index]

    report = quarantine.validate_contract(_ledger(node_id), clusters)

    assert report.errors == ("cluster widgets has empty exit_criteria",)


def test_removal_only_comparison_allows_shrinkage() -> None:
    baseline = {
        "tests/unit/test_widget.py::test_one",
        "tests/unit/test_widget.py::test_two",
    }
    current = {"tests/unit/test_widget.py::test_one"}

    assert quarantine.removal_only_errors(current, baseline) == ()


def test_removal_only_comparison_rejects_replacement_or_growth() -> None:
    baseline = {"tests/unit/test_widget.py::test_one"}
    current = {"tests/unit/test_widget.py::test_two"}

    assert quarantine.removal_only_errors(current, baseline) == (
        "quarantine ledger added node ID: tests/unit/test_widget.py::test_two",
    )


def test_cluster_node_ids_are_deterministic() -> None:
    first = "tests/unit/test_widget.py::test_b"
    second = "tests/unit/test_widget.py::test_a"
    ledger = _ledger(first, second)
    clusters = _clusters(("widgets", ["tests/unit/test_widget.py"]))

    assert quarantine.cluster_node_ids(ledger, clusters, "widgets") == (
        second,
        first,
    )


def test_load_json_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        quarantine.load_json(path)


def test_green_suite_enforces_contract_and_removal_only_base_comparison() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-standard.yml").read_text(
        encoding="utf-8"
    )

    assert "Validate Unit-Gate Quarantine Contract" in workflow
    assert "check_unit_gate_quarantine.py" in workflow
    assert "--baseline-ref" in workflow
    assert "origin/${{ github.base_ref }}" in workflow
