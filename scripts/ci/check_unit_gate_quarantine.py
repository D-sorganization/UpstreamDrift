#!/usr/bin/env python3
"""Validate and reproduce the removal-only unit-gate quarantine clusters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "scripts" / "config" / "unit_gate_quarantine.json"
CLUSTER_PATH = REPO_ROOT / "scripts" / "config" / "unit_gate_quarantine_clusters.json"
LEDGER_REPO_PATH = "scripts/config/unit_gate_quarantine.json"
SCHEMA_VERSION = "unit-gate-quarantine-clusters/v1"
CLUSTER_ID = re.compile(r"^[a-z][a-z0-9_]*$")
REQUIRED_CLUSTER_TEXT = (
    "owner",
    "rationale",
    "reproduction_command",
    "exit_criteria",
    "blocking_status",
)


@dataclass(frozen=True)
class ValidationReport:
    """Deterministic validation result for one ledger/configuration pair."""

    node_count: int
    cluster_count: int
    errors: tuple[str, ...]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path`` or fail closed.

    Postcondition: the returned value is a dictionary.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _node_ids(ledger: dict[str, Any]) -> list[str]:
    raw = ledger.get("node_ids")
    if not isinstance(raw, list):
        return []
    return [node_id for node_id in raw if isinstance(node_id, str)]


def _clusters(configuration: dict[str, Any]) -> list[dict[str, Any]]:
    raw = configuration.get("clusters")
    if not isinstance(raw, list):
        return []
    return [cluster for cluster in raw if isinstance(cluster, dict)]


def _matches(node_id: str, prefixes: list[str]) -> bool:
    test_path = node_id.split("::", maxsplit=1)[0]
    for prefix in prefixes:
        if prefix.endswith("/") and test_path.startswith(prefix):
            return True
        if test_path == prefix:
            return True
    return False


def _cluster_assignments(
    ledger: dict[str, Any], configuration: dict[str, Any]
) -> dict[str, tuple[str, ...]]:
    assignments: dict[str, list[str]] = {}
    for node_id in sorted(set(_node_ids(ledger))):
        matches: list[str] = []
        for cluster in _clusters(configuration):
            cluster_id = cluster.get("cluster_id")
            prefixes = cluster.get("test_path_prefixes")
            if not isinstance(cluster_id, str) or not isinstance(prefixes, list):
                continue
            valid_prefixes = [prefix for prefix in prefixes if isinstance(prefix, str)]
            if _matches(node_id, valid_prefixes):
                matches.append(cluster_id)
        assignments[node_id] = matches
    return {node_id: tuple(matches) for node_id, matches in assignments.items()}


def validate_contract(
    ledger: dict[str, Any], configuration: dict[str, Any]
) -> ValidationReport:
    """Validate schema, metadata, uniqueness, and exact cluster coverage.

    Postcondition: all detected errors are returned in deterministic order.
    """

    errors: list[str] = []
    node_ids = _node_ids(ledger)
    raw_node_ids = ledger.get("node_ids")
    if not isinstance(raw_node_ids, list):
        errors.append("quarantine ledger node_ids must be a list")
    elif len(node_ids) != len(raw_node_ids):
        errors.append("quarantine ledger node_ids must contain only strings")
    if configuration.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"cluster schema_version must equal {SCHEMA_VERSION}")
    if ledger.get("issue") != configuration.get("issue"):
        errors.append("ledger and cluster configuration issues must match")

    seen_nodes: set[str] = set()
    for node_id in node_ids:
        if node_id in seen_nodes:
            errors.append(f"duplicate quarantine node ID: {node_id}")
        seen_nodes.add(node_id)

    clusters = _clusters(configuration)
    if not isinstance(configuration.get("clusters"), list):
        errors.append("cluster configuration clusters must be a list")
    seen_cluster_ids: set[str] = set()
    for index, cluster in enumerate(clusters):
        cluster_id = cluster.get("cluster_id")
        label = cluster_id if isinstance(cluster_id, str) else f"index {index}"
        if not isinstance(cluster_id, str) or not CLUSTER_ID.fullmatch(cluster_id):
            errors.append(f"cluster {label} has invalid cluster_id")
        elif cluster_id in seen_cluster_ids:
            errors.append(f"duplicate cluster_id: {cluster_id}")
        else:
            seen_cluster_ids.add(cluster_id)
        prefixes = cluster.get("test_path_prefixes")
        if not isinstance(prefixes, list) or not prefixes:
            errors.append(f"cluster {label} has empty test_path_prefixes")
        elif any(
            not isinstance(prefix, str) or not prefix.startswith("tests/")
            for prefix in prefixes
        ):
            errors.append(f"cluster {label} has invalid test_path_prefixes")
        for field in REQUIRED_CLUSTER_TEXT:
            value = cluster.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"cluster {label} has empty {field}")

    if not any(error.startswith("duplicate quarantine node ID") for error in errors):
        for node_id, matches in _cluster_assignments(ledger, configuration).items():
            if not matches:
                errors.append(f"unassigned quarantine node ID: {node_id}")
            elif len(matches) > 1:
                cluster_list = ", ".join(matches)
                errors.append(
                    "quarantine node ID assigned to multiple clusters "
                    f"({cluster_list}): {node_id}"
                )
    return ValidationReport(
        node_count=len(set(node_ids)),
        cluster_count=len(clusters),
        errors=tuple(errors),
    )


def removal_only_errors(
    current_node_ids: set[str], baseline_node_ids: set[str]
) -> tuple[str, ...]:
    """Reject additions while allowing the quarantine set to shrink."""

    return tuple(
        f"quarantine ledger added node ID: {node_id}"
        for node_id in sorted(current_node_ids - baseline_node_ids)
    )


def cluster_node_ids(
    ledger: dict[str, Any], configuration: dict[str, Any], cluster_id: str
) -> tuple[str, ...]:
    """Return the sorted node IDs assigned to exactly one named cluster."""

    report = validate_contract(ledger, configuration)
    if report.errors:
        raise ValueError("cannot select from an invalid quarantine contract")
    known = {cluster.get("cluster_id") for cluster in _clusters(configuration)}
    if cluster_id not in known:
        raise ValueError(f"unknown quarantine cluster: {cluster_id}")
    assignments = _cluster_assignments(ledger, configuration)
    return tuple(
        node_id for node_id, matches in assignments.items() if matches == (cluster_id,)
    )


def _baseline_node_ids(ref: str) -> set[str]:
    completed = subprocess.run(
        ["git", "show", f"{ref}:{LEDGER_REPO_PATH}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("baseline quarantine ledger must contain a JSON object")
    return set(_node_ids(payload))


def _run_cluster(node_ids: tuple[str, ...], batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return_code = 0
    for start in range(0, len(node_ids), batch_size):
        batch = node_ids[start : start + batch_size]
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *batch],
            cwd=REPO_ROOT,
            check=False,
        )
        return_code = max(return_code, completed.returncode)
    return return_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--cluster")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", dest="list_nodes")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the ledger and optionally list or run one failure cluster."""

    args = _parser().parse_args(argv)
    try:
        ledger = load_json(LEDGER_PATH)
        configuration = load_json(CLUSTER_PATH)
        report = validate_contract(ledger, configuration)
        errors = list(report.errors)
        if args.baseline_ref:
            errors.extend(
                removal_only_errors(
                    set(_node_ids(ledger)), _baseline_node_ids(args.baseline_ref)
                )
            )
        if errors:
            for error in errors:
                sys.stderr.write(f"unit-gate quarantine contract FAILED: {error}\n")
            return 1
        if args.cluster:
            node_ids = cluster_node_ids(ledger, configuration, args.cluster)
            if args.run:
                return _run_cluster(node_ids, args.batch_size)
            for node_id in node_ids:
                print(node_id)
            return 0
        if args.list_nodes or args.run:
            raise ValueError("--list and --run require --cluster")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"unit-gate quarantine contract FAILED: {exc}\n")
        return 2
    print(
        "Unit-gate quarantine contract passed: "
        f"{report.node_count} node IDs in {report.cluster_count} clusters."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
