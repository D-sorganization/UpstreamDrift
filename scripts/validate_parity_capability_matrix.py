#!/usr/bin/env python3
"""Validate the #15794 B1 parity matrix, tolerances, and approval changelog.

The generated MS-71 ledger ``src/config/engine_capability_matrix.json`` stays
the qualification profile. This gate reads the hand-authored schema v2 files:

- ``src/config/engine_capability_matrix.v2.json``
- ``src/config/parity_tolerances.yaml``
- ``src/config/parity_capability_changelog.yaml``

Precondition: each document is a mapping with ``schema_version`` ``2.0.0``.
Postcondition: the return value is a list of ``code:detail`` strings, empty
when the package meets the epic gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "src" / "config"
MATRIX_PATH = CONFIG_DIR / "engine_capability_matrix.v2.json"
TOLERANCES_PATH = CONFIG_DIR / "parity_tolerances.yaml"
CHANGELOG_PATH = CONFIG_DIR / "parity_capability_changelog.yaml"

SCHEMA_VERSION = "2.0.0"
PARITY_ENGINES = frozenset(
    {"drake", "mujoco", "myosuite", "opensim", "pinocchio", "simscape"}
)
STATUS_RANK = {"verified": 2, "claimed": 1, "unsupported": 0}


def _finding(code: str, detail: str) -> str:
    if not code.strip() or not detail.strip():
        raise ValueError("findings need a code and a detail")
    return f"{code}:{detail}"


def _as_mapping(value: Any, label: str) -> tuple[Mapping[str, Any] | None, list[str]]:
    if isinstance(value, dict):
        return value, []
    return None, [_finding("schema_version", f"{label} must be a mapping")]


def _require_schema(document: Mapping[str, Any], label: str) -> list[str]:
    if document.get("schema_version") != SCHEMA_VERSION:
        return [_finding("schema_version", f"{label} schema_version must be 2.0.0")]
    return []


def _load_yaml(path: Path) -> tuple[Any, list[str]]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), []
    except (OSError, yaml.YAMLError) as exc:
        return None, [_finding("unreadable", f"{path.name}: {exc}")]


def _load_json(path: Path) -> tuple[Any, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, json.JSONDecodeError) as exc:
        return None, [_finding("unreadable", f"{path.name}: {exc}")]


def _approved(entries: Any, **need: object) -> bool:
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("codeowners_approval") is not True:
            continue
        if all(entry.get(key) == value for key, value in need.items()):
            return True
    return False


def _tolerance_ids(tolerances: Mapping[str, Any]) -> set[str]:
    regimes = tolerances.get("regimes")
    if not isinstance(regimes, dict):
        return set()
    return {str(key) for key in regimes}


def _bounds(tolerances: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    regimes = tolerances.get("regimes")
    parsed: dict[str, dict[str, float]] = {}
    if not isinstance(regimes, dict):
        return parsed
    for regime, body in regimes.items():
        if not isinstance(body, dict) or not isinstance(body.get("bounds"), dict):
            continue
        maxima: dict[str, float] = {}
        for name, spec in body["bounds"].items():
            if isinstance(spec, dict) and isinstance(spec.get("max"), (int, float)):
                maxima[str(name)] = float(spec["max"])
        parsed[str(regime)] = maxima
    return parsed


def _check_tolerances(tolerances: Mapping[str, Any]) -> list[str]:
    findings = _require_schema(tolerances, "tolerances")
    regimes = tolerances.get("regimes")
    if not isinstance(regimes, dict) or not regimes:
        findings.append(_finding("tolerances", "regimes must be a non-empty mapping"))
        return findings
    for regime, body in regimes.items():
        if not isinstance(body, dict) or not isinstance(body.get("bounds"), dict):
            findings.append(_finding("tolerances", f"{regime} needs a bounds mapping"))
            continue
        if not body["bounds"]:
            findings.append(
                _finding("tolerances", f"{regime} needs at least one bound")
            )
        for name, spec in body["bounds"].items():
            if not isinstance(spec, dict) or isinstance(spec.get("max"), bool):
                findings.append(_finding("tolerances", f"{regime}.{name} needs max"))
                continue
            if not isinstance(spec.get("max"), (int, float)):
                findings.append(
                    _finding("tolerances", f"{regime}.{name} max must be numeric")
                )
    return findings


def _check_engines(matrix: Mapping[str, Any]) -> list[str]:
    engines = matrix.get("engines")
    if not isinstance(engines, list) or any(
        not isinstance(item, str) for item in engines
    ):
        return [_finding("engines", "engines must list the six parity engines")]
    if len(engines) != len(set(engines)):
        return [_finding("duplicate_engine", "engines must be unique")]
    if set(engines) != PARITY_ENGINES:
        return [_finding("engines", "engines must be exactly the six parity engines")]
    return []


def _check_cell(
    capability: str,
    engine: str,
    cell: Any,
    *,
    known_tolerances: set[str],
    capability_tolerance: str,
    release_cycle: int,
    claim_expiry_cycles: int,
) -> list[str]:
    detail = f"{capability}.{engine}"
    if not isinstance(cell, dict):
        return [_finding("cell", f"{detail} must be a mapping")]
    status = cell.get("status")
    if status not in STATUS_RANK:
        return [
            _finding(
                "status", f"{detail} status must be verified, claimed, or unsupported"
            )
        ]
    findings: list[str] = []
    if status == "verified":
        fixture = cell.get("fixture_id")
        tolerance_id = cell.get("tolerance_id")
        if not isinstance(fixture, str) or not fixture.strip():
            findings.append(_finding("verified_missing_fixture", detail))
        if tolerance_id != capability_tolerance or tolerance_id not in known_tolerances:
            findings.append(_finding("unknown_tolerance", detail))
        return findings
    if status == "unsupported":
        issue = cell.get("issue")
        if isinstance(issue, bool) or not isinstance(issue, int) or issue <= 0:
            findings.append(_finding("unsupported_missing_issue", detail))
        return findings
    claimed_cycle = cell.get("claimed_cycle")
    if isinstance(claimed_cycle, bool) or not isinstance(claimed_cycle, int):
        findings.append(_finding("claimed_cycle", detail))
        return findings
    fixture = cell.get("fixture_id")
    has_fixture = isinstance(fixture, str) and bool(fixture.strip())
    if not has_fixture and release_cycle - claimed_cycle >= claim_expiry_cycles:
        findings.append(_finding("claim_expired", detail))
    return findings


def _check_capabilities(
    matrix: Mapping[str, Any], tolerances: Mapping[str, Any]
) -> list[str]:
    findings: list[str] = []
    release_cycle = matrix.get("release_cycle")
    expiry = matrix.get("claim_expiry_cycles")
    if isinstance(release_cycle, bool) or not isinstance(release_cycle, int):
        return [_finding("release_cycle", "release_cycle must be an integer")]
    if isinstance(expiry, bool) or not isinstance(expiry, int) or expiry <= 0:
        return [
            _finding(
                "claim_expiry_cycles", "claim_expiry_cycles must be a positive integer"
            )
        ]
    capabilities = matrix.get("capabilities")
    if not isinstance(capabilities, dict) or not capabilities:
        return [_finding("capabilities", "capabilities must be a non-empty mapping")]
    known = _tolerance_ids(tolerances)
    for name, body in capabilities.items():
        if not isinstance(body, dict):
            findings.append(_finding("capabilities", f"{name} must be a mapping"))
            continue
        tolerance_id = body.get("tolerance_id")
        if not isinstance(tolerance_id, str) or tolerance_id not in known:
            findings.append(_finding("unknown_tolerance", str(name)))
            tolerance_id = ""
        cells = body.get("cells")
        if not isinstance(cells, dict):
            findings.append(_finding("missing_engine", str(name)))
            continue
        for engine in sorted(PARITY_ENGINES):
            if engine not in cells:
                findings.append(_finding("missing_engine", f"{name}.{engine}"))
                continue
            findings.extend(
                _check_cell(
                    str(name),
                    engine,
                    cells[engine],
                    known_tolerances=known,
                    capability_tolerance=tolerance_id,
                    release_cycle=release_cycle,
                    claim_expiry_cycles=expiry,
                )
            )
        for engine in cells:
            if engine not in PARITY_ENGINES:
                findings.append(_finding("unknown_engine", f"{name}.{engine}"))
    return findings


def _check_drift(
    matrix: Mapping[str, Any],
    tolerances: Mapping[str, Any],
    changelog: Mapping[str, Any],
    baseline_matrix: Mapping[str, Any],
    baseline_tolerances: Mapping[str, Any],
) -> list[str]:
    entries = changelog.get("entries")
    findings: list[str] = []
    current_caps = matrix.get("capabilities")
    previous_caps = baseline_matrix.get("capabilities")
    if isinstance(current_caps, dict) and isinstance(previous_caps, dict):
        for name, previous_body in previous_caps.items():
            current_body = current_caps.get(name)
            if not isinstance(previous_body, dict) or not isinstance(
                current_body, dict
            ):
                continue
            previous_cells = previous_body.get("cells")
            current_cells = current_body.get("cells")
            if not isinstance(previous_cells, dict) or not isinstance(
                current_cells, dict
            ):
                continue
            for engine, previous_cell in previous_cells.items():
                current_cell = current_cells.get(engine)
                if not isinstance(previous_cell, dict) or not isinstance(
                    current_cell, dict
                ):
                    continue
                previous_status = previous_cell.get("status")
                current_status = current_cell.get("status")
                if (
                    previous_status not in STATUS_RANK
                    or current_status not in STATUS_RANK
                ):
                    continue
                if STATUS_RANK[current_status] >= STATUS_RANK[previous_status]:
                    continue
                if _approved(
                    entries,
                    kind="status_downgrade",
                    capability=name,
                    engine=engine,
                ):
                    continue
                findings.append(_finding("status_downgraded", f"{name}.{engine}"))
    current_bounds = _bounds(tolerances)
    previous_bounds = _bounds(baseline_tolerances)
    for regime, old_bounds in previous_bounds.items():
        new_bounds = current_bounds.get(regime, {})
        for bound, old_max in old_bounds.items():
            new_max = new_bounds.get(bound)
            if new_max is not None and new_max <= old_max:
                continue
            if _approved(
                entries,
                kind="tolerance_loosening",
                regime=regime,
                bound=bound,
            ):
                continue
            findings.append(_finding("tolerance_loosened", f"{regime}.{bound}"))
    return findings


def validate_matrix(
    matrix: Any,
    tolerances: Any,
    changelog: Any,
    *,
    baseline_matrix: Any = None,
    baseline_tolerances: Any = None,
) -> list[str]:
    """Return gate findings. Empty means the package may ship."""
    findings: list[str] = []
    matrix_map, matrix_errors = _as_mapping(matrix, "matrix")
    tolerances_map, tolerance_errors = _as_mapping(tolerances, "tolerances")
    changelog_map, changelog_errors = _as_mapping(changelog, "changelog")
    findings.extend(matrix_errors + tolerance_errors + changelog_errors)
    if matrix_map is None or tolerances_map is None or changelog_map is None:
        return findings
    findings.extend(_require_schema(matrix_map, "matrix"))
    findings.extend(_require_schema(changelog_map, "changelog"))
    findings.extend(_check_tolerances(tolerances_map))
    findings.extend(_check_engines(matrix_map))
    findings.extend(_check_capabilities(matrix_map, tolerances_map))
    if baseline_matrix is not None or baseline_tolerances is not None:
        base_matrix, base_matrix_errors = _as_mapping(
            baseline_matrix, "baseline matrix"
        )
        base_tolerances, base_tolerance_errors = _as_mapping(
            baseline_tolerances, "baseline tolerances"
        )
        findings.extend(base_matrix_errors + base_tolerance_errors)
        if base_matrix is not None and base_tolerances is not None:
            findings.extend(
                _check_drift(
                    matrix_map,
                    tolerances_map,
                    changelog_map,
                    base_matrix,
                    base_tolerances,
                )
            )
    if any(not isinstance(item, str) or not item.strip() for item in findings):
        raise RuntimeError("validator findings must be non-empty strings")
    return findings


def validate_paths(
    matrix_path: Path = MATRIX_PATH,
    tolerances_path: Path = TOLERANCES_PATH,
    changelog_path: Path = CHANGELOG_PATH,
) -> list[str]:
    """Load the three committed documents and validate them."""
    matrix, matrix_errors = _load_json(matrix_path)
    tolerances, tolerance_errors = _load_yaml(tolerances_path)
    changelog, changelog_errors = _load_yaml(changelog_path)
    findings = matrix_errors + tolerance_errors + changelog_errors
    if findings:
        return findings
    return validate_matrix(matrix, tolerances, changelog)


def main(argv: list[str] | None = None) -> int:
    """Exit 0 when the parity package passes, else 1."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--tolerances", type=Path, default=TOLERANCES_PATH)
    parser.add_argument("--changelog", type=Path, default=CHANGELOG_PATH)
    args = parser.parse_args(argv)
    findings = validate_paths(args.matrix, args.tolerances, args.changelog)
    for finding in findings:
        print(finding, file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
