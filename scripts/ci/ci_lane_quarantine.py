#!/usr/bin/env python3
"""Fleet CI lane quarantine (RM #2252 / UD-CI-030).

When a required workflow lane is red for a known, owned regression, the ledger
records an expiry and expected failing node IDs. The lane still runs and reports
JUnit output; adjudication exits 0 only while the quarantine is valid and failures
match the ledger (no tolerance widening, no silent skip).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import defusedxml.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "scripts/config/ci_lane_quarantine.json"
ISO_DATE_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True)
class LaneQuarantine:
    """One quarantined workflow job."""

    workflow: str
    job: str
    reason: str
    expected_failure_node_ids: tuple[str, ...]
    issue: str
    owner: str
    expires: str

    def expires_date(self) -> dt.date:
        return dt.datetime.strptime(self.expires, ISO_DATE_FORMAT).date()

    def is_active(self, as_of: dt.date | None = None) -> bool:
        ref = as_of or dt.date.today()
        return self.expires_date() >= ref


def load_ledger(path: Path = LEDGER_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def parse_lane_entry(ledger: dict[str, Any], lane: dict[str, Any]) -> LaneQuarantine:
    workflow = lane.get("workflow")
    job = lane.get("job")
    reason = lane.get("reason")
    expected = lane.get("expected_failure_node_ids")
    if not isinstance(workflow, str) or not workflow.strip():
        raise ValueError("lane.workflow must be a non-empty string")
    if not isinstance(job, str) or not job.strip():
        raise ValueError("lane.job must be a non-empty string")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("lane.reason must be a non-empty string")
    if not isinstance(expected, list) or not expected:
        raise ValueError("lane.expected_failure_node_ids must be a non-empty list")
    node_ids = tuple(str(item) for item in expected)
    issue = str(ledger.get("issue", "")).strip()
    owner = str(ledger.get("owner", "")).strip()
    expires = str(ledger.get("expires", "")).strip()
    if not issue:
        raise ValueError("ledger.issue must be set (link to tracking issue/PR)")
    if not owner:
        raise ValueError("ledger.owner must be set")
    dt.datetime.strptime(expires, ISO_DATE_FORMAT)
    return LaneQuarantine(
        workflow=workflow,
        job=job,
        reason=reason,
        expected_failure_node_ids=node_ids,
        issue=issue,
        owner=owner,
        expires=expires,
    )


def find_lane(ledger: dict[str, Any], workflow: str, job: str) -> LaneQuarantine | None:
    lanes = ledger.get("lanes")
    if not isinstance(lanes, list):
        return None
    for raw in lanes:
        if not isinstance(raw, dict):
            continue
        entry = parse_lane_entry(ledger, raw)
        if entry.workflow == workflow and entry.job == job:
            return entry
    return None


def failed_node_ids_from_junit(junit_path: Path) -> tuple[str, ...]:
    if not junit_path.is_file():
        raise FileNotFoundError(junit_path)
    tree = ET.parse(junit_path)
    root = tree.getroot()
    failed: list[str] = []
    for testcase in root.iter("testcase"):
        classname = testcase.get("classname") or ""
        name = testcase.get("name") or ""
        if not name:
            continue
        if classname:
            module_path = classname.replace(".", "/")
            node_id = f"{module_path}.py::{name}"
        else:
            node_id = name
        if any(child.tag in ("failure", "error") for child in testcase):
            failed.append(node_id)
    return tuple(sorted(set(failed)))


def adjudicate_pytest_exit(
    *,
    pytest_exit_code: int,
    junit_path: Path,
    workflow: str,
    job: str,
    ledger_path: Path = LEDGER_PATH,
    as_of: dt.date | None = None,
) -> int:
    """Return process exit code for the workflow step after pytest.

    Postcondition: returns 0 only when pytest failed solely due to quarantined,
    expected node IDs while the ledger entry is still active.
    """
    if pytest_exit_code == 0:
        return 0
    try:
        ledger = load_ledger(ledger_path)
    except FileNotFoundError:
        # No ledger committed means no lane is quarantined; fail closed.
        return pytest_exit_code
    lane = find_lane(ledger, workflow, job)
    if lane is None:
        return pytest_exit_code
    if not lane.is_active(as_of):
        print(
            f"::error::CI lane quarantine for {workflow}#{job} expired "
            f"({lane.expires}); remove ledger or extend with evidence.",
            file=sys.stderr,
        )
        return pytest_exit_code
    try:
        failed = failed_node_ids_from_junit(junit_path)
    except OSError as exc:
        print(f"::error::Could not read junit for quarantine adjudication: {exc}")
        return pytest_exit_code
    expected = set(lane.expected_failure_node_ids)
    failed_set = set(failed)
    if not failed_set:
        print(
            "::error::pytest failed but junit reports no failing testcases; "
            "quarantine cannot mask infra errors.",
            file=sys.stderr,
        )
        return pytest_exit_code
    unexpected = failed_set - expected
    if unexpected:
        print(
            "::error::CI lane quarantine rejected: unexpected failures "
            f"{sorted(unexpected)} (expected only {sorted(expected)}).",
            file=sys.stderr,
        )
        return pytest_exit_code
    missing = expected - failed_set
    if missing:
        print(
            "::warning::CI lane quarantine: some ledger failures absent this run "
            f"(partial green): {sorted(missing)}",
            file=sys.stderr,
        )
    print(
        f"::warning::CI lane quarantine active until {lane.expires} "
        f"(owner={lane.owner}, issue={lane.issue}). "
        f"Recorded failures: {sorted(failed_set)}. Job reports success; "
        "fix follow-up required before expiry.",
        file=sys.stderr,
    )
    return 0


def validate_ledger(path: Path = LEDGER_PATH) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        ledger = load_ledger(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (str(exc),)
    schema = ledger.get("schema_version")
    if schema != "ci-lane-quarantine/v1":
        errors.append(f"unsupported schema_version: {schema!r}")
    lanes = ledger.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        errors.append("lanes must be a non-empty list")
    else:
        for index, raw in enumerate(lanes):
            if not isinstance(raw, dict):
                errors.append(f"lanes[{index}] must be an object")
                continue
            try:
                parse_lane_entry(ledger, raw)
            except ValueError as exc:
                errors.append(f"lanes[{index}]: {exc}")
    return tuple(errors)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CI lane quarantine adjudication")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ledger schema and exit",
    )
    parser.add_argument(
        "--workflow", help="Workflow file name, e.g. ci-requires-opensim.yml"
    )
    parser.add_argument("--job", help="Job id within the workflow")
    parser.add_argument("--pytest-exit-code", type=int, default=0)
    parser.add_argument("--junitxml", type=Path, help="Pytest JUnit XML path")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    args = parser.parse_args(argv)
    if args.validate:
        errors = validate_ledger(args.ledger)
        if errors:
            for err in errors:
                print(f"ci_lane_quarantine: {err}", file=sys.stderr)
            return 1
        print("ci_lane_quarantine: ledger valid")
        return 0
    if not args.workflow or not args.job or args.junitxml is None:
        parser.error("--workflow, --job, and --junitxml are required unless --validate")
    return adjudicate_pytest_exit(
        pytest_exit_code=args.pytest_exit_code,
        junit_path=args.junitxml,
        workflow=args.workflow,
        job=args.job,
        ledger_path=args.ledger,
    )


if __name__ == "__main__":
    raise SystemExit(_main())
