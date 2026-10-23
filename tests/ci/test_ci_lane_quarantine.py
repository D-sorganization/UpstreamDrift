"""Contract tests for CI lane quarantine (RM #2252 / UD-CI-030)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.ci.ci_lane_quarantine import (
    adjudicate_pytest_exit,
    validate_ledger,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "scripts/config/ci_lane_quarantine.json"

SAMPLE_NODE = "tests/unit/example/test_mod.py::test_thing"
OTHER_NODE = "tests/unit/example/test_mod.py::test_other"


def _write_ledger(
    path: Path,
    *,
    expires: str = "2026-10-29",
    issue: str = "https://github.com/D-sorganization/UpstreamDrift/pull/1",
    owner: str = "owner",
    node_ids: tuple[str, ...] = (SAMPLE_NODE,),
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ci-lane-quarantine/v1",
                "issue": issue,
                "owner": owner,
                "expires": expires,
                "lanes": [
                    {
                        "workflow": "example.yml",
                        "job": "example-job",
                        "reason": "test fixture",
                        "expected_failure_node_ids": list(node_ids),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _junit_with_failures(*node_ids: str) -> str:
    cases = []
    for node_id in node_ids:
        module, test_name = node_id.split("::", maxsplit=1)
        classname = module.removesuffix(".py").replace("/", ".")
        cases.append(
            f'  <testcase classname="{classname}" name="{test_name}">'
            f"<failure message='assert False'/></testcase>"
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuite tests="{len(cases)}">{"".join(cases)}</testsuite>'
    )


class TestNoActiveQuarantine:
    """UpstreamDrift currently ships no active lane quarantine (#13564 fixed on main)."""

    def test_no_ledger_is_committed(self) -> None:
        assert not LEDGER.exists()


class TestAdjudicatePytestExit:
    def test_pytest_success_never_quarantines(self, tmp_path: Path) -> None:
        junit = tmp_path / "results.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuite tests="0"></testsuite>',
            encoding="utf-8",
        )
        assert (
            adjudicate_pytest_exit(
                pytest_exit_code=0,
                junit_path=junit,
                workflow="example.yml",
                job="example-job",
                ledger_path=tmp_path / "missing-ledger.json",
            )
            == 0
        )

    def test_missing_ledger_fails_closed(self, tmp_path: Path) -> None:
        """No committed ledger means no lane is quarantined: pytest's failure stands."""
        junit = tmp_path / "results.xml"
        junit.write_text(_junit_with_failures(SAMPLE_NODE), encoding="utf-8")
        rc = adjudicate_pytest_exit(
            pytest_exit_code=1,
            junit_path=junit,
            workflow="example.yml",
            job="example-job",
            ledger_path=tmp_path / "missing-ledger.json",
        )
        assert rc == 1

    def test_matching_failure_is_quarantined(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.json"
        _write_ledger(ledger_path, node_ids=(SAMPLE_NODE, OTHER_NODE))
        junit = tmp_path / "results.xml"
        junit.write_text(
            _junit_with_failures(SAMPLE_NODE, OTHER_NODE), encoding="utf-8"
        )
        rc = adjudicate_pytest_exit(
            pytest_exit_code=1,
            junit_path=junit,
            workflow="example.yml",
            job="example-job",
            ledger_path=ledger_path,
            as_of=date(2026, 10, 22),
        )
        assert rc == 0

    def test_expired_quarantine_fails_closed(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.json"
        _write_ledger(ledger_path)
        junit = tmp_path / "results.xml"
        junit.write_text(_junit_with_failures(SAMPLE_NODE), encoding="utf-8")
        rc = adjudicate_pytest_exit(
            pytest_exit_code=1,
            junit_path=junit,
            workflow="example.yml",
            job="example-job",
            ledger_path=ledger_path,
            as_of=date(2026, 11, 1),
        )
        assert rc == 1

    def test_unexpected_failure_rejects_quarantine(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.json"
        _write_ledger(ledger_path)
        junit = tmp_path / "results.xml"
        junit.write_text(
            _junit_with_failures(SAMPLE_NODE, OTHER_NODE), encoding="utf-8"
        )
        rc = adjudicate_pytest_exit(
            pytest_exit_code=1,
            junit_path=junit,
            workflow="example.yml",
            job="example-job",
            ledger_path=ledger_path,
            as_of=date(2026, 10, 22),
        )
        assert rc == 1


class TestValidateLedger:
    def test_missing_ledger_reports_error(self, tmp_path: Path) -> None:
        errors = validate_ledger(tmp_path / "missing-ledger.json")
        assert errors

    def test_well_formed_ledger_validates(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.json"
        _write_ledger(ledger_path)
        assert validate_ledger(ledger_path) == ()
