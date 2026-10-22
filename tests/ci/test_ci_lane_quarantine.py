"""Contract tests for CI lane quarantine (RM #2252 / #13568)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.ci.ci_lane_quarantine import (
    adjudicate_pytest_exit,
    load_ledger,
    validate_ledger,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "scripts/config/ci_lane_quarantine.json"

# From main run 39696833816 failed log (2026-10-21).
SQUAT_NODE = (
    "tests/unit/lifting/test_opensim_lift_joint_reactions.py"
    "::test_squat_joint_reactions_computed"
)
NATIVE_NODE = (
    "tests/opensim/test_native_muscle_effort_native.py"
    "::test_native_constant_interval_matches_fresh_raw_replay"
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


class TestLedgerContract:
    def test_committed_ledger_validates(self) -> None:
        assert validate_ledger(LEDGER) == ()

    def test_ledger_links_pr_13568_with_expiry(self) -> None:
        ledger = load_ledger(LEDGER)
        assert "13568" in str(ledger["issue"])
        assert ledger["expires"] == "2026-10-29"
        lane = ledger["lanes"][0]
        assert lane["workflow"] == "ci-requires-opensim.yml"
        assert lane["job"] == "requires-opensim"
        assert SQUAT_NODE in lane["expected_failure_node_ids"]
        assert NATIVE_NODE in lane["expected_failure_node_ids"]


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
                workflow="ci-requires-opensim.yml",
                job="requires-opensim",
                ledger_path=LEDGER,
            )
            == 0
        )

    def test_dual_failure_from_main_log_is_quarantined(self, tmp_path: Path) -> None:
        junit = tmp_path / "results.xml"
        junit.write_text(
            _junit_with_failures(SQUAT_NODE, NATIVE_NODE),
            encoding="utf-8",
        )
        rc = adjudicate_pytest_exit(
            pytest_exit_code=1,
            junit_path=junit,
            workflow="ci-requires-opensim.yml",
            job="requires-opensim",
            ledger_path=LEDGER,
            as_of=date(2026, 10, 22),
        )
        assert rc == 0

    def test_expired_quarantine_fails_closed(self, tmp_path: Path) -> None:
        junit = tmp_path / "results.xml"
        junit.write_text(_junit_with_failures(SQUAT_NODE), encoding="utf-8")
        rc = adjudicate_pytest_exit(
            pytest_exit_code=1,
            junit_path=junit,
            workflow="ci-requires-opensim.yml",
            job="requires-opensim",
            ledger_path=LEDGER,
            as_of=date(2026, 11, 1),
        )
        assert rc == 1

    def test_unexpected_failure_rejects_quarantine(self, tmp_path: Path) -> None:
        junit = tmp_path / "results.xml"
        junit.write_text(
            _junit_with_failures(
                SQUAT_NODE,
                "tests/unit/lifting/test_opensim_lift_muscles.py::test_other",
            ),
            encoding="utf-8",
        )
        rc = adjudicate_pytest_exit(
            pytest_exit_code=1,
            junit_path=junit,
            workflow="ci-requires-opensim.yml",
            job="requires-opensim",
            ledger_path=LEDGER,
            as_of=date(2026, 10, 22),
        )
        assert rc == 1
