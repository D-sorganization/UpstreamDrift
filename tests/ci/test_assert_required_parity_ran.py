"""Unit tests for the required-parity JUnit assertion helper (#6881).

These prove a report with only skipped JaxSim parity cases is rejected
(non-zero), while a report with at least one passing case is accepted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "ci" / "assert_required_parity_ran.py"

_spec = importlib.util.spec_from_file_location("assert_required_parity_ran", _SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ParityGateError = _mod.ParityGateError
assert_parity_ran = _mod.assert_parity_ran
main = _mod.main

_REQUIRED = "test_jaxsim_pinocchio_free_body_dynamics_terms_match"


def _write_junit(path: Path, *, body: str) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f"<testsuites><testsuite>{body}</testsuite></testsuites>\n",
        encoding="utf-8",
    )


def test_all_skipped_required_cases_fail(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    _write_junit(
        junit,
        body=(
            f'<testcase classname="tests.cross_engine.test_jaxsim_vs_pinocchio" '
            f'name="{_REQUIRED}[case0]"><skipped message="no jax"/></testcase>'
            f'<testcase classname="tests.cross_engine.test_jaxsim_vs_pinocchio" '
            f'name="{_REQUIRED}[case1]"><skipped message="no jax"/></testcase>'
        ),
    )
    with pytest.raises(ParityGateError):
        assert_parity_ran(junit, (_REQUIRED,))
    assert main(["--junit", str(junit), "--require-name", _REQUIRED]) == 1


def test_passing_required_case_succeeds(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    _write_junit(
        junit,
        body=(
            f'<testcase classname="tests.cross_engine.test_jaxsim_vs_pinocchio" '
            f'name="{_REQUIRED}[case0]"/>'
            f'<testcase classname="tests.cross_engine.test_jaxsim_vs_pinocchio" '
            f'name="{_REQUIRED}[case1]"><skipped/></testcase>'
        ),
    )
    counts = assert_parity_ran(junit, (_REQUIRED,))
    assert counts["passed"] == 1
    assert main(["--junit", str(junit), "--require-name", _REQUIRED]) == 0


def test_absent_required_case_fails(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    _write_junit(
        junit,
        body='<testcase classname="other" name="test_unrelated"/>',
    )
    with pytest.raises(ParityGateError):
        assert_parity_ran(junit, (_REQUIRED,))


def test_missing_report_fails(tmp_path: Path) -> None:
    assert (
        main(["--junit", str(tmp_path / "nope.xml"), "--require-name", _REQUIRED]) == 1
    )
