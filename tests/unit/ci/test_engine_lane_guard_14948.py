"""Unit tests for engine lane guard, floor check CLI, and dead job removal (#14948)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.ci.check_engine_test_floor import main as check_floor_main
from src.shared.python.engine_core.engine_lane_guard import (
    enforce_required_engines_on_report,
    verify_junit_engine_floor,
)
from _pytest.reports import TestReport

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FLOOR_CONFIG_PATH = REPO_ROOT / "scripts" / "config" / "engine_test_floor.json"
CI_STANDARD_PATH = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
NIGHTLY_CROSS_ENGINE_PATH = (
    REPO_ROOT / ".github" / "workflows" / "nightly-cross-engine.yml"
)


def test_ci_standard_matlab_job_removed() -> None:
    """Verify dead matlab-tests job is removed from ci-standard.yml (#14948)."""
    assert CI_STANDARD_PATH.exists()
    content = CI_STANDARD_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(content)
    jobs = workflow.get("jobs", {})

    msg = "matlab-tests job must be removed from ci-standard.yml"
    assert "matlab-tests" not in jobs, msg

    # Verify no job in ci-standard has literal && (false) condition for matlab
    for job_name, job in jobs.items():
        if_cond = str(job.get("if", ""))
        has_matlab_false = "matlab" in job_name.lower() and "(false)" in if_cond
        assert not has_matlab_false, f"Job '{job_name}' has (false): {if_cond}"


def test_engine_test_floor_documents_not_covered_engines() -> None:
    """Verify engine_test_floor.json explicitly records matlab and myosuite as not_covered."""
    assert FLOOR_CONFIG_PATH.exists()
    config = json.loads(FLOOR_CONFIG_PATH.read_text(encoding="utf-8"))

    not_covered = config.get("not_covered", {})
    assert "matlab" in not_covered
    assert "myosuite" in not_covered

    matlab_info = not_covered["matlab"]
    assert matlab_info.get("reason"), "matlab must have a non-empty reason"
    assert (
        "license" in matlab_info["reason"].lower()
        or "ci" in matlab_info["reason"].lower()
    )

    myosuite_info = not_covered["myosuite"]
    assert myosuite_info.get("reason"), "myosuite must have a non-empty reason"

    # verify_junit_engine_floor must raise ValueError on not_covered lanes
    dummy_xml = REPO_ROOT / "tests" / "fixtures" / "dummy.xml"
    with pytest.raises(ValueError, match="not_covered"):
        verify_junit_engine_floor(dummy_xml, FLOOR_CONFIG_PATH, "matlab")

    with pytest.raises(ValueError, match="not_covered"):
        verify_junit_engine_floor(dummy_xml, FLOOR_CONFIG_PATH, "myosuite")


def test_nightly_cross_engine_sets_ud_require_engines_and_floor_check() -> None:
    """Verify nightly-cross-engine.yml sets UD_REQUIRE_ENGINES and invokes check_engine_test_floor.py."""
    assert NIGHTLY_CROSS_ENGINE_PATH.exists()
    content = NIGHTLY_CROSS_ENGINE_PATH.read_text(encoding="utf-8")

    assert 'UD_REQUIRE_ENGINES: "drake,pinocchio,opensim"' in content
    assert "scripts/ci/check_engine_test_floor.py" in content
    assert "--engine pinocchio" in content or "--lane pinocchio" in content
    assert "/tmp/nightly-cross-engine-junit.xml" in content


def test_ci_standard_tests_job_sets_ud_require_engines_mujoco() -> None:
    """Verify ci-standard.yml core test step sets UD_REQUIRE_ENGINES: mujoco."""
    assert CI_STANDARD_PATH.exists()
    content = CI_STANDARD_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(content)

    tests_job = workflow.get("jobs", {}).get("tests", {})
    steps = tests_job.get("steps", [])

    core_step = next((s for s in steps if s.get("id") == "core-tests"), None)
    assert core_step is not None, "core-tests step must exist in tests job"

    env = core_step.get("env", {})
    assert env.get("UD_REQUIRE_ENGINES") == "mujoco"


def test_check_engine_test_floor_cli_aliases(tmp_path: Path) -> None:
    """Test check_engine_test_floor.py CLI with --engine, --junit-xml, and default config."""
    junit_path = tmp_path / "nightly-junit.xml"
    junit_path.write_text(
        '<testsuites><testsuite name="pinocchio" tests="2">'
        '<testcase classname="pinocchio" name="test_forward_kinematics" />'
        '<testcase classname="pinocchio" name="test_inverse_dynamics" />'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )

    args = [
        "--engine",
        "pinocchio",
        "--junit-xml",
        str(junit_path),
        "--min-executed",
        "1",
    ]
    assert check_floor_main(args) == 0

    # min_executed failure test
    fail_args = [
        "--engine",
        "pinocchio",
        "--junit-xml",
        str(junit_path),
        "--min-executed",
        "5",
    ]
    assert check_floor_main(fail_args) == 1


def test_enforce_required_engines_on_report_with_real_test_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify enforce_required_engines_on_report with a real _pytest.reports.TestReport."""
    monkeypatch.setenv("UD_REQUIRE_ENGINES", "pinocchio,drake")
    report = TestReport(
        nodeid="tests/unit/test_pinocchio_kinematics.py::test_fk",
        location=("tests/unit/test_pinocchio_kinematics.py", 10, "test_fk"),
        keywords={},
        outcome="skipped",
        longrepr=(
            "tests/unit/test_pinocchio_kinematics.py",
            10,
            "Skipped: pinocchio missing",
        ),
        when="call",
    )
    assert report.skipped is True
    assert report.failed is False

    enforce_required_engines_on_report(report, nodeid=report.nodeid)

    assert report.outcome == "failed"
    assert report.skipped is False
    assert report.failed is True
    assert "Engine lane guard (UD_REQUIRE_ENGINES)" in str(report.longrepr)
    assert "pinocchio" in str(report.longrepr)


def test_enforce_required_engines_ignores_unrelated_or_declared_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-required engine tests or declared skips are not turned into failures."""
    monkeypatch.setenv("UD_REQUIRE_ENGINES", "pinocchio")
    unrelated_report = TestReport(
        nodeid="tests/unit/test_general.py::test_math",
        location=("tests/unit/test_general.py", 5, "test_math"),
        keywords={},
        outcome="skipped",
        longrepr=("tests/unit/test_general.py", 5, "Skipped: optional test"),
        when="call",
    )
    enforce_required_engines_on_report(unrelated_report, nodeid=unrelated_report.nodeid)
    assert unrelated_report.outcome == "skipped"
    assert unrelated_report.skipped is True

    class DummyMarker:
        name = "declared_skip"
        args = ("#14948", "wip")
        kwargs = {"issue": "#14948", "reason": "wip"}

    class DummyItem:
        nodeid = "tests/unit/test_pinocchio_fk.py::test_fk"

        def get_closest_marker(self, name: str) -> DummyMarker | None:
            return DummyMarker() if name == "declared_skip" else None

    declared_report = TestReport(
        nodeid="tests/unit/test_pinocchio_fk.py::test_fk",
        location=("tests/unit/test_pinocchio_fk.py", 5, "test_fk"),
        keywords={},
        outcome="skipped",
        longrepr=("tests/unit/test_pinocchio_fk.py", 5, "Skipped: #14948 wip"),
        when="call",
    )
    enforce_required_engines_on_report(
        declared_report, nodeid=declared_report.nodeid, item=DummyItem()
    )
    assert declared_report.outcome == "skipped"
    assert declared_report.skipped is True


def test_verify_junit_engine_floor_fails_closed_on_failure(tmp_path: Path) -> None:
    """Verify verify_junit_engine_floor fails closed when failures/errors > 0."""
    junit_path = tmp_path / "failed-junit.xml"
    junit_path.write_text(
        '<testsuites><testsuite name="pinocchio" tests="2">'
        '<testcase classname="pinocchio" name="test_fk" />'
        '<testcase classname="pinocchio" name="test_dynamics">'
        '<failure message="assertion error">Traceback</failure>'
        "</testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    result = verify_junit_engine_floor(junit_path, FLOOR_CONFIG_PATH, "pinocchio")
    assert result.success is False
    assert result.failed == 1
    assert result.f == 1
    assert result.p == 1
    assert "had 1 failed/errored test(s)" in result.message


def test_verify_junit_engine_floor_fails_closed_when_all_skipped(
    tmp_path: Path,
) -> None:
    """Verify verify_junit_engine_floor fails closed when 0 passed and skipped > 0."""
    junit_path = tmp_path / "skipped-junit.xml"
    junit_path.write_text(
        '<testsuites><testsuite name="pinocchio" tests="2">'
        '<testcase classname="pinocchio" name="test_fk"><skipped message="skip" /></testcase>'
        '<testcase classname="pinocchio" name="test_ik"><skipped message="skip" /></testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    result = verify_junit_engine_floor(junit_path, FLOOR_CONFIG_PATH, "pinocchio")
    assert result.success is False
    assert result.p == 0
    assert result.s == 2
    assert "All 2 test(s) for engine 'pinocchio' were skipped" in result.message


def test_verify_junit_engine_floor_multi_engine_filtering(tmp_path: Path) -> None:
    """Verify multi-engine JUnit XML filters testcases so non-engine tests are not credited."""
    junit_path = tmp_path / "multi-engine-junit.xml"
    junit_path.write_text(
        "<testsuites>"
        '<testsuite name="nightly-cross-engine">'
        '<testcase classname="pinocchio" name="test_fk" />'
        '<testcase classname="drake" name="test_ik" />'
        '<testcase classname="mujoco" name="test_step">'
        '<failure message="mujoco error">Traceback</failure>'
        "</testcase>"
        "</testsuite>"
        "</testsuites>",
        encoding="utf-8",
    )
    pinocchio_res = verify_junit_engine_floor(
        junit_path, FLOOR_CONFIG_PATH, "pinocchio"
    )
    assert pinocchio_res.success is True
    assert pinocchio_res.p == 1
    assert pinocchio_res.f == 0

    drake_res = verify_junit_engine_floor(junit_path, FLOOR_CONFIG_PATH, "drake")
    assert drake_res.success is True
    assert drake_res.p == 1
    assert drake_res.f == 0

    mujoco_res = verify_junit_engine_floor(junit_path, FLOOR_CONFIG_PATH, "mujoco")
    assert mujoco_res.success is False
    assert mujoco_res.p == 0
    assert mujoco_res.f == 1
