"""Tests for physics engine lane guard and JUnit floor enforcement (issue #14948)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.check_engine_test_floor import main as cli_main
from src.shared.python.engine_core.engine_lane_guard import (
    enforce_required_engine_skip_guard,
    parse_required_engines,
    verify_junit_engine_floor,
)

pytestmark = pytest.mark.unit
FLOOR_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "scripts/config/engine_test_floor.json"
)


class _DummyMarker:
    def __init__(
        self,
        name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.name, self.args, self.kwargs = name, args, kwargs or {}


class _DummyItem:
    def __init__(self, nodeid: str, markers: list[_DummyMarker]) -> None:
        self.nodeid, self.markers = nodeid, markers

    def iter_markers(self) -> Any:
        return iter(self.markers)

    def get_closest_marker(self, name: str) -> _DummyMarker | None:
        return next((m for m in reversed(self.markers) if m.name == name), None)


class _DummyReport:
    def __init__(
        self,
        outcome: str = "skipped",
        longrepr: str = "Skipped",
        wasxfail: str | None = None,
    ) -> None:
        self.outcome = outcome
        self.skipped = outcome == "skipped"
        self.failed = outcome == "failed"
        self.passed = outcome == "passed"
        self.longrepr = longrepr
        if wasxfail:
            self.wasxfail = wasxfail


def test_oracle_1_required_engine_skipped_converts_to_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1. UD_REQUIRE_ENGINES=jaxsim with skipped requires_jaxsim test converts outcome to 'failed'."""
    monkeypatch.setenv("UD_REQUIRE_ENGINES", "jaxsim")
    rep = _DummyReport("skipped")
    item = _DummyItem("t::test_a", [_DummyMarker("requires_jaxsim")])
    enforce_required_engine_skip_guard(item, rep)
    assert rep.outcome == "failed"
    assert rep.failed and not rep.skipped
    assert "Engine lane guard (UD_REQUIRE_ENGINES)" in str(rep.longrepr)


def test_oracle_2_unset_required_engines_leaves_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2. UD_REQUIRE_ENGINES unset leaves a skipped requires_jaxsim test as 'skipped'."""
    monkeypatch.delenv("UD_REQUIRE_ENGINES", raising=False)
    rep = _DummyReport("skipped", "Skipped: uninstalled")
    item = _DummyItem("t::test_a", [_DummyMarker("requires_jaxsim")])
    enforce_required_engine_skip_guard(item, rep)
    assert rep.outcome == "skipped"
    assert rep.skipped and not rep.failed


def test_oracle_3_declared_skip_bypasses_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3. declared_skip(issue='#...', reason='...') permits skip under UD_REQUIRE_ENGINES."""
    monkeypatch.setenv("UD_REQUIRE_ENGINES", "mujoco")
    ok_kw = {"issue": "#14948", "reason": "ok"}
    bad_kw = {"issue": "14948", "reason": "bad"}
    cases = [
        ([], "failed"),
        ([_DummyMarker("declared_skip", kwargs=ok_kw)], "skipped"),
        ([_DummyMarker("declared_skip", args=("#14948", "ok"))], "skipped"),
        ([_DummyMarker("declared_skip", kwargs=bad_kw)], "failed"),
    ]
    for marks, exp in cases:
        rep = _DummyReport("skipped")
        item = _DummyItem("t", [_DummyMarker("requires_mujoco"), *marks])
        enforce_required_engine_skip_guard(item, rep)
        assert rep.outcome == exp


def test_oracle_4_zero_executed_fails_floor(tmp_path: Path) -> None:
    """4. JUnit XML with 0 executed fails the floor check."""
    junit = tmp_path / "empty.xml"
    junit.write_text(
        '<testsuites><testsuite name="p" tests="0"></testsuite></testsuites>',
        encoding="utf-8",
    )
    report = verify_junit_engine_floor(junit, FLOOR_CONFIG_PATH, "mujoco")
    assert not report.success and report.executed == 0


def test_oracle_5_xfail_vs_skipped_and_floor_config(tmp_path: Path) -> None:
    """5. Accurate xfail vs skip counting; engine_test_floor.json schema & not_covered validation."""
    junit = tmp_path / "results.xml"
    xml_content = (
        '<testsuites><testsuite name="p" tests="4">'
        '<testcase classname="p" name="t_pass" />'
        '<testcase classname="p" name="t_xfail"><skipped type="pytest.xfail" message="exp" /></testcase>'
        '<testcase classname="p" name="t_skip"><skipped type="pytest.skip" message="skip" /></testcase>'
        '<testcase classname="p" name="t_xp"><failure message="[XPASS(strict)] unexp" /></testcase>'
        "</testsuite></testsuites>"
    )
    junit.write_text(xml_content, encoding="utf-8")
    rep = verify_junit_engine_floor(junit, FLOOR_CONFIG_PATH, "pinocchio")
    counts = (rep.passed, rep.xfailed, rep.skipped, rep.failed, rep.xpassed_strict)
    assert counts == (1, 1, 1, 1, 1)
    assert rep.executed == 3 and not rep.success

    cfg = json.loads(FLOOR_CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(cfg["lanes"]) == {"mujoco", "pinocchio", "drake", "opensim", "jaxsim"}
    assert cfg["not_covered"]["matlab"]["issue"].startswith("#")
    assert cfg["not_covered"]["myosuite"]["issue"].startswith("#")
    with pytest.raises(ValueError, match="not_covered"):
        verify_junit_engine_floor(junit, FLOOR_CONFIG_PATH, "matlab")


def test_oracle_6_unknown_engine_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """6. Unknown engine name in UD_REQUIRE_ENGINES raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        parse_required_engines("mujuco")
    monkeypatch.setenv("UD_REQUIRE_ENGINES", "mujuco")
    with pytest.raises(ValueError, match="Unknown engine"):
        item = _DummyItem("t", [_DummyMarker("requires_mujoco")])
        enforce_required_engine_skip_guard(item, _DummyReport())


def test_xfail_report_not_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    """xfail tests carry wasxfail attribute and must not convert to failed."""
    monkeypatch.setenv("UD_REQUIRE_ENGINES", "mujoco")
    rep = _DummyReport("skipped", wasxfail="reason")
    enforce_required_engine_skip_guard(
        _DummyItem("t", [_DummyMarker("requires_mujoco")]), rep
    )
    assert rep.outcome == "skipped"


def test_cli_floor_checker(tmp_path: Path) -> None:
    """Test check_engine_test_floor.py CLI."""
    junit = tmp_path / "pass.xml"
    content = '<testsuites><testsuite name="p" tests="1"><testcase classname="p" name="t" /></testsuite></testsuites>'
    junit.write_text(content, encoding="utf-8")
    args = [
        "--junit",
        str(junit),
        "--config",
        str(FLOOR_CONFIG_PATH),
        "--lane",
        "mujoco",
    ]
    assert cli_main(args) == 0
