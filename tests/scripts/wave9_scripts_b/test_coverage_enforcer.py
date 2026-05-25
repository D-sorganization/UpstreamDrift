"""Tests for scripts/config/coverage_enforcer.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType


def _load_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "config" / "coverage_enforcer.py"
    assert script_path.is_file(), f"missing script: {script_path}"
    spec = importlib.util.spec_from_file_location("coverage_enforcer", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_coverage_xml(path: Path, body: str) -> None:
    path.write_text(
        dedent(
            f"""\
            <?xml version="1.0" ?>
            <coverage version="7.0">
              <packages>
                <package name="pkg">
                  <classes>
            {body}
                  </classes>
                </package>
              </packages>
            </coverage>
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_collect_policy_results_accepts_threshold_compliant_groups(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    coverage_xml = tmp_path / "coverage.xml"
    _write_coverage_xml(
        coverage_xml,
        """
        <class filename="src/api/routes/chat_ws.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
            <line number="3" hits="1" />
            <line number="4" hits="1" />
            <line number="5" hits="1" />
            <line number="6" hits="1" />
            <line number="7" hits="0" />
          </lines>
        </class>
        <class filename="src/shared/python/engine_core/engine_manager.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
            <line number="3" hits="1" />
            <line number="4" hits="1" />
          </lines>
        </class>
        <class filename="src/api/task_manager.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
            <line number="3" hits="1" />
            <line number="4" hits="1" />
            <line number="5" hits="0" />
          </lines>
        </class>
        <class filename="src/shared/python/logging_pkg/logging_config.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
            <line number="3" hits="1" />
            <line number="4" hits="0" />
          </lines>
        </class>
        """,
    )

    report = mod.parse_coverage_report(coverage_xml)
    failures = mod.find_policy_failures(report, mod.DEFAULT_POLICIES)

    assert failures == []


def test_collect_policy_results_flags_undercovered_group(tmp_path: Path) -> None:
    mod = _load_module()
    coverage_xml = tmp_path / "coverage.xml"
    _write_coverage_xml(
        coverage_xml,
        """
        <class filename="src/api/routes/chat_ws.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="0" />
            <line number="3" hits="0" />
            <line number="4" hits="0" />
          </lines>
        </class>
        <class filename="src/shared/python/engine_core/engine_manager.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
          </lines>
        </class>
        <class filename="src/api/task_manager.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
          </lines>
        </class>
        <class filename="src/shared/python/logging_pkg/logging_config.py">
          <lines>
            <line number="1" hits="1" />
            <line number="2" hits="1" />
          </lines>
        </class>
        """,
    )

    report = mod.parse_coverage_report(coverage_xml)
    failures = mod.find_policy_failures(report, mod.DEFAULT_POLICIES)

    assert any("api_routes_services" in failure for failure in failures)
    assert any("25.0%" in failure for failure in failures)


def test_find_policy_failures_flags_missing_policy_matches(tmp_path: Path) -> None:
    mod = _load_module()
    coverage_xml = tmp_path / "coverage.xml"
    _write_coverage_xml(
        coverage_xml,
        """
        <class filename="src/shared/python/logging_pkg/logging_config.py">
          <lines>
            <line number="1" hits="1" />
          </lines>
        </class>
        """,
    )

    report = mod.parse_coverage_report(coverage_xml)
    failures = mod.find_policy_failures(report, mod.DEFAULT_POLICIES)

    assert any("api_routes_services" in failure for failure in failures)
    assert any("engine_core_control_interface" in failure for failure in failures)
    assert any("task_management" in failure for failure in failures)
