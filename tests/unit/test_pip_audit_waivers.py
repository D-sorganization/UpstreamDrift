"""Tests for the pip-audit waiver helper."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest
pytestmark = pytest.mark.unit



def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_load_waivers_and_emit_ignore_flags(tmp_path):
    module = _load_script_module("check_pip_audit_waivers")
    waiver_file = tmp_path / "waivers.yml"
    waiver_file.write_text(
        """
waivers:
  - id: CVE-2024-0001
    package: demo
    reason: Waiting for upstream fix.
    expires_at: 2099-01-01
""".strip(),
        encoding="utf-8",
    )

    waivers = module.load_waivers(waiver_file)

    assert len(waivers) == 1
    assert module.build_ignore_flags(waivers) == ["--ignore-vuln", "CVE-2024-0001"]


def test_find_expired_waivers_detects_past_dates(tmp_path):
    module = _load_script_module("check_pip_audit_waivers")
    waiver_file = tmp_path / "waivers.yml"
    waiver_file.write_text(
        """
waivers:
  - id: CVE-2024-0001
    package: demo
    reason: Waiting for upstream fix.
    expires_at: 2020-01-01
""".strip(),
        encoding="utf-8",
    )

    waivers = module.load_waivers(waiver_file)
    expired = module.find_expired_waivers(waivers, today=date(2026, 4, 23))

    assert [waiver.id for waiver in expired] == ["CVE-2024-0001"]


def test_load_waivers_rejects_missing_fields(tmp_path):
    module = _load_script_module("check_pip_audit_waivers")
    waiver_file = tmp_path / "waivers.yml"
    waiver_file.write_text(
        """
waivers:
  - id: CVE-2024-0001
    package: demo
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing waiver field: reason"):
        module.load_waivers(waiver_file)
