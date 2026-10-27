"""Regression tests for stale architecture-budget exception rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci import check_architecture_budget as mod

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "scripts" / "config" / "architecture_budget.json"
_EXPORTER_PATH = _REPO_ROOT / "src" / "tools" / "model_converter" / "urdf_exporter.py"
_SYMBOL = "_add_revolute_joint"
_RULE = "parameters"
_REL_EXPORTER = "src/tools/model_converter/urdf_exporter.py"


def _load_budget_config() -> dict:
    if not _CONFIG_PATH.is_file():
        raise FileNotFoundError(f"architecture budget config missing: {_CONFIG_PATH}")
    raw = _CONFIG_PATH.read_text(encoding="utf-8")
    config = json.loads(raw)
    if not isinstance(config, dict) or "exceptions" not in config:
        raise ValueError(f"malformed architecture budget config: {_CONFIG_PATH}")
    return config


def _exception_rows_for_symbol(config: dict) -> list[dict]:
    rows: list[dict] = []
    for entry in config["exceptions"]:
        if (
            entry.get("path") == _REL_EXPORTER
            and entry.get("symbol") == _SYMBOL
            and entry.get("rule") == _RULE
        ):
            rows.append(entry)
    return rows


@pytest.mark.unit
def test_urdf_exporter_add_revolute_joint_has_no_stale_budget_exception() -> None:
    """_add_revolute_joint is within budget; no parameters exception row may remain."""
    if not _EXPORTER_PATH.is_file():
        raise FileNotFoundError(f"urdf exporter missing: {_EXPORTER_PATH}")

    config = _load_budget_config()
    max_function_lines = int(config["max_function_lines"])
    max_parameters = int(config["max_parameters"])

    findings = mod.analyze_python_file(
        _EXPORTER_PATH,
        max_function_lines=max_function_lines,
        max_parameters=max_parameters,
    )
    symbol_findings = [f for f in findings if f.symbol == _SYMBOL]
    assert symbol_findings == [], (
        f"{_SYMBOL} should meet budget (expected 2 parameters, max {max_parameters}); "
        f"got findings: {symbol_findings}"
    )

    stale_rows = _exception_rows_for_symbol(config)
    assert stale_rows == [], (
        f"stale exception row(s) for {_SYMBOL}/{_RULE} must be removed: {stale_rows}"
    )
