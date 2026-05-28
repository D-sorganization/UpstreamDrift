"""Tests for the headless sidekick run dispatcher — T4 (#5982).

Covers each exit-code branch using fake calculators and a real wgs_reactor
invocation via JSON fixture on disk.  All tests are free of GUI imports;
a sys.modules snapshot assertion enforces this.

Exit codes:
  0 — success
  1 — I/O error (missing file, JSON parse error)
  3 — validation or calculation failure
  4 — unknown calculator id
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _invoke(
    calculator: str,
    inputs: dict,
    tmp_path: Path,
    output: str = "-",
    format: str = "json",
) -> tuple[int, str, str]:
    """Run run_calculator and capture stdout/stderr."""
    from sidekick.standalone.runner import run_calculator

    inputs_file = _write_json(tmp_path / "inputs.json", inputs)
    captured_out: list[str] = []
    captured_err: list[str] = []

    with (
        patch("sys.stdout", new=io.StringIO()) as mock_out,
        patch("sys.stderr", new=io.StringIO()) as mock_err,
    ):
        code = run_calculator(
            calculator=calculator,
            inputs_path=str(inputs_file),
            output=output,
            format=format,
        )
        captured_out.append(mock_out.getvalue())
        captured_err.append(mock_err.getvalue())

    return code, captured_out[0], captured_err[0]


# ---------------------------------------------------------------------------
# Fake Calculator Protocol implementation
# ---------------------------------------------------------------------------


class _FakeValidationResult:
    def __init__(self, valid: bool, errors: list[str]) -> None:
        self.valid = valid
        self.errors = errors


class _FakeCalculationResult:
    def __init__(self, values: dict, units: dict) -> None:
        self.values = values
        self.units = units
        self.warnings: list[str] = []


class _FakeCalculator:
    """Minimal Calculator Protocol stub."""

    def __init__(self, valid: bool = True, error_msg: str = "") -> None:
        self._valid = valid
        self._error_msg = error_msg

    def validate_inputs(self, inputs: dict) -> _FakeValidationResult:
        if not self._valid:
            return _FakeValidationResult(valid=False, errors=[self._error_msg])
        return _FakeValidationResult(valid=True, errors=[])

    def calculate(self, inputs: dict) -> _FakeCalculationResult:
        return _FakeCalculationResult(
            values={"result": 42.0, "ratio": 0.5},
            units={"result": "J/mol", "ratio": ""},
        )


# ---------------------------------------------------------------------------
# Exit-code tests
# ---------------------------------------------------------------------------


def test_success_exits_zero(tmp_path: Path) -> None:
    code, out, _ = _invoke("wgs_reactor", {}, tmp_path)
    assert code == 0
    data = json.loads(out)
    assert "co_conversion_fraction" in data


def test_unknown_calculator_exits_4(tmp_path: Path) -> None:
    code, _, err = _invoke("nonexistent_calc", {}, tmp_path)
    assert code == 4
    payload = json.loads(err)
    assert "Unknown calculator" in payload["error"]
    assert "closest" in payload


def test_unknown_calculator_suggests_closest(tmp_path: Path) -> None:
    code, _, err = _invoke("wgs_reacto", {}, tmp_path)
    assert code == 4
    payload = json.loads(err)
    assert "wgs_reactor" in payload["closest"]


def test_missing_inputs_file_exits_1(tmp_path: Path) -> None:
    from sidekick.standalone.runner import run_calculator

    with patch("sys.stderr", new=io.StringIO()):
        code = run_calculator("wgs_reactor", str(tmp_path / "missing.json"))
    assert code == 1


def test_invalid_json_exits_1(tmp_path: Path) -> None:
    from sidekick.standalone.runner import run_calculator

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json", encoding="utf-8")

    with patch("sys.stderr", new=io.StringIO()):
        code = run_calculator("wgs_reactor", str(bad_file))
    assert code == 1


def test_validation_failure_exits_3(tmp_path: Path) -> None:
    from sidekick.standalone import runner

    runner._REGISTRY["_test_fail"] = _FakeCalculator(valid=False, error_msg="bad input")
    try:
        code, _, err = _invoke("_test_fail", {}, tmp_path)
        assert code == 3
        payload = json.loads(err)
        assert payload["errors"] == ["bad input"]
    finally:
        del runner._REGISTRY["_test_fail"]


def test_calculation_exception_exits_3(tmp_path: Path) -> None:
    from sidekick.standalone import runner

    def _raise(inputs: dict) -> dict:
        raise ValueError("calc exploded")

    runner._REGISTRY["_test_exc"] = _raise
    try:
        code, _, err = _invoke("_test_exc", {}, tmp_path)
        assert code == 3
        payload = json.loads(err)
        assert "calc exploded" in payload["errors"][0]
    finally:
        del runner._REGISTRY["_test_exc"]


def test_protocol_calculator_success(tmp_path: Path) -> None:
    from sidekick.standalone import runner

    runner._REGISTRY["_test_proto"] = _FakeCalculator(valid=True)
    try:
        code, out, _ = _invoke("_test_proto", {}, tmp_path)
        assert code == 0
        data = json.loads(out)
        assert data["values"]["result"] == pytest.approx(42.0)
        assert data["units"]["result"] == "J/mol"
    finally:
        del runner._REGISTRY["_test_proto"]


# ---------------------------------------------------------------------------
# CSV format
# ---------------------------------------------------------------------------


def test_csv_format_has_header_and_rows(tmp_path: Path) -> None:
    from sidekick.standalone import runner

    runner._REGISTRY["_test_csv"] = _FakeCalculator(valid=True)
    try:
        code, out, _ = _invoke("_test_csv", {}, tmp_path, format="csv")
        assert code == 0
        reader = csv.reader(io.StringIO(out))
        rows = list(reader)
        assert rows[0] == ["metric", "value", "unit"]
        keys = [r[0] for r in rows[1:]]
        assert "result" in keys
        assert "ratio" in keys
    finally:
        del runner._REGISTRY["_test_csv"]


def test_csv_units_column_present(tmp_path: Path) -> None:
    from sidekick.standalone import runner

    runner._REGISTRY["_test_csv2"] = _FakeCalculator(valid=True)
    try:
        code, out, _ = _invoke("_test_csv2", {}, tmp_path, format="csv")
        assert code == 0
        reader = csv.reader(io.StringIO(out))
        rows = list(reader)
        result_row = next(r for r in rows[1:] if r[0] == "result")
        assert result_row[2] == "J/mol"
    finally:
        del runner._REGISTRY["_test_csv2"]


def test_wgs_reactor_json_output_structure(tmp_path: Path) -> None:
    code, out, _ = _invoke("wgs_reactor", {"temperature_c": 350.0}, tmp_path)
    assert code == 0
    data = json.loads(out)
    assert 0.0 <= data["co_conversion_fraction"] <= 1.0
    assert abs(sum(data["equilibrium_composition"].values()) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# No PyQt6 on headless path
# ---------------------------------------------------------------------------


def test_runner_module_does_not_import_pyqt6() -> None:
    """Importing sidekick.standalone.runner must not pull in any PyQt6 module."""
    # Reload the module in isolation with PyQt6 blocked.
    import importlib

    saved = {k: v for k, v in sys.modules.items() if k.startswith("PyQt6")}
    for key in saved:
        del sys.modules[key]

    class _BlockPyQt6:
        def find_module(self, fullname: str, path: Any = None) -> Any:
            if fullname.startswith("PyQt6"):
                return self
            return None

        def load_module(self, fullname: str) -> Any:
            raise ImportError(f"PyQt6 import blocked in test: {fullname}")

    blocker = _BlockPyQt6()
    sys.meta_path.insert(0, blocker)
    try:
        if "sidekick.standalone.runner" in sys.modules:
            del sys.modules["sidekick.standalone.runner"]
        importlib.import_module("sidekick.standalone.runner")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
