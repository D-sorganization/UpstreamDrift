from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from sidekick.protocols import CalculationResult, ValidationResult
from sidekick.standalone import runner

pytestmark = pytest.mark.unit


class FakeCalculator:
    @property
    def name(self) -> str:
        return "Fake"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate_inputs(self, inputs: dict[str, Any]) -> ValidationResult:
        if inputs.get("valid", True):
            return ValidationResult(valid=True, warnings=["check"])
        return ValidationResult(valid=False, errors=["bad input"])

    def calculate(self, inputs: dict[str, Any]) -> CalculationResult:
        value = float(inputs.get("value", 2.5))
        return CalculationResult(
            values={"answer": value},
            units={"answer": "kg"},
            warnings=["rounded"],
            metadata={"source": "fake"},
        )


@pytest.fixture(autouse=True)
def _fake_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_REGISTRY", {"fake": FakeCalculator()})


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_calculator_writes_json_to_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _write_json(tmp_path / "inputs.json", {"value": 3})

    code = runner.run_calculator("fake", str(inputs), fmt="json")

    assert code == runner.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "values": {"answer": 3.0},
        "units": {"answer": "kg"},
        "warnings": ["rounded"],
        "metadata": {"source": "fake"},
    }


def test_run_calculator_validation_failure_exits_3(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _write_json(tmp_path / "inputs.json", {"valid": False})

    code = runner.run_calculator("fake", str(inputs), fmt="json")

    assert code == runner.EXIT_VALIDATION
    payload = json.loads(capsys.readouterr().err)
    assert payload["errors"] == ["bad input"]


def test_unknown_calculator_exits_4_with_suggestions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _write_json(tmp_path / "inputs.json", {})

    code = runner.run_calculator("fale", str(inputs), fmt="json")

    assert code == runner.EXIT_UNKNOWN_CALCULATOR
    payload = json.loads(capsys.readouterr().err)
    assert payload["calculator"] == "fale"
    assert payload["suggestions"][0] == "fake"
    assert len(payload["suggestions"]) <= 3


def test_csv_output_has_one_row_per_value_and_units_row(
    tmp_path: Path,
) -> None:
    inputs = _write_json(tmp_path / "inputs.json", {"value": 4})
    output = tmp_path / "result.csv"

    code = runner.run_calculator("fake", str(inputs), str(output), fmt="csv")

    assert code == runner.EXIT_OK
    assert output.read_text(encoding="utf-8").splitlines() == [
        "key,value",
        "answer,4.0",
        "units,answer=kg",
    ]


def test_yaml_inputs_are_loaded_by_extension(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("yaml")
    inputs = tmp_path / "inputs.yaml"
    inputs.write_text("value: 8\n", encoding="utf-8")

    code = runner.run_calculator("fake", str(inputs), fmt="json")

    assert code == runner.EXIT_OK
    assert json.loads(capsys.readouterr().out)["values"]["answer"] == 8.0


def test_output_parent_must_exist(tmp_path: Path) -> None:
    inputs = _write_json(tmp_path / "inputs.json", {})

    code = runner.run_calculator(
        "fake",
        str(inputs),
        str(tmp_path / "missing" / "result.json"),
        fmt="json",
    )

    assert code == runner.EXIT_GENERIC


def test_headless_run_does_not_import_pyqt6(tmp_path: Path) -> None:
    inputs = _write_json(tmp_path / "inputs.json", {})
    pyqt_before = {name for name in sys.modules if name.startswith("PyQt6")}

    code = runner.run_calculator("fake", str(inputs), fmt="json")

    pyqt_after = {name for name in sys.modules if name.startswith("PyQt6")}
    assert code == runner.EXIT_OK
    assert pyqt_after == pyqt_before
