"""Contract and falsification tests for headless ud-sim smoke (Epic #14085 / Issue #14091)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.ci.ud_sim_headless_smoke import (
    assert_no_qt_modules,
    run_headless_smoke,
    validate_run_record,
)

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_falsifying_smoke_fails_if_qt_module_imported() -> None:
    """Falsification test: assert_no_qt_modules MUST raise RuntimeError if any Qt module is present."""
    # Test in-process validation with simulated PyQt6
    fake_modules: dict[str, Any] = {
        "sys": None,
        "os": None,
        "PyQt6": object(),
        "PyQt6.QtCore": object(),
    }
    with pytest.raises(RuntimeError, match="Falsification triggered: Qt module.*PyQt6"):
        assert_no_qt_modules(fake_modules)

    # Test in-process validation with simulated PySide6
    fake_modules_pyside: dict[str, Any] = {
        "sys": None,
        "PySide6.QtWidgets": object(),
    }
    with pytest.raises(
        RuntimeError, match="Falsification triggered: Qt module.*PySide6"
    ):
        assert_no_qt_modules(fake_modules_pyside)

    # Clean modules must pass and return empty list
    clean_modules: dict[str, Any] = {"sys": None, "math": None, "numpy": None}
    assert assert_no_qt_modules(clean_modules) == []


def test_falsifying_smoke_subprocess_fails_when_qt_injected(tmp_path: Path) -> None:
    """Falsification test: runner subprocess aborts when mock Qt module is imported."""
    fixture_case = REPO_ROOT / "tests" / "fixtures" / "sim_run" / "smoke_case.json"
    assert fixture_case.is_file(), f"Fixture missing at {fixture_case}"

    results_dir = tmp_path / "sim_results_qt"
    with pytest.raises(RuntimeError, match="Falsification triggered.*PyQt6"):
        run_headless_smoke(fixture_case, results_dir, inject_mock_qt=True)


def test_falsifying_smoke_fails_if_run_not_succeeded() -> None:
    """Falsification test: validate_run_record MUST raise RuntimeError if status is not 'succeeded'."""
    failed_record = {
        "run_id": "run_test_fail",
        "status": "failed",
        "error": "Solver divergence",
        "result": None,
    }
    with pytest.raises(
        RuntimeError, match="Falsification triggered: Simulation run did not succeed"
    ):
        validate_run_record(failed_record)

    invalid_result_record = {
        "run_id": "run_test_invalid",
        "status": "succeeded",
        "result": {"validation_status": "corrupt"},
    }
    with pytest.raises(
        RuntimeError, match="Falsification triggered: Simulation result invalid"
    ):
        validate_run_record(invalid_result_record)

    valid_record = {
        "run_id": "run_test_ok",
        "status": "succeeded",
        "result": {"validation_status": "valid"},
    }
    validate_run_record(valid_record)


def test_smoke_end_to_end_succeeds_and_verifies_clean_environment(
    tmp_path: Path,
) -> None:
    """End-to-end execution of headless smoke over real case fixture produces valid record."""
    fixture_case = REPO_ROOT / "tests" / "fixtures" / "sim_run" / "smoke_case.json"
    assert fixture_case.is_file(), f"Fixture missing at {fixture_case}"

    results_dir = tmp_path / "sim_results"
    record = run_headless_smoke(fixture_case, results_dir)

    assert record["status"] == "succeeded"
    assert record["request"]["engine_type"] == "pendulum"
    assert record["result"]["validation_status"] == "valid"
    assert (results_dir / record["run_id"] / "record.json").is_file()


def test_smoke_fails_on_missing_case(tmp_path: Path) -> None:
    """Smoke runner fails fast with FileNotFoundError if case file is missing."""
    nonexistent = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        run_headless_smoke(nonexistent, tmp_path)
