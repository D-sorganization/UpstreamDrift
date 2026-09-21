"""Unit tests for the isolated optional motion-runtime qualification harness."""

from __future__ import annotations

import subprocess
from copy import deepcopy

import pytest

from scripts.ci import check_motion_runtime as checker

pytestmark = pytest.mark.unit


def test_bounded_output_keeps_diagnostic_tail() -> None:
    value = "head\n" + ("x" * (checker.MAX_CAPTURE_CHARS + 50)) + "\ntail"

    bounded = checker.bounded_output(value)

    assert len(bounded) <= checker.MAX_CAPTURE_CHARS
    assert "tail" in bounded
    assert "truncated" in bounded


def test_isolated_probe_reports_missing_interpreter(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("interpreter missing")

    monkeypatch.setattr(checker.subprocess, "run", missing)

    result = checker.run_isolated_probe(
        "missing", timeout_s=1.0, python="missing-python"
    )

    assert result["status"] == "fail"
    assert result["returncode"] is None
    assert "interpreter missing" in result["stderr"]


def test_isolated_probe_reports_timeout_and_captures_partial_output(
    monkeypatch,
) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            kwargs.get("timeout", 1),
            args[0],
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(checker.subprocess, "run", timeout)

    result = checker.run_isolated_probe("slow", timeout_s=0.01)

    assert result["status"] == "fail"
    assert result["timed_out"] is True
    assert result["stdout"] == "partial stdout"
    assert "partial stderr" in result["stderr"]
    assert "timed out" in result["stderr"]


def _valid_receipt() -> dict:
    component = {
        "available": True,
        "version": "1.0",
        "module_file": "/runtime/module.py",
        "source_sha256": "a" * 64,
    }
    return {
        "status": "pass",
        "checker_source_sha256": "b" * 64,
        "source_freshness": {"status": "unknown"},
        "child_python": {"version": "3.12.0", "executable": "/python"},
        "tolerances": dict(checker.TOLERANCES),
        "components": {
            name: deepcopy(component) for name in checker.REQUIRED_COMPONENTS
        },
        "inventory_probe": {"status": "pass", "returncode": 0, "timed_out": False},
        "probes": [
            {
                "name": "crocoddyl_abi",
                "status": "pass",
                "returncode": 0,
                "timed_out": False,
                "result": {"healthy": True, "reason": ""},
            },
            {
                "name": "pink_hard_equality",
                "status": "pass",
                "returncode": 0,
                "timed_out": False,
                "result": {
                    "difference_norm": 0.0,
                    "step_error": 0.0,
                    "q_next0": 0.2,
                    "velocity0": 2.0,
                    "velocity": [2.0, 0.0],
                    "q_next": [0.2, 0.0],
                    "nq": 2,
                    "nv": 2,
                },
            },
            {
                "name": "pink_infeasible_qp",
                "status": "pass",
                "returncode": 0,
                "timed_out": False,
                "result": {
                    "raised": "pink.exceptions.NoSolutionFound",
                    "message": "infeasible",
                },
            },
        ],
    }


def test_receipt_validation_fails_closed_for_missing_or_failed_named_probe() -> None:
    base = _valid_receipt()

    assert checker.validate_receipt(base) == (True, [])

    missing = {**base, "components": {}}
    ok, reasons = checker.validate_receipt(missing)
    assert not ok
    assert any("component" in reason for reason in reasons)

    failed_probe = {
        **base,
        "probes": [{"name": checker.REQUIRED_PROBES[0], "status": "fail"}],
    }
    ok, reasons = checker.validate_receipt(failed_probe)
    assert not ok
    assert any(checker.REQUIRED_PROBES[0] in reason for reason in reasons)


def test_receipt_rejects_success_status_with_failed_process() -> None:
    receipt = _valid_receipt()
    receipt["probes"][0]["returncode"] = 7

    ok, reasons = checker.validate_receipt(receipt)

    assert not ok
    assert any("returncode" in reason for reason in reasons)


def test_receipt_rejects_failed_inventory_process() -> None:
    receipt = _valid_receipt()
    receipt["inventory_probe"]["status"] = "pass"
    receipt["inventory_probe"]["returncode"] = 1

    ok, reasons = checker.validate_receipt(receipt)

    assert not ok
    assert any("inventory process" in reason for reason in reasons)


def test_receipt_rejects_nan_equality_and_duplicate_probe_names() -> None:
    receipt = _valid_receipt()
    receipt["probes"][1]["result"]["difference_norm"] = float("nan")
    receipt["probes"].append(deepcopy(receipt["probes"][0]))

    ok, reasons = checker.validate_receipt(receipt)

    assert not ok
    assert any("finite" in reason for reason in reasons)
    assert any("duplicate" in reason for reason in reasons)


def test_receipt_rejects_malformed_probe_types_and_negative_errors() -> None:
    receipt = _valid_receipt()
    receipt["probes"][1]["result"] = None
    receipt["probes"].append({"name": {"unhashable": True}, "status": "pass"})

    ok, reasons = checker.validate_receipt(receipt)

    assert not ok
    assert any("malformed" in reason for reason in reasons)

    receipt = _valid_receipt()
    receipt["probes"][1]["result"]["difference_norm"] = -1e-3
    ok, reasons = checker.validate_receipt(receipt)
    assert not ok
    assert any("non-negative" in reason for reason in reasons)


def test_timeout_must_be_finite_and_positive(monkeypatch) -> None:
    called = False

    def run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid timeout reached subprocess")

    monkeypatch.setattr(checker.subprocess, "run", run)

    result = checker.run_isolated_probe("bad-timeout", timeout_s=float("nan"))

    assert result["status"] == "fail"
    assert "finite" in result["stderr"]
    assert not called


def test_cli_rejects_infinite_timeout() -> None:
    assert checker.main(["--timeout", "inf"]) == 2


def test_linux_lock_is_standard_explicit_manifest_without_host_prefix() -> None:
    lock = (
        checker.REPO_ROOT
        / "scripts"
        / "config"
        / "motion_runtime"
        / "linux-64.explicit.txt"
    ).read_text(encoding="utf-8")
    lines = lock.splitlines()

    assert "@EXPLICIT" in lines
    assert not any("/home/" in line or "prefix:" in line for line in lines)
    urls = [line for line in lines if line.startswith("https://")]
    assert len(urls) > 100
    assert all("conda-forge" in url for url in urls)


def test_receipt_rejects_nonfinite_unreported_velocity_component() -> None:
    receipt = _valid_receipt()
    receipt["probes"][1]["result"]["velocity"] = [2.0, float("nan")]
    ok, reasons = checker.validate_receipt(receipt)
    assert not ok
    assert any("velocity" in reason for reason in reasons)
