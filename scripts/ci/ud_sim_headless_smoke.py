#!/usr/bin/env python3
"""CI headless smoke test for ud-sim container (Epic #14085 / Issue #14091).

Executes a minimal simulation case via the ud-sim CLI and asserts:
1. Pure headless operation (zero Qt/GUI modules loaded in sys.modules during execution).
2. The simulation run status is 'succeeded' with a valid result record.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

DISALLOWED_QT_PREFIXES: tuple[str, ...] = (
    "PyQt6",
    "PyQt5",
    "PySide6",
    "PySide2",
    "qtpy",
)


def assert_no_qt_modules(modules_dict: dict[str, Any] | None = None) -> list[str]:
    """Verify that no Qt modules are present in sys.modules.

    Design by Contract:
    - Precondition: modules_dict is a mapping of module names (defaults to sys.modules).
    - Postcondition: Returns empty list if clean; raises RuntimeError if Qt found.
    """
    target_modules = sys.modules if modules_dict is None else modules_dict
    found_qt: list[str] = [
        name
        for name in target_modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in DISALLOWED_QT_PREFIXES
        )
    ]
    if found_qt:
        raise RuntimeError(
            f"Falsification triggered: Qt module(s) detected in headless simulation: {found_qt}"
        )
    return found_qt


def validate_run_record(record_data: dict[str, Any]) -> None:
    """Validate that the simulation run succeeded with valid results.

    Design by Contract:
    - Postcondition: Raises RuntimeError if status is not 'succeeded' or validation invalid.
    """
    status = record_data.get("status")
    if status != "succeeded":
        raise RuntimeError(
            f"Falsification triggered: Simulation run did not succeed (status={status!r}, "
            f"error={record_data.get('error')})"
        )

    result = record_data.get("result")
    if not isinstance(result, dict) or result.get("validation_status") != "valid":
        raise RuntimeError(
            f"Falsification triggered: Simulation result invalid: {result}"
        )


def run_headless_smoke(
    case_path: Path,
    results_root: Path,
    python_executable: str = sys.executable,
    inject_mock_qt: bool = False,
) -> dict[str, Any]:
    """Run one case through ud-sim CLI and verify headless contracts.

    Spawns an isolated Python subprocess that runs the simulation CLI and
    asserts within that subprocess that no Qt module was ever imported.
    """
    if not case_path.is_file():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    results_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    # Subprocess execution script that runs the CLI and checks sys.modules inside the simulation process
    runner_code = (
        "import sys\n"
        "from src.sim_run.cli import main\n"
        + ("sys.modules['PyQt6'] = object()\n" if inject_mock_qt else "")
        + "rc = main(['run', sys.argv[1], '--results-root', sys.argv[2]])\n"
        f"disallowed = {DISALLOWED_QT_PREFIXES!r}\n"
        "found = [m for m in sys.modules if any(m == p or m.startswith(p + '.') for p in disallowed)]\n"
        "if found:\n"
        "    sys.stderr.write(f'Falsification triggered: Qt module(s) detected: {found}\\n')\n"
        "    sys.exit(77)\n"
        "sys.exit(rc)\n"
    )

    cmd = [
        python_executable,
        "-c",
        runner_code,
        str(case_path),
        str(results_root),
    ]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
    )

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120.0,
    )

    if proc.returncode == 77:
        raise RuntimeError(
            f"Falsification triggered: Qt module(s) detected in headless simulation: {proc.stderr.strip()}"
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Falsification triggered: ud-sim execution failed with code {proc.returncode}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # Parse stdout JSON record
    try:
        record_data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ud-sim CLI stdout was not valid JSON: {exc}\nSTDOUT:\n{proc.stdout}"
        ) from exc

    validate_run_record(record_data)
    return record_data


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for docker smoke script."""
    parser = argparse.ArgumentParser(description="Headless ud-sim smoke test")
    parser.add_argument(
        "--case",
        type=Path,
        default=Path("tests/fixtures/sim_run/smoke_case.json"),
        help="Path to test case JSON file",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Directory to write simulation results (defaults to tempdir)",
    )
    args = parser.parse_args(argv)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    results_root = args.results_root
    if results_root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="ud_sim_smoke_")
        results_root = Path(temp_dir.name)

    try:
        record = run_headless_smoke(args.case, results_root)
        print(
            f"PASS: Headless simulation succeeded (run_id={record.get('run_id')}, no-Qt verified)"
        )
        return 0
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())
