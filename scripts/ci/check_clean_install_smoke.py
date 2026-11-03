#!/usr/bin/env python
"""Verify clean-environment installation and core runtime smoke (#14275)."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("clean_install_smoke")

REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_TOOLS_ROOT = REPO_ROOT / "vendor" / "ud-tools"
for _entry in (
    REPO_ROOT / "src" / "shared" / "python",
    REPO_ROOT / "src",
    REPO_ROOT,
    _VENDOR_TOOLS_ROOT / "src",
    _VENDOR_TOOLS_ROOT / "src" / "python" / "src",
    _VENDOR_TOOLS_ROOT / "src" / "shared" / "python",
):
    _path = str(_entry)
    if _entry.is_dir() and _path not in sys.path:
        sys.path.insert(0, _path)

CORE_REQUIRED_PACKAGES: tuple[tuple[str, str | None], ...] = (
    ("numpy", None),
    ("scipy", None),
    ("fastapi", None),
    ("uvicorn", None),
    ("pydantic", None),
    ("mujoco", None),
    ("sqlalchemy", None),
    ("bcrypt", None),
    ("PyJWT", "jwt"),
    ("defusedxml", None),
    ("pandas", None),
    ("platformdirs", None),
    ("structlog", None),
)

CORE_SURFACE_MODULES: tuple[str, ...] = (
    "src.api.local_server",
    "src.engines.physics_engines.mujoco",
    "src.shared.python.physics",
    "src.shared.python.spatial_algebra",
    "src.shared.python.engine_core",
)

FORBIDDEN_ENGINES: tuple[str, ...] = (
    "drake",
    "pinocchio",
    "opensim",
    "myosuite",
    "jaxsim",
)


def check_python_version() -> tuple[bool, str]:
    """Check Python version satisfies ``>= 3.11, < 3.13``."""
    info = sys.version_info
    ver_str = f"{info[0]}.{info[1]}.{info[2] if len(info) > 2 else 0}"
    if (info[0], info[1]) in ((3, 11), (3, 12)):
        return True, f"Python {ver_str} is within supported range (3.11-3.12)"
    return False, f"Python {ver_str} is unsupported; requires >= 3.11 and < 3.13"


def check_core_dependencies() -> list[tuple[str, bool, str]]:
    """Check that all required core runtime dependencies are importable."""
    results: list[tuple[str, bool, str]] = []
    for display_name, import_name in CORE_REQUIRED_PACKAGES:
        target = import_name or display_name
        try:
            mod = importlib.import_module(target)
            results.append(
                (
                    display_name,
                    True,
                    f"{display_name} ({getattr(mod, '__version__', 'ok')})",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                (display_name, False, f"{display_name} import failed: {exc}")
            )
    return results


def check_core_import_surface() -> list[tuple[str, bool, str]]:
    """Verify that the core public modules load without missing symbols."""
    results: list[tuple[str, bool, str]] = []
    for mod_name in CORE_SURFACE_MODULES:
        try:
            importlib.import_module(mod_name)
            results.append((mod_name, True, f"{mod_name} imported successfully"))
        except Exception as exc:  # noqa: BLE001
            results.append((mod_name, False, f"{mod_name} import failed: {exc}"))
    return results


def check_forbidden_engine_isolation() -> list[tuple[str, bool, str]]:
    """Verify optional heavy engines are not loaded in the core environment."""
    results: list[tuple[str, bool, str]] = []
    for engine in FORBIDDEN_ENGINES:
        loaded = any(m == engine or m.startswith(f"{engine}.") for m in sys.modules)
        if loaded:
            results.append((engine, False, f"{engine} loaded (breaks core isolation)"))
        else:
            results.append((engine, True, f"{engine} isolated"))
    return results


def check_cli_entrypoints(
    python_bin: Path | None = None,
) -> list[tuple[str, bool, str]]:
    """Verify that canonical CLI entrypoints render help output without error."""
    py_exec = str(python_bin or sys.executable)
    results: list[tuple[str, bool, str]] = []
    commands = (
        ("upstream-drift launcher", [py_exec, "launch_upstream_drift.py", "--help"]),
        ("ud-sim cli", [py_exec, "-m", "src.sim_run.cli", "--help"]),
    )
    for label, cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            if proc.returncode == 0:
                results.append((label, True, f"{label} returned 0"))
            else:
                err = (proc.stderr or proc.stdout).strip()[:120]
                results.append(
                    (label, False, f"{label} exited {proc.returncode}: {err}")
                )
        except Exception as exc:  # noqa: BLE001
            results.append((label, False, f"{label} failed: {exc}"))
    return results


def run_clean_install_smoke_checks(
    *, check_cli: bool = False, python_bin: Path | None = None
) -> dict[str, Any]:
    """Execute all smoke checks and return a structured report dictionary."""
    py_ok, py_msg = check_python_version()
    core_deps = check_core_dependencies()
    core_surface = check_core_import_surface()
    isolation = check_forbidden_engine_isolation()
    cli_res = check_cli_entrypoints(python_bin=python_bin) if check_cli else []

    all_checks = [(py_ok, py_msg)]
    all_checks.extend((ok, msg) for _, ok, msg in core_deps)
    all_checks.extend((ok, msg) for _, ok, msg in core_surface)
    all_checks.extend((ok, msg) for _, ok, msg in isolation)
    if check_cli:
        all_checks.extend((ok, msg) for _, ok, msg in cli_res)

    passed_count = sum(1 for ok, _ in all_checks if ok)
    total_count = len(all_checks)
    status = "passed" if passed_count == total_count else "failed"

    return {
        "status": status,
        "summary": {"passed": passed_count, "total": total_count},
        "checks": {
            "python_version": {"passed": py_ok, "detail": py_msg},
            "core_dependencies": [
                {"name": n, "passed": k, "detail": m} for n, k, m in core_deps
            ],
            "core_surface": [
                {"name": n, "passed": k, "detail": m} for n, k, m in core_surface
            ],
            "isolation": [
                {"name": n, "passed": k, "detail": m} for n, k, m in isolation
            ],
            "cli": [{"name": n, "passed": k, "detail": m} for n, k, m in cli_res]
            if check_cli
            else [],
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the clean-environment install smoke test CLI."""
    parser = argparse.ArgumentParser(
        description="Verify clean-environment install smoke invariants."
    )
    parser.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    parser.add_argument(
        "--check-cli", action="store_true", help="Run CLI entrypoint checks"
    )
    parser.add_argument(
        "--python", type=Path, default=None, help="Target Python executable"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_clean_install_smoke_checks(
        check_cli=args.check_cli, python_bin=args.python
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        logger.info(
            "Clean-Environment Install Smoke: %s (%d/%d passed)",
            report["status"].upper(),
            report["summary"]["passed"],
            report["summary"]["total"],
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
