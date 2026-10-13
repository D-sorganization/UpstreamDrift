"""Clean-process import diagnostic for the OpenSim SDK and ``ezc3d`` (#12065).

Every probe runs in a fresh, isolated (``-I``) interpreter that imports only
the modules in its order, so no probe inherits a DLL loaded by another.  The
report names the interpreter, each module's version and file, the
``ezc3d.dll`` loaded after each import (its provenance) and the exact error
of every failed import.  Failures are recorded, never suppressed or retried.

    python -m scripts.diagnostics.native_runtime_identity \
        --python PATH/TO/python.exe --output identity.json

Exit status is 1 when a runtime fails to import on its own (a broken
install) or a probe crashes; the documented mixed-process incompatibility is
a reported identity, not a diagnostic failure.  This module imports only the
standard library so it can run under any interpreter.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Stdlib-only Tools guard, loaded by path so the probe imports no repo package.
_GUARD_PATH = REPO_ROOT / "src/shared/python/sidekick/utils/native_dll_guard.py"
PROBE_ORDERS: tuple[tuple[str, ...], ...] = (
    ("ezc3d",),
    ("opensim",),
    ("opensim", "ezc3d"),
    ("ezc3d", "opensim"),
)
_RUNTIMES = {"ezc3d": "c3d-preparation", "opensim": "native-opensim"}
Probe = Mapping[str, Any]
_SCHEMA = "upstreamdrift.native_runtime_identity.v1"


def _load_dll_guard():  # noqa: ANN202 - module object from a file path
    spec = importlib.util.spec_from_file_location("_ud_native_dll_guard", _GUARD_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the native DLL guard from {_GUARD_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_version(module: object) -> str | None:
    version = getattr(module, "__version__", None)
    if version is None and hasattr(module, "GetVersionAndDate"):
        version = module.GetVersionAndDate()
    return None if version is None else str(version)


def _probe(order: Sequence[str]) -> dict[str, object]:
    """Import ``order`` in this (fresh) process and record every outcome."""
    guard = _load_dll_guard()
    clean = not any(name in sys.modules for name in _RUNTIMES)
    steps: list[dict[str, object]] = []
    for name in order:
        step: dict[str, object] = {"module": name}
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            step.update(
                status="not-installed",
                error_type=type(error).__name__,
                error=str(error),
            )
        except ImportError as error:
            step.update(
                status="import-error", error_type=type(error).__name__, error=str(error)
            )
        else:
            step.update(
                status="ok",
                version=_module_version(module),
                file=getattr(module, "__file__", None),
            )
        step["ezc3d_dll"] = guard.loaded_dll_path(guard.EZC3D_DLL)
        steps.append(step)
    return {"order": list(order), "clean_start": clean, "steps": steps}


def _run_probe(python: Path, order: Sequence[str]) -> dict[str, object]:
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    result = subprocess.run(
        [str(python), "-I", str(Path(__file__).resolve()), "--probe", *order],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        return {
            "order": list(order),
            "clean_start": True,
            "crashed": True,
            "returncode": result.returncode,
            "stderr": result.stderr.strip()[-2000:],
            "steps": [],
        }
    return json.loads(result.stdout.strip().splitlines()[-1])


def _alone(probes: Sequence[Probe], name: str) -> Probe:
    for probe in probes:
        if probe["order"] == [name]:
            steps = probe["steps"]
            if not steps:
                return {"status": "import-error", "error_type": "ProbeCrash"}
            return steps[0]
    raise ValueError(f"No single-runtime probe for {name}")


def _runtime_identity(step: Mapping[str, object]) -> str:
    if step["status"] == "ok":
        return "supported"
    if step["status"] == "not-installed":
        return f"unavailable: {step.get('module', 'runtime')} is not installed"
    return f"broken: {step.get('error_type')}: {step.get('error', '')}".strip()


def classify_runtimes(probes: Sequence[Probe]) -> dict[str, str]:
    """Return the explicit identity of each runtime and of a mixed process.

    Precondition: every probe started clean (neither runtime preloaded).
    """
    if not all(probe["clean_start"] for probe in probes):
        raise ValueError("Every probe must start in a clean process")
    alone = {name: _alone(probes, name) for name in _RUNTIMES}
    identities = {
        _RUNTIMES[name]: _runtime_identity(step) for name, step in alone.items()
    }
    unusable = [name for name, step in alone.items() if step["status"] != "ok"]
    if unusable:
        reasons = "; ".join(
            f"{name} is not installed"
            if alone[name]["status"] == "not-installed"
            else f"{name} fails to import alone"
            for name in unusable
        )
        identities["mixed-process"] = (
            f"not-evaluable: {reasons} (no qualified evidence)"
        )
        return identities
    failures = [
        " -> ".join(probe["order"])
        for probe in probes
        if len(probe["order"]) > 1
        and (not probe["steps"] or any(s["status"] != "ok" for s in probe["steps"]))
    ]
    identities["mixed-process"] = (
        f"incompatible: {', '.join(failures)} fails; keep C3D preparation and the "
        "native SDK in separate processes"
        if failures
        else "compatible"
    )
    return identities


def exit_status(probes: Sequence[Probe]) -> int:
    """1 when a probe crashed or a runtime cannot be imported on its own."""
    if any(probe.get("crashed") for probe in probes):
        return 1
    broken = [
        name for name in _RUNTIMES if _alone(probes, name)["status"] == "import-error"
    ]
    return 1 if broken else 0


def build_report(python: Path) -> dict[str, Any]:
    """Run every probe order in its own fresh ``python`` process."""
    probes = [_run_probe(python, order) for order in PROBE_ORDERS]
    interpreter = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json, platform, sys; print(json.dumps({'version': "
            "sys.version.split()[0], 'executable': sys.executable, "
            "'implementation': platform.python_implementation(), "
            "'platform': platform.platform()}))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return {
        "schema": _SCHEMA,
        "python": json.loads(interpreter.stdout),
        "diagnostic_host": platform.platform(),
        "probes": probes,
        "runtime_identities": classify_runtimes(probes),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write the identity report (or a single probe result) as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.probe:
        print(json.dumps(_probe(args.probe)))
        return 0
    report = build_report(args.python)
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return exit_status(report["probes"])


if __name__ == "__main__":
    raise SystemExit(main())
