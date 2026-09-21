"""Run native-engine pytest lanes and emit hashed nightly receipts (MS-43 #10342).

OpenSim and MyoSuite tests are skipped in standard CI; this harness is intended
for ControlTower or another labeled runner with the qualified SDK venv.  Receipt
shape follows ``scripts/ci/check_motion_runtime.py`` (schema version, checker
hash, repository revision, source freshness, per-probe outcomes).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_contract_helpers() -> tuple[Any, Any]:
    """Load DbC helpers without importing ``src.shared.python`` package init."""

    path = REPO_ROOT / "src" / "shared" / "python" / "contracts.py"
    spec = importlib.util.spec_from_file_location("native_lane_contracts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load contracts module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.precondition, module.postcondition


precondition, postcondition = _load_contract_helpers()
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "docs"
    / "development"
    / "matched_swing_program"
    / "evidence"
    / "nightly"
)
DEFAULT_TIMEOUT_SECONDS = 600.0
WARN_FRESHNESS_DAYS = 7
FAIL_FRESHNESS_DAYS = 30

ENGINE_LANES: dict[str, dict[str, Any]] = {
    "opensim": {
        "pytest_marker": "requires_opensim",
        "python_module": "opensim",
        "distribution": "opensim",
        "default_venv": "/home/dieterolson/opensim-10003",
        "receipt_filename": "opensim_receipt.json",
        "runner_hint": "ControlTower opensim-10003 venv; nightly-cross-engine.yml",
    },
    "myosuite": {
        "pytest_marker": "requires_myosuite",
        "python_module": "myosuite",
        "distribution": "myosuite",
        "default_venv": None,
        "receipt_filename": "myosuite_receipt.json",
        "runner_hint": "ControlTower with myosuite extra; nightly-cross-engine.yml",
    },
}

CONTRACT_PATHS = (
    "scripts/ci/run_native_engine_lane.py",
    "scripts/ci/run_native_engine_lane.sh",
    "pyproject.toml",
)


def _module_hash(module_file: str | None) -> str | None:
    if not module_file:
        return None
    try:
        digest = hashlib.sha256()
        with open(module_file, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def contract_hashes(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Return SHA-256 digests for the lane contract files."""

    hashes: dict[str, str] = {}
    for relative in CONTRACT_PATHS:
        digest = _file_sha256(repo_root / relative)
        if digest is not None:
            hashes[relative] = digest
    return hashes


def _repository_revision(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode == 0:
        return completed.stdout.strip() or None
    return None


def _repository_freshness(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    paths = list(CONTRACT_PATHS)
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unknown", "reason": str(exc)}
    if completed.returncode != 0:
        return {
            "status": "unknown",
            "reason": completed.stderr or "git status failed",
        }
    output = completed.stdout.strip()
    return {
        "status": "dirty" if output else "clean",
        "paths": paths,
        "details": output,
    }


def _parse_iso8601(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def assess_receipt_age_days(
    generated_at: str,
    *,
    now: datetime | None = None,
) -> float:
    """Return whole-day age of a receipt timestamp."""

    reference = now or datetime.now(tz=UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    delta = reference - _parse_iso8601(generated_at)
    return max(delta.total_seconds(), 0.0) / 86400.0


@precondition(
    lambda generated_at: isinstance(generated_at, str) and bool(generated_at.strip())
)
@postcondition(lambda result: result[0] in {"ok", "warn", "fail"} and result[1] >= 0)
def assess_receipt_freshness(
    generated_at: str,
    *,
    now: datetime | None = None,
    warn_days: int = WARN_FRESHNESS_DAYS,
    fail_days: int = FAIL_FRESHNESS_DAYS,
) -> tuple[str, float]:
    """Classify receipt freshness as ok, warn, or fail."""

    age_days = assess_receipt_age_days(generated_at, now=now)
    if age_days >= fail_days:
        return "fail", age_days
    if age_days >= warn_days:
        return "warn", age_days
    return "ok", age_days


def _engine_inventory(
    *,
    module_name: str,
    distribution: str,
    python: str,
) -> dict[str, Any]:
    probe = (
        "import importlib, importlib.metadata, json, platform, sys\n"
        f"module = importlib.import_module({module_name!r})\n"
        f"dist = {distribution!r}\n"
        "version = None\n"
        "try:\n"
        "    version = importlib.metadata.version(dist)\n"
        "except importlib.metadata.PackageNotFoundError:\n"
        "    version = getattr(module, '__version__', None)\n"
        "print(json.dumps({\n"
        "    'available': version is not None,\n"
        "    'version': version,\n"
        '    "module_file": getattr(module, "__file__", None),\n'
        "    'child_python': {\n"
        "        'executable': sys.executable,\n"
        "        'version': platform.python_version(),\n"
        "    },\n"
        "}, sort_keys=True))\n"
    )
    command = [python, "-c", probe]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "version": None,
            "module_file": None,
            "source_sha256": None,
            "error": str(exc),
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "version": None,
            "module_file": None,
            "source_sha256": None,
            "error": (completed.stderr or completed.stdout or "import failed").strip(),
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "available": False,
            "version": None,
            "module_file": None,
            "source_sha256": None,
            "error": "malformed inventory JSON",
        }
    module_file = payload.get("module_file")
    payload["source_sha256"] = _module_hash(module_file)
    return payload


def _parse_junit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "cases": [],
            "parse_error": f"missing junit file: {path}",
        }
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "cases": [],
            "parse_error": str(exc),
        }

    cases: list[dict[str, Any]] = []
    passed = failed = skipped = errors = 0
    for case in root.iter("testcase"):
        nodeid = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        full_name = f"{nodeid}::{name}" if nodeid else name
        if case.find("failure") is not None:
            status = "fail"
            failed += 1
        elif case.find("error") is not None:
            status = "error"
            errors += 1
        elif case.find("skipped") is not None:
            status = "skip"
            skipped += 1
        else:
            status = "pass"
            passed += 1
        cases.append({"nodeid": full_name, "status": status})
    collected = passed + failed + skipped + errors
    return {
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "cases": cases,
    }


def run_pytest_lane(
    *,
    engine: str,
    python: str,
    timeout_s: float,
    junit_path: Path,
) -> dict[str, Any]:
    lane = ENGINE_LANES[engine]
    marker = lane["pytest_marker"]
    command = [
        python,
        "-m",
        "pytest",
        "tests",
        "-m",
        marker,
        "-q",
        "--tb=line",
        f"--junitxml={junit_path}",
    ]
    probe: dict[str, Any] = {
        "name": "native_pytest_lane",
        "status": "fail",
        "command": command,
        "returncode": None,
        "timed_out": False,
        "stdout": "",
        "stderr": "",
    }
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        probe["timed_out"] = True
        probe["stderr"] = (getattr(exc, "stderr", "") or "pytest timed out").strip()
        probe["result"] = _parse_junit(junit_path)
        return probe
    except OSError as exc:
        probe["stderr"] = str(exc)
        probe["result"] = _parse_junit(junit_path)
        return probe

    probe["returncode"] = completed.returncode
    probe["stdout"] = completed.stdout[-4000:]
    probe["stderr"] = completed.stderr[-4000:]
    probe["result"] = _parse_junit(junit_path)
    executed = (
        probe["result"]["passed"]
        + probe["result"]["failed"]
        + probe["result"]["errors"]
    )
    probe["status"] = (
        "pass"
        if completed.returncode == 0 and executed > 0 and probe["result"]["failed"] == 0
        else "fail"
    )
    return probe


@precondition(lambda receipt: isinstance(receipt, dict))
@postcondition(
    lambda result: isinstance(result[0], bool) and isinstance(result[1], list)
)
def validate_receipt(receipt: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a native lane receipt without trusting malformed process output."""

    reasons: list[str] = []
    if receipt.get("schema_version") != 1:
        reasons.append("schema_version must be 1")
    engine = receipt.get("engine")
    if engine not in ENGINE_LANES:
        reasons.append("engine is missing or unsupported")
    checker_hash = receipt.get("checker_source_sha256")
    if not isinstance(checker_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checker_hash
    ):
        reasons.append("checker source hash is missing or malformed")
    generated_at = receipt.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        reasons.append("generated_at timestamp is missing")
    else:
        try:
            _parse_iso8601(generated_at)
        except ValueError:
            reasons.append("generated_at timestamp is malformed")
    contract = receipt.get("contract_hashes")
    if not isinstance(contract, dict) or not contract:
        reasons.append("contract_hashes are missing")
    else:
        for path in CONTRACT_PATHS:
            digest = contract.get(path)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                reasons.append(f"contract hash missing or malformed: {path}")
    revision = receipt.get("repository_revision")
    if not isinstance(revision, str) or len(revision) < 7:
        reasons.append("repository_revision is missing or malformed")
    freshness = receipt.get("source_freshness")
    if not isinstance(freshness, dict) or freshness.get("status") not in {
        "clean",
        "dirty",
        "unknown",
    }:
        reasons.append("source freshness is missing or malformed")
    inventory = receipt.get("engine_inventory")
    if not isinstance(inventory, dict):
        reasons.append("engine inventory is missing")
    else:
        digest = inventory.get("source_sha256")
        if inventory.get("available") is True and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            reasons.append("engine module hash is missing while SDK is available")
    tests = receipt.get("tests")
    if not isinstance(tests, dict):
        reasons.append("tests summary is missing")
    else:
        for key in ("collected", "passed", "failed", "skipped", "errors", "executed"):
            value = tests.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                reasons.append(f"tests.{key} must be a non-negative integer")
        executed = tests.get("executed")
        if receipt.get("status") == "pass" and executed == 0:
            reasons.append("pass receipts must record nonzero executed test counts")
    probe = receipt.get("pytest_probe")
    if not isinstance(probe, dict):
        reasons.append("pytest probe is missing")
    elif probe.get("status") == "pass":
        if probe.get("returncode") != 0:
            reasons.append("pytest probe has nonzero returncode")
        if probe.get("timed_out") is not False:
            reasons.append("pytest probe timed out")
    return not reasons, reasons


def build_receipt(
    *,
    engine: str,
    out_dir: Path,
    python: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    if engine not in ENGINE_LANES:
        raise ValueError(f"unsupported engine: {engine}")
    lane = ENGINE_LANES[engine]
    interpreter = python or sys.executable
    out_dir.mkdir(parents=True, exist_ok=True)
    junit_path = out_dir / f"{engine}_native_lane_junit.xml"
    inventory = _engine_inventory(
        module_name=lane["python_module"],
        distribution=lane["distribution"],
        python=interpreter,
    )
    probe = run_pytest_lane(
        engine=engine,
        python=interpreter,
        timeout_s=timeout_s,
        junit_path=junit_path,
    )
    result = probe.get("result", {})
    executed = (
        int(result.get("passed", 0))
        + int(result.get("failed", 0))
        + int(result.get("errors", 0))
    )
    timestamp = (generated_at or datetime.now(tz=UTC)).isoformat()
    status = "pass" if probe.get("status") == "pass" and executed > 0 else "fail"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "scope": (
            "native-engine pytest lane only; no model, fit, acceptance, or release claim"
        ),
        "engine": engine,
        "platform": platform.platform(),
        "python": interpreter,
        "runner_hint": lane["runner_hint"],
        "generated_at": timestamp,
        "checker_source_sha256": _module_hash(str(Path(__file__).resolve())),
        "contract_hashes": contract_hashes(),
        "repository_revision": _repository_revision(),
        "source_freshness": _repository_freshness(),
        "engine_inventory": inventory,
        "tests": {
            "collected": int(result.get("collected", 0)),
            "passed": int(result.get("passed", 0)),
            "failed": int(result.get("failed", 0)),
            "skipped": int(result.get("skipped", 0)),
            "errors": int(result.get("errors", 0)),
            "executed": executed,
        },
        "test_cases": result.get("cases", []),
        "pytest_probe": probe,
        "failure_reasons": [],
    }
    if inventory.get("available") and inventory.get("version"):
        receipt["engine_version"] = inventory["version"]
    final_ok, final_reasons = validate_receipt(receipt)
    if not final_ok or status != "pass":
        receipt["status"] = "fail"
        receipt["failure_reasons"] = final_reasons
    return receipt


def write_receipt(receipt: dict[str, Any], out_dir: Path, engine: str) -> Path:
    filename = ENGINE_LANES[engine]["receipt_filename"]
    path = out_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _resolve_python(engine: str, venv: str | None) -> str:
    lane = ENGINE_LANES[engine]
    if venv:
        candidate = Path(venv)
        if candidate.is_dir():
            for name in ("bin/python", "Scripts/python.exe"):
                interpreter = candidate / name
                if interpreter.is_file():
                    return str(interpreter)
        if Path(venv).is_file():
            return venv
    default_venv = lane.get("default_venv")
    if default_venv:
        for name in ("bin/python", "Scripts/python.exe"):
            interpreter = Path(default_venv) / name
            if interpreter.is_file():
                return str(interpreter)
    return sys.executable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=tuple(ENGINE_LANES),
        required=True,
        help="native engine lane to execute",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for receipt JSON and junit artifacts",
    )
    parser.add_argument(
        "--venv",
        help="qualified interpreter or venv root (ControlTower opensim-10003 default)",
    )
    parser.add_argument("--python", help="explicit python executable")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--validate-only",
        type=Path,
        help="validate an existing receipt file and exit",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.validate_only:
        payload = json.loads(args.validate_only.read_text(encoding="utf-8"))
        ok, reasons = validate_receipt(payload)
        if not ok:
            for reason in reasons:
                print(reason, file=sys.stderr)
            return 1
        print(f"Valid: {args.validate_only}")
        return 0
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        print("--timeout must be finite and positive", file=sys.stderr)
        return 2
    python = args.python or _resolve_python(args.engine, args.venv)
    receipt = build_receipt(
        engine=args.engine,
        out_dir=args.out,
        python=python,
        timeout_s=args.timeout,
    )
    path = write_receipt(receipt, args.out, args.engine)
    encoded = json.dumps(receipt, indent=2, sort_keys=True)
    print(encoded)
    print(f"Wrote {path}", file=sys.stderr)
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
