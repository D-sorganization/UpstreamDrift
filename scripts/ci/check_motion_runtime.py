"""Qualify the optional Pinocchio, Pink, and Crocoddyl motion runtime.

The checks deliberately run in child interpreters.  These packages load native
libraries, and an import or solver failure must not poison the process running
the caller's test suite.  The command emits a JSON receipt and returns nonzero
unless every required component and named probe succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_CAPTURE_CHARS = 4_000
DEFAULT_TIMEOUT_SECONDS = 90.0
REQUIRED_COMPONENTS = ("pinocchio", "pink", "crocoddyl", "qpsolvers", "quadprog")
REQUIRED_PROBES = (
    "crocoddyl_abi",
    "pink_hard_equality",
    "pink_infeasible_qp",
)
TOLERANCES = {
    "pink_hard_equality_norm": 1e-10,
    "pink_hard_equality_step": 1e-10,
}


def bounded_output(value: Any, limit: int = MAX_CAPTURE_CHARS) -> str:
    """Convert captured process output to bounded, useful diagnostic text."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    if len(text) <= limit:
        return text
    marker = f"\n...[truncated {len(text) - limit} chars]...\n"
    keep = max(0, limit - len(marker))
    # Keep the tail because native-library failures commonly report the cause
    # at the end of stderr.  A short head remains to identify the command.
    head = keep // 3
    return text[:head] + marker + text[-(keep - head) :]


def run_isolated_probe(
    name: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    python: str | None = None,
) -> dict[str, Any]:
    """Run one probe in a fresh interpreter and retain bounded diagnostics."""

    interpreter = python or sys.executable
    command = [interpreter, str(Path(__file__).resolve()), "--child", name]
    result: dict[str, Any] = {
        "name": name,
        "status": "fail",
        "command": command,
        "returncode": None,
        "timed_out": False,
        "stdout": "",
        "stderr": "",
    }
    try:
        timeout_valid = math.isfinite(timeout_s) and timeout_s > 0
    except (TypeError, ValueError):
        timeout_valid = False
    if not timeout_valid:
        result["stderr"] = "timeout must be finite and positive"
        return result
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
        result["timed_out"] = True
        result["stdout"] = bounded_output(getattr(exc, "stdout", None))
        result["stderr"] = bounded_output(getattr(exc, "stderr", None))
        result["stderr"] = (result["stderr"] + "\nprobe timed out").strip()
        return result
    except (FileNotFoundError, OSError) as exc:
        result["stderr"] = bounded_output(str(exc))
        return result

    result["returncode"] = completed.returncode
    result["stdout"] = bounded_output(completed.stdout)
    result["stderr"] = bounded_output(completed.stderr)
    result["status"] = "pass" if completed.returncode == 0 else "fail"
    return result


def _module_hash(module_file: str | None) -> str | None:
    if not module_file:
        return None
    try:
        digest = hashlib.sha256()
        with open(module_file, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError):
        return None


def _child_inventory() -> int:
    """Print component metadata without hiding a partial import failure."""

    components: dict[str, dict[str, Any]] = {}
    for distribution, module_name in (
        ("pinocchio", "pinocchio"),
        ("pink", "pink"),
        ("crocoddyl", "crocoddyl"),
        ("qpsolvers", "qpsolvers"),
        ("quadprog", "quadprog"),
    ):
        item: dict[str, Any] = {"available": False, "distribution": distribution}
        try:
            module = importlib.import_module(module_name)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                # Conda packages can omit Python dist-info while exposing a
                # module version.  This is still useful runtime evidence.
                version = getattr(module, "__version__", None)
            item.update(
                available=version is not None,
                version=version,
                module_file=str(getattr(module, "__file__", "")),
                source_sha256=_module_hash(getattr(module, "__file__", None)),
            )
        except Exception as exc:  # noqa: BLE001 - native imports vary by wheel
            item["error"] = f"{type(exc).__name__}: {exc}"
        components[distribution] = item
    print(
        json.dumps(
            {
                "child_python": {
                    "executable": sys.executable,
                    "version": platform.python_version(),
                },
                "components": components,
            },
            sort_keys=True,
        )
    )
    return 0 if all(item["available"] for item in components.values()) else 1


def _child_crocoddyl() -> int:
    # This is the repository's existing ABI probe.  Keep it as the authority
    # for FDDP/native-library compatibility rather than duplicating its model.
    sys.path.insert(0, str(REPO_ROOT))
    from src.shared.python.optimization.crocoddyl_backend import (
        crocoddyl_stack_healthy,
    )

    healthy, reason = crocoddyl_stack_healthy(timeout_s=60.0)
    print(
        json.dumps(
            {
                "child_python": {
                    "executable": sys.executable,
                    "version": platform.python_version(),
                },
                "healthy": healthy,
                "reason": reason,
            }
        )
    )
    return 0 if healthy else 1


def _child_pink_equality() -> int:
    import numpy as np
    import pinocchio as pin
    import pink
    from pink.configuration import Configuration
    from pink.tasks import PostureTask

    dt = 0.1
    model = pin.buildSampleModelManipulator()
    q = pin.neutral(model)
    configuration = Configuration(model, model.createData(), q)
    task = PostureTask(cost=1.0)
    target = q.copy()
    target[0] = 0.2
    task.set_target(target)
    velocity = pink.solve_ik(
        configuration,
        [],
        dt,
        solver="quadprog",
        damping=1e-12,
        constraints=[task],
    )
    q_next = pin.integrate(model, q, velocity * dt)
    error_norm = float(np.linalg.norm(pin.difference(model, q_next, target)))
    step_error = abs(float(q_next[0]) - 0.2)
    payload = {
        "child_python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "solver": "quadprog",
        "dt_s": dt,
        "q_next0": float(q_next[0]),
        "velocity0": float(velocity[0]),
        "velocity": np.asarray(velocity).tolist(),
        "q_next": np.asarray(q_next).tolist(),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "difference_norm": error_norm,
        "step_error": step_error,
    }
    print(json.dumps(payload, sort_keys=True))
    return int(
        not np.isfinite(velocity).all()
        or not np.isfinite(q_next).all()
        or not math.isfinite(error_norm)
        or not math.isfinite(step_error)
        or error_norm > TOLERANCES["pink_hard_equality_norm"]
        or step_error > TOLERANCES["pink_hard_equality_step"]
    )


def _child_pink_infeasible() -> int:
    import pinocchio as pin
    import pink
    from pink.configuration import Configuration
    from pink.tasks import PostureTask

    model = pin.buildSampleModelManipulator()
    q = pin.neutral(model)
    configuration = Configuration(model, model.createData(), q)
    first = PostureTask(cost=1.0)
    first_target = q.copy()
    first_target[0] = 0.2
    first.set_target(first_target)
    second = PostureTask(cost=1.0)
    second_target = q.copy()
    second_target[0] = -0.2
    second.set_target(second_target)
    try:
        pink.solve_ik(
            configuration,
            [],
            0.1,
            solver="quadprog",
            damping=1e-12,
            constraints=[first, second],
        )
    except Exception as exc:  # noqa: BLE001 - Pink's exception varies by release
        payload = {
            "child_python": {
                "executable": sys.executable,
                "version": platform.python_version(),
            },
            "raised": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": str(exc),
        }
        print(json.dumps(payload, sort_keys=True))
        return 0 if type(exc).__name__ == "NoSolutionFound" else 1
    print(
        json.dumps(
            {
                "child_python": {
                    "executable": sys.executable,
                    "version": platform.python_version(),
                },
                "raised": None,
            },
            sort_keys=True,
        )
    )
    return 1


def _run_child(name: str) -> int:
    if name == "inventory":
        return _child_inventory()
    if name == "crocoddyl_abi":
        return _child_crocoddyl()
    if name == "pink_hard_equality":
        return _child_pink_equality()
    if name == "pink_infeasible_qp":
        return _child_pink_infeasible()
    print(f"unknown child probe: {name}", file=sys.stderr)
    return 2


def _parse_json_output(result: dict[str, Any]) -> dict[str, Any] | None:
    try:
        payload = json.loads(result["stdout"])
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _repository_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and completed.returncode == 0:
        return completed.stdout.strip()

    # A Windows-created worktree has a .git file containing a Windows path;
    # WSL's git cannot resolve that path.  Read the worktree HEAD directly so
    # the receipt still records the exact revision in either host.
    try:
        marker = (REPO_ROOT / ".git").read_text(encoding="utf-8").strip()
        match = re.fullmatch(r"gitdir:\s*(.+)", marker, flags=re.IGNORECASE)
        if not match:
            return None
        gitdir = Path(match.group(1).strip().replace("\\", "/"))
        drive = re.match(r"^([A-Za-z]):/(.*)$", str(gitdir))
        if drive:
            gitdir = Path("/mnt") / drive.group(1).lower() / drive.group(2)
        head_text = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if not head_text.startswith("ref: "):
            return head_text or None
        ref = head_text[5:]
        common = gitdir / "commondir"
        common_dir = gitdir
        if common.exists():
            common_dir = (gitdir / common.read_text(encoding="utf-8").strip()).resolve()
        ref_file = common_dir / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = common_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#") and not line.startswith("^"):
                    revision, packed_ref = line.split(" ", 1)
                    if packed_ref == ref:
                        return revision
    except (OSError, ValueError):
        pass
    return None


def _repository_freshness() -> dict[str, Any]:
    """Report whether the checked source paths are clean on this host."""

    paths = [
        "scripts/ci/check_motion_runtime.py",
        "scripts/config/motion_runtime/environment.yml",
        "scripts/config/motion_runtime/linux-64.explicit.txt",
    ]
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unknown", "reason": bounded_output(str(exc))}
    if completed.returncode != 0:
        return {
            "status": "unknown",
            "reason": bounded_output(completed.stderr or "git status failed"),
        }
    output = bounded_output(completed.stdout)
    return {
        "status": "dirty" if output else "clean",
        "paths": paths,
        "details": output,
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_receipt_metadata(receipt: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if receipt.get("status") != "pass":
        reasons.append("receipt status is not pass")
    if receipt.get("tolerances") != TOLERANCES:
        reasons.append("numeric tolerances do not match canonical values")
    checker_hash = receipt.get("checker_source_sha256")
    if not isinstance(checker_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checker_hash
    ):
        reasons.append("checker source hash is missing or malformed")
    freshness = receipt.get("source_freshness")
    if not isinstance(freshness, dict) or freshness.get("status") not in {
        "clean",
        "dirty",
        "unknown",
    }:
        reasons.append("source freshness is missing or malformed")
    child_python = receipt.get("child_python")
    if not isinstance(child_python, dict) or not all(
        isinstance(child_python.get(key), str) and child_python[key]
        for key in ("version", "executable")
    ):
        reasons.append("child interpreter version is missing")
    return reasons


def _validate_components(receipt: dict[str, Any]) -> list[str]:
    components = receipt.get("components")
    if not isinstance(components, dict):
        return ["component inventory is missing"]
    reasons: list[str] = []
    for name in REQUIRED_COMPONENTS:
        item = components.get(name)
        valid_hash = isinstance(item, dict) and isinstance(
            item.get("source_sha256"), str
        )
        if valid_hash:
            valid_hash = bool(re.fullmatch(r"[0-9a-f]{64}", item["source_sha256"]))
        if not (
            isinstance(item, dict)
            and item.get("available") is True
            and isinstance(item.get("version"), str)
            and bool(item["version"])
            and isinstance(item.get("module_file"), str)
            and bool(item["module_file"])
            and valid_hash
        ):
            reasons.append(f"required component unavailable: {name}")
    return reasons


def _probe_map(receipt: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    probes = receipt.get("probes")
    if not isinstance(probes, list):
        return {}, ["probe list is missing or malformed"]
    by_name: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    names: list[str] = []
    for item in probes:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            reasons.append("malformed probe entry")
            continue
        name = item["name"]
        names.append(name)
        if name in by_name:
            reasons.append("duplicate probe names in receipt")
        by_name[name] = item
    if (
        len(names) != len(set(names))
        and "duplicate probe names in receipt" not in reasons
    ):
        reasons.append("duplicate probe names in receipt")
    return by_name, reasons


def _validate_probe_processes(
    receipt: dict[str, Any], by_name: dict[str, dict[str, Any]]
) -> list[str]:
    reasons: list[str] = []
    for name in REQUIRED_PROBES:
        item = by_name.get(name)
        if not isinstance(item, dict):
            reasons.append(f"required probe failed: {name}")
            continue
        if item.get("status") != "pass":
            reasons.append(f"required probe failed: {name}")
        if item.get("returncode") != 0:
            reasons.append(f"required probe has nonzero returncode: {name}")
        if item.get("timed_out") is not False:
            reasons.append(f"required probe timed out: {name}")
        if not isinstance(item.get("result"), dict):
            reasons.append(f"required probe has malformed result: {name}")
    inventory = receipt.get("inventory_probe")
    if (
        not isinstance(inventory, dict)
        or inventory.get("status") != "pass"
        or inventory.get("returncode") != 0
        or inventory.get("timed_out") is not False
    ):
        reasons.append("component inventory process failed")
    return reasons


def _validate_probe_semantics(by_name: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    croc = by_name.get("crocoddyl_abi", {}).get("result", {})
    if not isinstance(croc, dict) or croc.get("healthy") is not True:
        reasons.append("crocoddyl ABI probe did not report healthy=true")
    equality = by_name.get("pink_hard_equality", {}).get("result", {})
    if not isinstance(equality, dict):
        equality = {}
    for field, dimension in (("q_next", "nq"), ("velocity", "nv")):
        values = equality.get(field)
        count = equality.get(dimension)
        if not (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            and isinstance(values, list)
            and len(values) == count
            and all(_finite_number(value) for value in values)
        ):
            reasons.append(f"Pink equality {field} vector is malformed or nonfinite")
    for field in ("difference_norm", "step_error", "q_next0", "velocity0"):
        if not _finite_number(equality.get(field)):
            reasons.append(f"Pink equality result is not finite: {field}")
    for field, tolerance_key in (
        ("difference_norm", "pink_hard_equality_norm"),
        ("step_error", "pink_hard_equality_step"),
    ):
        value = equality.get(field)
        if _finite_number(value) and value < 0:
            reasons.append(f"Pink equality {field} must be non-negative")
        if _finite_number(value) and value > TOLERANCES[tolerance_key]:
            reasons.append(f"Pink equality {field} exceeds tolerance")
    infeasible = by_name.get("pink_infeasible_qp", {}).get("result", {})
    if not isinstance(infeasible, dict) or not (
        isinstance(infeasible.get("raised"), str)
        and infeasible["raised"].endswith(".NoSolutionFound")
        and isinstance(infeasible.get("message"), str)
        and bool(infeasible["message"])
    ):
        reasons.append("Pink infeasible-QP probe did not raise NoSolutionFound")
    return reasons


def validate_receipt(receipt: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a runtime receipt without trusting malformed process output."""

    reasons = _validate_receipt_metadata(receipt)
    reasons.extend(_validate_components(receipt))
    by_name, probe_reasons = _probe_map(receipt)
    reasons.extend(probe_reasons)
    reasons.extend(_validate_probe_processes(receipt, by_name))
    reasons.extend(_validate_probe_semantics(by_name))
    return not reasons, reasons


def build_receipt(*, timeout_s: float, python: str | None) -> dict[str, Any]:
    try:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout must be finite and positive")
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be finite and positive") from exc
    inventory = run_isolated_probe("inventory", timeout_s=timeout_s, python=python)
    inventory_payload = _parse_json_output(inventory) or {}
    components = inventory_payload.get("components", {})
    probes: list[dict[str, Any]] = []
    for name in REQUIRED_PROBES:
        result = run_isolated_probe(name, timeout_s=timeout_s, python=python)
        payload = _parse_json_output(result)
        if payload is not None:
            result["result"] = payload
        probes.append(result)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "scope": "runtime capability only; no model, fitting, renderer, or acceptance claim",
        "platform": platform.platform(),
        "python": python or sys.executable,
        "checker_source_sha256": _module_hash(str(Path(__file__).resolve())),
        "source_freshness": _repository_freshness(),
        "repository_revision": _repository_revision(),
        "required_components": list(REQUIRED_COMPONENTS),
        "required_probes": list(REQUIRED_PROBES),
        "tolerances": TOLERANCES,
        "components": components,
        "probes": probes,
        "inventory_probe": inventory,
        "child_python": inventory_payload.get("child_python"),
    }
    ok, reasons = validate_receipt(receipt)
    receipt["status"] = "pass" if ok else "fail"
    receipt["failure_reasons"] = reasons
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("inventory", *REQUIRED_PROBES))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--python", help="qualified interpreter to use for child probes"
    )
    parser.add_argument(
        "--receipt", type=Path, help="write the JSON receipt to this path"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.child:
        return _run_child(args.child)
    try:
        timeout_valid = math.isfinite(args.timeout) and args.timeout > 0
    except (TypeError, ValueError):
        timeout_valid = False
    if not timeout_valid:
        print("--timeout must be finite and positive", file=sys.stderr)
        return 2
    receipt = build_receipt(timeout_s=args.timeout, python=args.python)
    encoded = json.dumps(receipt, indent=2, sort_keys=True)
    print(encoded)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(encoded + "\n", encoding="utf-8")
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
