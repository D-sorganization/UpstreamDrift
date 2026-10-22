#!/usr/bin/env python3
"""Provider-agnostic agent hook runner driven by fleet-hooks/manifest.yml."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import yaml

from shared_scripts.run_mypy_diff import resolve_changed_files
from src.contracts import require

log = logging.getLogger(__name__)
CheckStatus = Literal["pass", "fail", "skip"]
REPO_REGEN_CANDIDATES = (
    "scripts/agent-check-regen",
    "scripts/agent-check-regen.py",
    "scripts/agent-check-regen.sh",
)


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: CheckStatus
    message: str
    duration_ms: int
    files: tuple[str, ...] = ()


@dataclass
class RunReport:
    schema_version: int
    enabled: bool
    exit_code: int
    changed_files: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "exit_code": self.exit_code,
            "changed_files": self.changed_files,
            "checks": [asdict(c) for c in self.checks],
        }
        return json.dumps(payload, indent=2) + "\n"


def _result(
    check_id: str,
    status: CheckStatus,
    message: str,
    files: Sequence[str],
    duration_ms: int,
) -> CheckResult:
    return CheckResult(
        id=check_id,
        status=status,
        message=message,
        duration_ms=duration_ms,
        files=tuple(files),
    )


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "fleet-hooks" / "manifest.yml"
    require(path.is_file(), f"missing fleet hook manifest: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "manifest must be a mapping")
    require(int(data.get("schema_version", 0)) == 1, "unsupported manifest schema")
    checks = data.get("checks")
    require(isinstance(checks, list), "manifest checks must be a list")
    require(bool(checks), "manifest checks must be non-empty")
    return cast(dict[str, Any], data)


def _glob_match(norm: str, pattern: str) -> bool:
    path = PurePosixPath(norm)
    if path.match(pattern) or fnmatch.fnmatch(norm, pattern):
        return True
    if pattern.startswith("**/"):
        tail = pattern[3:]
        return path.match(tail) or fnmatch.fnmatch(norm, tail)
    return False


def filter_paths(
    root: Path,
    files: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str] = (),
) -> list[str]:
    out: list[str] = []
    for rel in files:
        norm = rel.replace("\\", "/")
        if not (root / norm).is_file():
            continue
        if not any(_glob_match(norm, g) for g in include):
            continue
        if any(_glob_match(norm, g) for g in exclude):
            continue
        out.append(norm)
    return sorted(set(out))


def _ruff_exe() -> str | None:
    try:
        import ruff  # noqa: F401

        return sys.executable
    except ImportError:
        return shutil.which("ruff")


def _run(
    cmd: list[str],
    root: Path,
    timeout: int,
    *,
    ruff_module: bool = False,
) -> subprocess.CompletedProcess[str]:
    if ruff_module:
        cmd = [sys.executable, "-m", "ruff", *cmd[1:]]
    return subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )


def _timed(
    check_id: str,
    files: list[str],
    timeout: int,
    runner: Any,
) -> CheckResult:
    if not files:
        return _result(check_id, "skip", "no matching files in change set", files, 0)
    started = time.monotonic()
    try:
        ok, message = runner()
    except subprocess.TimeoutExpired:
        return _result(
            check_id, "fail", f"timed out after {timeout}s", files, timeout * 1000
        )
    ms = int((time.monotonic() - started) * 1000)
    return _result(check_id, "pass" if ok else "fail", message, files, ms)


def run_ruff(
    check_id: str, subcmd: str, files: list[str], root: Path, timeout: int
) -> CheckResult:
    exe = _ruff_exe()
    if not exe:
        return _result(check_id, "skip", "ruff not on PATH", files, 0)

    def _go() -> tuple[bool, str]:
        proc = _run(
            ["ruff", subcmd, *files]
            if subcmd == "check"
            else ["ruff", "format", "--check", *files],
            root,
            timeout,
            ruff_module=exe == sys.executable,
        )
        out = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode == 0, out or f"ruff {subcmd} failed"

    return _timed(check_id, files, timeout, _go)


def run_detect_secrets(
    check_id: str, files: list[str], root: Path, timeout: int
) -> CheckResult:
    if not (root / ".secrets.baseline").is_file():
        return _result(
            check_id, "skip", "no .secrets.baseline; skipping detect-secrets", files, 0
        )
    if not shutil.which("detect-secrets"):
        try:
            import detect_secrets  # noqa: F401
        except ImportError:
            return _result(check_id, "skip", "detect-secrets not on PATH", files, 0)

    def _go() -> tuple[bool, str]:
        cmd = (
            ["detect-secrets", "scan", "--baseline", ".secrets.baseline", *files]
            if shutil.which("detect-secrets")
            else [
                sys.executable,
                "-m",
                "detect_secrets",
                "scan",
                "--baseline",
                ".secrets.baseline",
                *files,
            ]
        )
        proc = _run(cmd, root, timeout)
        out = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode == 0, out or "detect-secrets failed"

    return _timed(check_id, files, timeout, _go)


def run_actionlint(
    check_id: str, files: list[str], root: Path, timeout: int
) -> CheckResult:
    tool = shutil.which("actionlint")
    if not tool:
        return _result(check_id, "skip", "actionlint not on PATH", files, 0)
    config = root / ".github" / "actionlint.yaml"

    def _go() -> tuple[bool, str]:
        failures: list[str] = []
        for rel in files:
            cmd = [tool]
            if config.is_file():
                cmd.extend(["-config-file", str(config)])
            cmd.append(rel)
            proc = _run(cmd, root, timeout)
            if proc.returncode != 0:
                failures.append((proc.stdout or proc.stderr or rel).strip())
        return not failures, "\n".join(failures) or "actionlint clean"

    return _timed(check_id, files, timeout, _go)


def run_repo_regen(
    check_id: str, files: list[str], root: Path, timeout: int
) -> CheckResult:
    script = next(
        (root / p for p in REPO_REGEN_CANDIDATES if (root / p).is_file()), None
    )
    if script is None:
        return _result(
            check_id,
            "skip",
            "no scripts/agent-check-regen hook in repository",
            files,
            0,
        )
    cmd = [sys.executable, str(script)] if script.suffix == ".py" else [str(script)]

    def _go() -> tuple[bool, str]:
        proc = _run(cmd, root, timeout)
        out = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode == 0, out or "repo regen hook failed"

    return _timed(check_id, files, timeout, _go)


def run_check(
    spec: dict[str, Any], changed: list[str], root: Path, defaults: dict[str, Any]
) -> CheckResult:
    check_id = str(spec["id"])
    kind = str(spec.get("kind") or check_id)
    timeout = int(spec.get("timeout_seconds") or defaults.get("timeout_seconds") or 120)
    files = filter_paths(
        root,
        changed,
        spec.get("globs") or [],
        spec.get("exclude_globs") or [],
    )
    tool = spec.get("tool")
    if kind == "detect-secrets":
        pass
    elif kind == "repo-regen":
        pass
    elif tool == "ruff" and not _ruff_exe():
        return _result(check_id, "skip", "ruff not on PATH", files, 0)
    elif tool and tool != "ruff" and not shutil.which(str(tool)):
        return _result(check_id, "skip", f"{tool} not on PATH", files, 0)

    if kind == "ruff-format":
        return run_ruff(check_id, "format", files, root, timeout)
    if kind == "ruff-check":
        return run_ruff(check_id, "check", files, root, timeout)
    if kind == "detect-secrets":
        return run_detect_secrets(check_id, files, root, timeout)
    if kind == "actionlint":
        return run_actionlint(check_id, files, root, timeout)
    if kind == "repo-regen":
        return run_repo_regen(check_id, files, root, timeout)
    return _result(check_id, "skip", f"unknown check kind: {kind}", files, 0)


def run_all(root: Path | None = None) -> RunReport:
    base = (root or Path.cwd()).resolve()
    if os.environ.get("AGENT_HOOKS", "").strip().lower() in {"0", "false", "off", "no"}:
        return RunReport(schema_version=1, enabled=False, exit_code=0)
    manifest = load_manifest(base)
    changed = [f.replace("\\", "/") for f in resolve_changed_files(cwd=base)]
    results = []
    for spec in manifest["checks"]:
        require(isinstance(spec, dict), "each check must be a mapping")
        require(bool(spec.get("id")), "each check needs an id")
        results.append(run_check(spec, changed, base, manifest.get("defaults") or {}))
    code = 1 if any(r.status == "fail" for r in results) else 0
    return RunReport(1, True, code, changed, results)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    del argv
    report = run_all()
    sys.stdout.write(report.to_json())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
