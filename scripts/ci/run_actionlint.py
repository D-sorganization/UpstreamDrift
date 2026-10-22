#!/usr/bin/env python3
"""Run rhysd/actionlint on GitHub workflow files (UD-CI / issue #13572 regression)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ACTIONLINT_VERSION = "1.7.7"
DOWNLOAD_SCRIPT_URL = (
    "https://raw.githubusercontent.com/rhysd/actionlint/main/"
    "scripts/download-actionlint.bash"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config() -> Path:
    return _repo_root() / ".github" / "actionlint.yaml"


def resolve_actionlint_binary(root: Path | None = None) -> Path:
    """Return an actionlint executable path, downloading a pinned release if needed."""
    root = root or _repo_root()
    env_path = os.environ.get("ACTIONLINT_PATH")
    if env_path:
        candidate = Path(env_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"ACTIONLINT_PATH does not exist: {candidate}")
        return candidate

    for name in ("actionlint", "./actionlint"):
        found = shutil.which(name)
        if found:
            return Path(found)
        local = root / name.lstrip("./")
        if local.is_file():
            return local

    script_body = urllib.request.urlopen(DOWNLOAD_SCRIPT_URL, timeout=60).read()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".bash", delete=False) as tmp:
        tmp.write(script_body)
        script_path = Path(tmp.name)
    try:
        subprocess.run(
            ["bash", str(script_path), ACTIONLINT_VERSION],
            cwd=root,
            check=True,
        )
    finally:
        script_path.unlink(missing_ok=True)
    downloaded = root / "actionlint"
    if not downloaded.is_file():
        raise RuntimeError("actionlint download completed but binary is missing")
    dest = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "actionlint"
    shutil.copy2(downloaded, dest)
    dest.chmod(0o755)
    downloaded.unlink(missing_ok=True)
    return dest


def run_actionlint(
    paths: list[Path],
    *,
    config: Path | None = None,
    root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run actionlint on ``paths``; raises ``ValueError`` when ``paths`` is empty."""
    if not paths:
        raise ValueError("paths must not be empty")

    root = root or _repo_root()
    config = config or _default_config()
    if not config.is_file():
        raise FileNotFoundError(f"actionlint config not found: {config}")

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "workflow file(s) not found: " + ", ".join(str(p) for p in missing)
        )

    binary = resolve_actionlint_binary(root)
    cmd = [str(binary), f"-config-file={config}", *[str(p) for p in paths]]
    return subprocess.run(
        cmd,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Workflow YAML files to lint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="actionlint config file (default: .github/actionlint.yaml)",
    )
    args = parser.parse_args(argv)

    try:
        result = run_actionlint(args.paths, config=args.config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
