"""Mypy runner wrapper that filters out excluded files."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    # Fallback to manual parsing if on Python < 3.11
    tomllib = None  # type: ignore[assignment]


def load_exclusions() -> list[str]:
    """Load mypy exclude paths from pyproject.toml."""
    if tomllib is not None:
        try:
            with open("pyproject.toml", "rb") as f:
                data = tomllib.load(f)
            exclusions = data.get("tool", {}).get("mypy", {}).get("exclude", [])
            if isinstance(exclusions, list):
                return [str(x) for x in exclusions]
        except Exception:  # noqa: BLE001
            pass

    # Fallback manual parsing
    exclusions_list: list[str] = []
    try:
        with open("pyproject.toml", encoding="utf-8") as f:
            in_exclude = False
            for line in f:
                line_str = line.strip()
                if "exclude = [" in line_str:
                    in_exclude = True
                elif line_str.startswith("]") and in_exclude:
                    in_exclude = False
                elif in_exclude:
                    m = re.search(r'"([^"]+)"', line_str)
                    if m:
                        exclusions_list.append(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    return exclusions_list


def _sanitized_mypy_env() -> dict[str, str]:
    """Remove shared package roots that make mypy see duplicate modules."""
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    if not pythonpath:
        return env

    repo_root = Path.cwd().resolve()
    duplicate_roots = {
        repo_root / "src" / "shared" / "python",
        repo_root / "vendor" / "ud-tools" / "src" / "shared" / "python",
    }

    kept_entries: list[str] = []
    for entry in pythonpath.split(os.pathsep):
        if not entry:
            continue
        with contextlib.suppress(OSError):
            if Path(entry).expanduser().resolve() in duplicate_roots:
                continue
        kept_entries.append(entry)

    if kept_entries:
        env["PYTHONPATH"] = os.pathsep.join(kept_entries)
    else:
        env.pop("PYTHONPATH", None)
    return env


def main() -> None:
    """Filter files using exclusions and run mypy."""
    exclusions = load_exclusions()
    exclude_patterns = []
    for pat in exclusions:
        with contextlib.suppress(Exception):
            exclude_patterns.append(re.compile(pat))

    args = sys.argv[1:]
    files: list[str] = []
    other_args: list[str] = []
    for arg in args:
        if not arg.startswith("-") and arg.endswith(".py"):
            files.append(arg)
        else:
            other_args.append(arg)

    filtered_files: list[str] = []
    for f in files:
        f_norm = f.replace("\\", "/")
        is_excluded = False
        for pat in exclude_patterns:
            if pat.search(f_norm):
                is_excluded = True
                break
        if is_excluded:
            print(f"Skipping excluded file: {f}")
        else:
            filtered_files.append(f)

    if files and not filtered_files:
        print("All target files were excluded. Skipping mypy run.")
        sys.exit(0)

    cmd = [sys.executable, "-m", "mypy"] + other_args + filtered_files
    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False, env=_sanitized_mypy_env())
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
