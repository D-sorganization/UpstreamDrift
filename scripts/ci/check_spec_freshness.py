#!/usr/bin/env python3
"""Decide whether a pull request keeps SPEC.md fresh (spec-check.yml).

A pull request that changes source files must either edit ``SPEC.md`` or carry
a per-PR change fragment, ``changes/<issue>-<slug>.md``
(Repository_Management#1894, rolled out here by Repository_Management#1976).
After merge, ``collate-changes.yml`` folds each fragment into the SPEC.md
change log keyed by the real pull request number, so concurrently queued pull
requests stop conflicting over the same table.

A fragment only counts when the pull request carries at least one and *every*
fragment validates, exactly like ``_has_valid_fragment`` in Repository_Management's
``shared_scripts/fleet_hooks.py``. Validation reuses the vendored
``shared_scripts/changes_fragment.py``; when that module is absent, fragments
never satisfy the gate (fail closed).

Prints ``key=value`` lines suitable for ``$GITHUB_OUTPUT``: ``source_changed``,
``spec_changed``, ``fragment_valid`` and ``needs_update``. Always exits 0; the
workflow decides how to fail on ``needs_update=true``.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

log = logging.getLogger("check_spec_freshness")

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAGMENT_MODULE = Path("shared_scripts") / "changes_fragment.py"

# Kept in step with the patterns spec-check.yml has always used.
SOURCE_PREFIXES = ("src/", "tests/", "config/")
SOURCE_FILES = frozenset(
    {
        "pyproject.toml",
        "Cargo.toml",
        "CMakeLists.txt",
        "package.json",
        "requirements.txt",
    }
)


def is_source(path: str) -> bool:
    """True when ``path`` is a file whose change requires a SPEC update."""
    return path.startswith(SOURCE_PREFIXES) or path in SOURCE_FILES


def load_changes_fragment(repo_root: Path) -> ModuleType | None:
    """Load the vendored fragment module by path, or ``None`` if absent."""
    module_path = repo_root / FRAGMENT_MODULE
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_spec_freshness_changes_fragment", module_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the fragment schema defines dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def has_valid_fragment(files: Sequence[str], repo_root: Path) -> bool:
    """True when ``files`` carry at least one fragment and all of them are valid.

    Deleted paths (not on disk) and non-fragment files under ``changes/``,
    such as its README, are ignored.
    """
    module = load_changes_fragment(repo_root)
    if module is None:
        return False
    fragments = [
        repo_root / path
        for path in files
        if module.is_fragment_path(path) and (repo_root / path).is_file()
    ]
    findings = [
        f"{path}: {finding}"
        for path in fragments
        for finding in module.validate_fragment_file(path)
    ]
    for finding in findings:
        log.error("invalid change fragment %s", finding)
    return bool(fragments) and not findings


def decide(files: Sequence[str], repo_root: Path) -> dict[str, bool]:
    """Return the freshness outputs for the changed ``files``.

    Postcondition: ``needs_update`` is true exactly when a source file changed
    and neither SPEC.md nor a valid fragment accompanies it.
    """
    source_changed = any(is_source(path) for path in files)
    spec_changed = "SPEC.md" in files
    fragment_valid = has_valid_fragment(files, repo_root)
    return {
        "source_changed": source_changed,
        "spec_changed": spec_changed,
        "fragment_valid": fragment_valid,
        "needs_update": source_changed and not (spec_changed or fragment_valid),
    }


def changed_files(base: str, repo_root: Path) -> list[str]:
    """Paths changed between ``base`` and ``HEAD``."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: print the outputs for ``--base`` or explicit paths."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--base", help="base commit; diffed against HEAD")
    parser.add_argument("files", nargs="*", help="changed paths (instead of --base)")
    args = parser.parse_args(argv)
    if bool(args.base) == bool(args.files):
        parser.error("give exactly one of --base or explicit files")
    files = args.files or changed_files(args.base, args.repo_root)
    for key, value in decide(files, args.repo_root).items():
        sys.stdout.write(f"{key}={str(value).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
