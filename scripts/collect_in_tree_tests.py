"""Diagnostic collector for legacy in-tree ``src/**/tests`` suites (issue #7126).

The default ``pytest`` run excludes ``src/`` (``norecursedirs``), so the
grandfathered in-tree test directories tracked in
``scripts.check_test_layout.LEGACY_SRC_TEST_DIRS`` are not part of the blocking
signal. This script makes that exclusion explicit and inspectable: it runs a
``--collect-only`` pass (tolerating collection errors) over every legacy in-tree
directory that still exists and contains Python ``test_*.py`` files, and reports
which ones collect cleanly versus which still carry migration debt.

It is intentionally a *diagnostic*, not a gate: several legacy subtrees have
known import/relative-path collection errors (the very debt this policy
tracks). New tests must be added under ``tests/`` where the default lane
collects them; the ``Test Layout Guard`` blocks new in-tree directories.

Usage::

    python3 scripts/collect_in_tree_tests.py        # collect-only diagnostic
    make test-in-tree
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.check_test_layout import LEGACY_SRC_TEST_DIRS

_REPO_ROOT = Path(__file__).resolve().parents[1]


def in_tree_dirs_with_python_tests(repo_root: Path | None = None) -> list[Path]:
    """Return existing legacy in-tree dirs that contain ``test_*.py`` files.

    Postcondition: every returned path is an existing directory under
    ``repo_root`` that is listed in ``LEGACY_SRC_TEST_DIRS`` and holds at least
    one ``test_*.py`` file.
    """
    root = repo_root or _REPO_ROOT
    found: list[Path] = []
    for rel in sorted(LEGACY_SRC_TEST_DIRS):
        candidate = root / rel
        if candidate.is_dir() and any(candidate.glob("test_*.py")):
            found.append(candidate)
    return found


def main(argv: list[str] | None = None) -> int:
    """Run a collect-only diagnostic over the legacy in-tree test dirs."""
    del argv  # no options; kept for symmetry with other CI scripts
    targets = in_tree_dirs_with_python_tests()
    if not targets:
        print("No legacy in-tree test directories with Python tests found.")
        return 0

    print(
        f"Inspecting {len(targets)} legacy in-tree test directory(ies) "
        "(collect-only; see docs/development/in_tree_test_policy.md):"
    )
    for target in targets:
        rel = target.relative_to(_REPO_ROOT).as_posix()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "-m",
                "pytest",
                str(target),
                "--collect-only",
                "-q",
                "--continue-on-collection-errors",
                "--override-ini=norecursedirs=",
                "-p",
                "no:cacheprovider",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        status = "ok" if result.returncode == 0 else "has collection debt"
        print(f"  [{status}] {rel}")

    # Always succeed: this is a diagnostic, not a gate (issue #7126).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
