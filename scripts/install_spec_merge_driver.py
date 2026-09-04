#!/usr/bin/env python3
"""Register the ``spec-rows`` merge driver in this clone.

Two halves have to arrive together, and **neither can be committed**:

* the attribute ``SPEC.md merge=spec-rows``, and
* the driver *definition* in git config.

A committed ``.gitattributes`` naming the driver is a footgun, not a
convenience. Git does **not** fall back to the default driver when the named
driver is unregistered — it aborts::

    $ git merge feature-b
    fatal: custom merge driver spec-rows lacks command line.

so every clone that has not run this script — a CI checkout, a fresh clone,
another agent's worktree — would fail to merge SPEC.md at all. That is
strictly worse than the conflict the driver exists to avoid. The attribute
therefore goes in ``$GIT_COMMON_DIR/info/attributes``, which is per-clone,
uncommitted, shared across worktrees, and written here in the same breath as
the config, so the two can never be out of step.

Run once per clone (``scripts/install_workspace_hooks.py`` calls it for you);
git config and ``info/attributes`` are both shared across a clone's worktrees,
so once really is enough. Idempotent: re-running rewrites the same config and
does not duplicate the attribute line.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DRIVER_NAME = "spec-rows"
ATTRIBUTE_LINE = f"SPEC.md merge={DRIVER_NAME}"
ATTRIBUTE_BLOCK = f"""# Union SPEC.md change-log rows instead of conflicting on
# adjacent inserts (Repository_Management#1520).
# Deliberately NOT committed in .gitattributes:
# git aborts a merge outright when a named driver is unregistered, so a clone
# without the driver could not merge SPEC.md at all. Installed by
# scripts/install_spec_merge_driver.py together with the driver definition.
{ATTRIBUTE_LINE}
"""
ROOT = Path(__file__).resolve().parent.parent


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


#: Path to the driver, RELATIVE to the top of the worktree.
#:
#: Git runs a merge driver with its working directory at the top of the
#: worktree being merged, so a relative path resolves to that worktree's own
#: copy of the script. An absolute path would pin the config -- which is shared
#: by every worktree of the clone -- to whichever worktree happened to run the
#: installer; when that worktree is later removed, every SPEC.md merge in the
#: clone aborts with `fatal: custom merge driver spec-rows lacks command line`.
#: (Observed during the AffineDrift rollout.)
DRIVER_SCRIPT = "scripts/spec_rows_merge_driver.py"


def driver_command(repo_root: Path) -> str:
    """Return the ``merge.spec-rows.driver`` command for ``repo_root``."""
    del repo_root  # the command is deliberately worktree-relative
    interpreter = Path(sys.executable).as_posix()
    return f'"{interpreter}" "{DRIVER_SCRIPT}" %O %A %B %P'


def attributes_path(repo_root: Path) -> Path | None:
    """Return ``$GIT_COMMON_DIR/info/attributes`` for ``repo_root``."""
    result = _git(["rev-parse", "--git-common-dir"], repo_root)
    if result.returncode != 0:
        return None
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = (repo_root / common).resolve()
    return common / "info" / "attributes"


def install_attribute(repo_root: Path, *, dry_run: bool = False) -> str:
    """Add the per-clone attribute, without duplicating it."""
    path = attributes_path(repo_root)
    if path is None:
        return "SKIPPED (not a git repository)"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if ATTRIBUTE_LINE in existing:
        return f"already present in {path}"
    updated = existing
    if updated and not updated.endswith("\n"):
        updated += "\n"
    if updated:
        updated += "\n"
    updated += ATTRIBUTE_BLOCK
    if dry_run:
        return f"would append to {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8", newline="\n")
    return f"appended to {path}"


def install(repo_root: Path, *, dry_run: bool = False) -> int:
    command = driver_command(repo_root)
    settings = [
        (f"merge.{DRIVER_NAME}.name", "union SPEC.md change-log rows (RM#1520)"),
        (f"merge.{DRIVER_NAME}.driver", command),
    ]
    for key, value in settings:
        if dry_run:
            print(f"would set {key}={value}")
            continue
        result = _git(["config", "--local", key, value], repo_root)
        if result.returncode != 0:
            print(f"ERROR: git config {key} failed: {result.stderr.strip()}")
            return 1
        print(f"set {key}")
    print(f"attribute: {install_attribute(repo_root, dry_run=dry_run)}")
    if not dry_run:
        check = _git(["check-attr", "merge", "--", "SPEC.md"], repo_root)
        print(check.stdout.strip() or check.stderr.strip())
        if f"merge: {DRIVER_NAME}" not in check.stdout:
            print(
                "ERROR: SPEC.md is not routed to the driver; the merge would "
                "use git's default behaviour."
            )
            return 1
        print("spec-rows merge driver installed and SPEC.md routed to it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return install(Path(args.repo_root).resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
