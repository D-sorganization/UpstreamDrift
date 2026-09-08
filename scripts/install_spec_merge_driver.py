#!/usr/bin/env python3
"""Register the ``spec-rows`` merge driver in this clone.

Both halves are written together and neither is committed. The *definition* of
a merge driver lives in git config, which is per-clone; the attribute
``SPEC.md merge=spec-rows`` goes in ``$GIT_COMMON_DIR/info/attributes`` rather
than a committed ``.gitattributes``. Run once per clone; git config and
``info/attributes`` are both shared across worktrees, so once is enough.

**Wire this into your repository's existing hook-setup entry point** rather
than relying on people to run it by hand -- in Repository_Management that is
``scripts/install_workspace_hooks.py``; in Tools it is
``scripts/setup_hooks.py``, which already registers that repo's other merge
driver the same way. A vendored copy of this file that nothing invokes leaves
the driver inert, and this docstring is exactly where someone checks whether it
is automatic, so do not let it name an entry point the repository does not
have.

Idempotent: re-running rewrites the same two config values.

What git actually does, measured rather than assumed
----------------------------------------------------

An earlier version of this docstring claimed git *aborts* a merge when an
attribute names an unregistered driver, and that a committed ``.gitattributes``
would therefore make SPEC.md unmergeable in any clone without the driver. That
was wrong, and it is corrected here because it was the stated reason for the
whole per-clone design. The three states behave differently:

===============================================  ==========================
clone state (attribute present in every case)    ``git merge`` result
===============================================  ==========================
no ``merge.spec-rows.*`` config at all           exit 1, ordinary ``UU``
  (a fresh clone, a CI checkout)                 conflict -- graceful
``.name`` set, ``.driver`` missing               **exit 128**, ``fatal:
  (half-configured)                              custom merge driver
                                                 spec-rows lacks command
                                                 line`` -- merge aborts
``.driver`` set, script absent from the          exit 1, ordinary ``UU``
  worktree being merged (e.g. a checkout         conflict -- graceful
  predating the driver's commit)
===============================================  ==========================

So an unconfigured clone degrades gracefully, and committing the attribute
would have been survivable. The attribute still stays out of ``.gitattributes``
-- keeping the two halves in one place, written and removed together, is what
prevents the half-configured state -- but that is a tidiness argument, not the
catastrophe previously described.

**Removing this: unset BOTH keys.** Unsetting only ``merge.spec-rows.driver``
and leaving ``merge.spec-rows.name`` behind produces the one state above that
aborts merges. To disarm cleanly::

    git config --unset merge.spec-rows.driver
    git config --unset merge.spec-rows.name
    # then delete the `SPEC.md merge=spec-rows` line from
    # "$(git rev-parse --git-common-dir)/info/attributes"

See Repository_Management#1520.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DRIVER_NAME = "spec-rows"
#: Path to the driver, RELATIVE to the top of the worktree.
#:
#: Git runs a merge driver with its working directory at the top of the
#: worktree being merged, so a relative path resolves to that worktree's own
#: copy of the script. An absolute path would pin the config -- which is shared
#: by every worktree of the clone -- to whichever worktree happened to run the
#: installer, and once that worktree is removed the script is gone. Per the
#: table above that is the graceful case (exit 1, an ordinary conflict), not a
#: fatal one, but it silently disables the driver and emits a confusing
#: interpreter error, so the relative path is still correct.
#: (Observed during the AffineDrift rollout, then measured.)
DRIVER_SCRIPT = "scripts/spec_rows_merge_driver.py"
ATTRIBUTE_LINE = f"SPEC.md merge={DRIVER_NAME}"
ATTRIBUTE_BLOCK = f"""# Union SPEC.md change-log rows instead of conflicting on
# adjacent inserts (Repository_Management#1520).
# Kept here rather than in a committed .gitattributes so that this line and the
# merge.spec-rows.* config are written -- and removed -- together. A clone with
# neither degrades gracefully (an ordinary conflict); a clone with only
# merge.spec-rows.name set and no .driver aborts every SPEC.md merge with
# "fatal: custom merge driver spec-rows lacks command line", so when removing
# this, unset BOTH config keys as well as this line. Installed by
# scripts/install_spec_merge_driver.py.
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


def driver_command(repo_root: Path) -> str:
    """Return the ``merge.spec-rows.driver`` command.

    The script path is **worktree-relative on purpose**. Git config is shared by
    every worktree of a clone, so an absolute path would pin the driver to
    whichever worktree happened to run the installer; once that worktree is
    removed the script is gone and the driver silently stops working. Per the
    measured table in the module docstring that is the *graceful* failure (exit
    1, an ordinary conflict) rather than the fatal one -- but it disables the
    driver while leaving it configured, and emits a confusing interpreter error
    that nobody will connect to this campaign. Git runs a merge driver with its
    working directory at the top of the worktree being merged, so a relative
    path resolves to that worktree's own copy and is correct for all of them.
    """
    del repo_root  # deliberately unused: the command must not be worktree-specific
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
