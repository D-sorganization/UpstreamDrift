"""Forbid uncommitted modifications inside the ``vendor/ud-tools`` submodule.

See UpstreamDrift issue #5623. The vendor submodule is a downstream copy of
Tools. Edits to its working tree are erased on the next
``git submodule update`` or vendor bump, silently destroying work.

This test runs ``git status --porcelain=v2 vendor/ud-tools`` from the repo
root and asserts that no working-tree modifications exist *inside* the
submodule. The submodule pointer itself moving (``M vendor/ud-tools`` at
the parent level) is fine — that records a deliberate vendor bump.

Design-by-contract:

- The ``git`` invocation must not raise. If ``git`` is unavailable (e.g.
  on a stripped-down CI runner) the test :func:`pytest.skip` s with a
  clear reason rather than reporting a spurious failure.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - git invocation, no shell, fixed args
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_PREFIX = "vendor/ud-tools/"


@pytest.mark.unit
def test_vendor_submodule_has_no_uncommitted_edits() -> None:
    """Working tree inside ``vendor/ud-tools`` must be clean.

    A dirty working tree inside the submodule means somebody edited
    vendored Tools code in place. Those edits will be lost on the next
    ``git submodule update`` or vendor bump — see issue #5623.
    """
    git_bin = shutil.which("git")
    if git_bin is None:
        pytest.skip("git executable not available on PATH")

    try:
        result = subprocess.run(  # nosec B603 - fixed args, no shell
            [git_bin, "status", "--porcelain=v2", "vendor/ud-tools"],
            cwd=str(_REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git status invocation failed: {exc}")

    if result.returncode != 0:
        pytest.skip(
            "git status returned non-zero — likely not a git checkout "
            f"(rc={result.returncode}, stderr={result.stderr.strip()!r})"
        )

    # porcelain=v2 format: each tracked change starts with "1 " (ordinary
    # change), "2 " (rename/copy), "u " (unmerged), or "? " (untracked).
    # Lines like "# branch.oid <sha>" start with "#" — ignore those.
    #
    # The submodule pointer change at the parent level appears as a single
    # "1 .M ..." line whose final path field is exactly "vendor/ud-tools".
    # We only flag entries that reference a path *inside* the submodule —
    # i.e. paths starting with "vendor/ud-tools/".
    offenders: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        if not line.startswith(("1 ", "2 ", "u ")):
            continue
        # The path is the last whitespace-separated field for "1 ", and
        # appears with a tab separator for "2 ". Checking simple substring
        # containment of the prefix is sufficient and robust to either form.
        if _VENDOR_PREFIX in line:
            offenders.append(line)

    assert not offenders, (
        "vendor/ud-tools submodule has uncommitted modifications in its "
        "working tree. Tools is the source of truth for these files — "
        "changes here are erased on the next `git submodule update` or "
        "vendor bump. Make the edit in the Tools repo, land it there, then "
        "bump the submodule pointer. See UpstreamDrift issue #5623.\n\n"
        "Offending entries:\n  " + "\n  ".join(offenders)
    )
