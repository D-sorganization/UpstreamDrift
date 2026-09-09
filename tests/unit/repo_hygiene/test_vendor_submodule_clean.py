"""Forbid uncommitted modifications inside the ``vendor/ud-tools`` submodule.

See UpstreamDrift issue #5623. The vendor submodule is a downstream copy of
Tools. Edits to its working tree are erased on the next ``git submodule
update`` or vendor bump, silently destroying work.

How this is detected
--------------------
Git **never** reports individual files inside a submodule from the
superproject. All submodule-internal state collapses into the single gitlink
entry, whose porcelain-v2 ``<sub>`` field encodes it as ``S<c><m><u>``:

===== ==========================================================
 ``c``  the submodule's HEAD differs from the recorded gitlink
 ``m``  the submodule has modified *tracked* files
 ``u``  the submodule has *untracked* files
===== ==========================================================

Only ``m`` and ``u`` are working-tree dirt. A bare ``c`` (``SC..``) is a
deliberate vendor pointer bump and is allowed.

This matters: the previous "hardened" implementation looked for status lines
whose path started with ``vendor/ud-tools/`` (trailing slash). Git emits the
path as exactly ``vendor/ud-tools``, so that predicate never matched and the
guard could not fire even when it ran. See :func:`_dirty_submodule_entries`.

Design-by-contract:

- ``--ignore-submodules=none`` must be passed so that git's default of
  ``--ignore-submodules=all`` cannot silently hide submodule dirt (UD #5626).
- A missing vendor submodule **fails closed in CI** and degrades to a skip only
  on a developer machine, mirroring ``test_tools_child_copy_contract.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - git invocation, no shell, fixed args
from pathlib import Path

import pytest

from tests.helpers.seam_guards import require_vendor_path

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_PATH = "vendor/ud-tools"
_VENDOR_DIR = _REPO_ROOT / "vendor" / "ud-tools"
_IGNORE_SUBMODULES_FLAG = "--ignore-submodules=none"


def _require_vendor_dir() -> Path:
    """Return the vendor submodule directory, or fail closed."""
    return require_vendor_path(_VENDOR_DIR)


def _dirty_submodule_entries(status_output: str) -> list[str]:
    """Return porcelain-v2 lines showing working-tree dirt inside a submodule.

    Parses the ``<sub>`` field rather than the path, because git collapses all
    submodule-internal changes onto the gitlink entry itself. A pointer bump
    (``SC..``) is deliberate and is *not* reported.
    """
    offenders: list[str] = []
    for raw_line in status_output.splitlines():
        line = raw_line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        # Ordinary ("1 ") and rename/copy ("2 ") entries share the leading
        # layout: <kind> <XY> <sub> ...
        if not line.startswith(("1 ", "2 ")):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        sub = fields[2]
        if not sub.startswith("S") or len(sub) < 4:
            continue
        has_modified_tracked = sub[2] == "M"
        has_untracked = sub[3] == "U"
        if has_modified_tracked or has_untracked:
            offenders.append(line)
    return offenders


def _git_status(*paths: str) -> subprocess.CompletedProcess[str] | None:
    git_bin = shutil.which("git")
    if git_bin is None:
        return None
    return subprocess.run(  # nosec B603 - fixed args, no shell
        [
            git_bin,
            "status",
            "--porcelain=v2",
            _IGNORE_SUBMODULES_FLAG,
            *paths,
        ],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _vendor_internal_detail() -> str:
    """Best-effort per-file listing from inside the submodule for the message."""
    git_bin = shutil.which("git")
    if git_bin is None:
        return ""
    result = subprocess.run(  # nosec B603 - fixed args, no shell
        [git_bin, "status", "--porcelain"],
        cwd=str(_VENDOR_DIR),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


# ── regression cover for the detection logic ─────────────────────────────────


def test_modified_tracked_files_inside_submodule_are_detected() -> None:
    """``S.M.`` is dirt. Regression: the old path-prefix predicate missed this.

    This is the exact line git emits for an edited tracked file inside
    ``vendor/ud-tools``. Note the path has no trailing slash, which is why the
    previous ``"vendor/ud-tools/" in line`` check could never match.
    """
    line = (
        "1 .M S.M. 160000 160000 160000 "
        "4744422d39aea03f5b6f59c8908f5e79ce246d92 "
        "4744422d39aea03f5b6f59c8908f5e79ce246d92 vendor/ud-tools"
    )
    assert _dirty_submodule_entries(line) == [line]
    assert f"{_VENDOR_PATH}/" not in line


def test_untracked_files_inside_submodule_are_detected() -> None:
    """``S..U`` is dirt too — a stray new file inside the vendor tree."""
    line = (
        "1 .M S..U 160000 160000 160000 "
        "4744422d39aea03f5b6f59c8908f5e79ce246d92 "
        "4744422d39aea03f5b6f59c8908f5e79ce246d92 vendor/ud-tools"
    )
    assert _dirty_submodule_entries(line) == [line]


def test_pointer_bump_alone_is_allowed() -> None:
    """``SC..`` is a deliberate vendor bump, not working-tree dirt."""
    line = (
        "1 .M SC.. 160000 160000 160000 "
        "1111111111111111111111111111111111111111 "
        "2222222222222222222222222222222222222222 vendor/ud-tools"
    )
    assert _dirty_submodule_entries(line) == []


def test_non_submodule_changes_are_ignored() -> None:
    """Ordinary tracked-file edits elsewhere in the repo are out of scope."""
    line = (
        "1 .M N... 100644 100644 100644 "
        "1111111111111111111111111111111111111111 "
        "1111111111111111111111111111111111111111 src/shared/python/config.py"
    )
    assert _dirty_submodule_entries(line) == []


def test_status_invocation_disables_submodule_ignoring() -> None:
    """Regression UD#5626: the flag must be present on the real invocation.

    Without ``--ignore-submodules=none`` git defaults to ``all``, which hides
    every change inside a submodule and makes this guard pass unconditionally.
    """
    assert _IGNORE_SUBMODULES_FLAG == "--ignore-submodules=none"
    result = _git_status(_VENDOR_PATH)
    if result is None:
        pytest.skip("git executable not available on PATH")
    # A clean tree yields no output; the point is that the call succeeds with
    # the flag accepted by this git version.
    assert result.returncode == 0, (
        f"git rejected {_IGNORE_SUBMODULES_FLAG}: {result.stderr.strip()!r}"
    )


# ── the guard itself ─────────────────────────────────────────────────────────


def test_vendor_submodule_has_no_uncommitted_edits() -> None:
    """Working tree inside ``vendor/ud-tools`` must be clean.

    A dirty working tree inside the submodule means somebody edited vendored
    Tools code in place. Those edits are lost on the next ``git submodule
    update`` or vendor bump — see issue #5623.
    """
    _require_vendor_dir()

    result = _git_status(_VENDOR_PATH)
    if result is None:
        if os.environ.get("CI"):
            raise AssertionError(
                "git is required in CI for the vendor-clean guard, but no git "
                "executable was found on PATH."
            )
        pytest.skip("git executable not available on PATH")

    if result.returncode != 0:
        if os.environ.get("CI"):
            raise AssertionError(
                "git status failed in CI for the vendor-clean guard "
                f"(rc={result.returncode}, stderr={result.stderr.strip()!r})."
            )
        pytest.skip(
            "git status returned non-zero — likely not a git checkout "
            f"(rc={result.returncode}, stderr={result.stderr.strip()!r})"
        )

    offenders = _dirty_submodule_entries(result.stdout)
    detail = _vendor_internal_detail() if offenders else ""

    assert not offenders, (
        "vendor/ud-tools has uncommitted modifications in its working tree. "
        "Tools is the source of truth for these files — changes here are "
        "erased on the next `git submodule update` or vendor bump. Make the "
        "edit in the Tools repo, land it there, then bump the submodule "
        "pointer. See UpstreamDrift issue #5623.\n\n"
        "Superproject entries:\n  " + "\n  ".join(offenders) + "\n\n"
        "Files inside the submodule:\n  " + (detail or "(unavailable)")
    )
