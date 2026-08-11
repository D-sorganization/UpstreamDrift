"""Immutable Tools gitlink authority for production launcher providers."""

from __future__ import annotations

import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TOOLS_GITLINK_PATH = Path("vendor/ud-tools")
TOOLS_GITLINK_SHA = "ff4240217005e1415ca409fd124e50b64ee642d2"
_GIT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ToolsVendorAuthority:
    """Fail-closed result of validating the pinned Tools gitlink checkout."""

    root: Path
    expected_sha: str
    available: bool
    reason: str | None = None


class ProviderUnavailableError(ValueError):
    """Raised when a declared provider cannot satisfy its authority contract."""


def _run_git_command(args: tuple[str, ...], *, cwd: Path) -> tuple[int, str]:
    """Run one fixed-purpose local Git query with bounded output capture."""
    try:
        completed = subprocess.run(  # nosec B603
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return completed.returncode, completed.stdout.rstrip("\r\n")


def _is_reparse_point(path: Path) -> bool:
    """Return whether a path is a symlink or Windows reparse point."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(file_attributes & reparse_flag)


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a canonical path is equal to or beneath a root."""
    return path == root or root in path.parents


def inspect_tools_vendor_authority(repo_root: Path) -> ToolsVendorAuthority:
    """Validate the exact clean Tools gitlink checkout used in production.

    Postcondition:
        ``available`` is true only for a normal directory backed by the exact
        superproject gitlink and checked out at the declared SHA with no dirty
        or untracked state.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")

    canonical_repo = repo_root.resolve(strict=False)
    vendor_root = canonical_repo / _TOOLS_GITLINK_PATH

    def unavailable(reason: str) -> ToolsVendorAuthority:
        return ToolsVendorAuthority(
            root=vendor_root,
            expected_sha=TOOLS_GITLINK_SHA,
            available=False,
            reason=reason,
        )

    if not canonical_repo.is_dir():
        return unavailable("UpstreamDrift repository root is unavailable")

    index_code, index_output = _run_git_command(
        ("ls-files", "--stage", "--", _TOOLS_GITLINK_PATH.as_posix()),
        cwd=canonical_repo,
    )
    if index_code != 0 or len(index_output.splitlines()) != 1:
        return unavailable("tracked Tools gitlink entry is missing")
    index_fields = index_output.split(maxsplit=3)
    if len(index_fields) != 4:
        return unavailable("tracked Tools gitlink entry is missing")
    mode, tracked_sha, stage, tracked_path = index_fields
    if (
        mode != "160000"
        or stage != "0"
        or tracked_path != _TOOLS_GITLINK_PATH.as_posix()
    ):
        return unavailable("tracked Tools entry is not the declared gitlink")
    if tracked_sha != TOOLS_GITLINK_SHA:
        return unavailable("tracked Tools gitlink SHA does not match the declared pin")

    if not vendor_root.is_dir() or _is_reparse_point(vendor_root):
        return unavailable("Tools gitlink checkout is missing or replaced")
    try:
        canonical_vendor = vendor_root.resolve(strict=True)
    except OSError:
        return unavailable("Tools gitlink checkout cannot be resolved")
    if canonical_vendor != vendor_root:
        return unavailable("Tools gitlink checkout escapes its canonical path")

    git_marker = canonical_vendor / ".git"
    if not git_marker.is_file() or _is_reparse_point(git_marker):
        return unavailable("Tools checkout is not an initialized gitlink worktree")

    src_root = canonical_vendor / "src"
    try:
        canonical_src = src_root.resolve(strict=True)
    except OSError:
        return unavailable("Tools gitlink src directory is missing")
    if not canonical_src.is_dir() or not _is_within(canonical_src, canonical_vendor):
        return unavailable("Tools gitlink src directory escapes vendor authority")

    submodule_code, submodule_output = _run_git_command(
        ("submodule", "status", "--", _TOOLS_GITLINK_PATH.as_posix()),
        cwd=canonical_repo,
    )
    expected_status = f" {TOOLS_GITLINK_SHA} {_TOOLS_GITLINK_PATH.as_posix()}"
    status_matches = submodule_output == expected_status or submodule_output.startswith(
        f"{expected_status} "
    )
    if submodule_code != 0 or not status_matches:
        return unavailable("Tools checkout is not synchronized to the tracked gitlink")

    head_code, checked_out_sha = _run_git_command(
        ("rev-parse", "--verify", "HEAD"), cwd=canonical_vendor
    )
    if head_code != 0 or checked_out_sha != TOOLS_GITLINK_SHA:
        return unavailable("Tools checkout HEAD does not match the declared pin")

    top_code, top_level = _run_git_command(
        ("rev-parse", "--show-toplevel"), cwd=canonical_vendor
    )
    if top_code != 0 or Path(top_level).resolve(strict=False) != canonical_vendor:
        return unavailable("Tools checkout is not rooted at the gitlink path")

    parent_code, superproject = _run_git_command(
        ("rev-parse", "--show-superproject-working-tree"), cwd=canonical_vendor
    )
    if (
        parent_code != 0
        or not superproject
        or Path(superproject).resolve(strict=False) != canonical_repo
    ):
        return unavailable("Tools checkout is not attached to this superproject")

    dirty_code, dirty_output = _run_git_command(
        ("status", "--porcelain=v1", "--untracked-files=all"), cwd=canonical_vendor
    )
    if dirty_code != 0 or dirty_output:
        return unavailable("Tools gitlink checkout is dirty")

    parent_dirty_code, parent_dirty_output = _run_git_command(
        (
            "status",
            "--porcelain=v1",
            "--ignore-submodules=none",
            "--",
            _TOOLS_GITLINK_PATH.as_posix(),
        ),
        cwd=canonical_repo,
    )
    if parent_dirty_code != 0 or parent_dirty_output:
        return unavailable("superproject reports a modified Tools gitlink")

    return ToolsVendorAuthority(
        root=canonical_vendor,
        expected_sha=TOOLS_GITLINK_SHA,
        available=True,
    )


__all__ = [
    "ProviderUnavailableError",
    "TOOLS_GITLINK_SHA",
    "ToolsVendorAuthority",
    "inspect_tools_vendor_authority",
]
