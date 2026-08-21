"""Immutable Tools gitlink authority for production launcher providers.

The single source of truth for the Tools pin is the ``vendor/ud-tools``
gitlink recorded in the superproject index (``git ls-files --stage``).
There is deliberately no hand-maintained SHA constant: a governed
submodule bump is a one-line gitlink change that this module and its
tests validate automatically (issue #8852).
"""

from __future__ import annotations

import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TOOLS_GITLINK_PATH = Path("vendor/ud-tools")
_GIT_TIMEOUT_SECONDS = 5.0
_GITLINK_MODE = "160000"


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


def _tracked_gitlink_sha(canonical_repo: Path) -> tuple[str | None, str | None]:
    """Read the pinned SHA from the one declared superproject gitlink.

    Precondition:
        ``canonical_repo`` is the resolved superproject root.

    Postcondition:
        Returns ``(sha, None)`` where ``sha`` is the 40-hex gitlink object
        recorded in the index, or ``(None, reason)`` when the index does not
        contain exactly the declared gitlink entry.
    """
    index_code, index_output = _run_git_command(
        ("ls-files", "--stage", "--", _TOOLS_GITLINK_PATH.as_posix()),
        cwd=canonical_repo,
    )
    if index_code != 0 or len(index_output.splitlines()) != 1:
        return None, "tracked Tools gitlink entry is missing"
    index_fields = index_output.split(maxsplit=3)
    if len(index_fields) != 4:
        return None, "tracked Tools gitlink entry is missing"
    mode, tracked_sha, stage, tracked_path = index_fields
    if (
        mode != _GITLINK_MODE
        or stage != "0"
        or tracked_path != _TOOLS_GITLINK_PATH.as_posix()
    ):
        return None, "tracked Tools entry is not the declared gitlink"
    if len(tracked_sha) != 40 or any(c not in "0123456789abcdef" for c in tracked_sha):
        return None, "tracked Tools gitlink SHA is malformed"
    return tracked_sha, None


def expected_tools_gitlink_sha(repo_root: Path) -> str | None:
    """Return the pinned Tools SHA derived from the tracked gitlink.

    This is the DRY authority for the pin: there is no duplicated constant
    to go stale. Returns ``None`` when the gitlink cannot be read.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")
    sha, _error = _tracked_gitlink_sha(repo_root.resolve(strict=False))
    return sha


def _resolve_checkout(vendor_root: Path) -> tuple[Path | None, str | None]:
    """Resolve and validate the initialized, non-reparse checkout layout."""

    if not vendor_root.is_dir() or _is_reparse_point(vendor_root):
        return None, "Tools gitlink checkout is missing or replaced"
    try:
        canonical_vendor = vendor_root.resolve(strict=True)
    except OSError:
        return None, "Tools gitlink checkout cannot be resolved"
    if canonical_vendor != vendor_root:
        return None, "Tools gitlink checkout escapes its canonical path"

    git_marker = canonical_vendor / ".git"
    if not git_marker.is_file() or _is_reparse_point(git_marker):
        return None, "Tools checkout is not an initialized gitlink worktree"

    src_root = canonical_vendor / "src"
    try:
        canonical_src = src_root.resolve(strict=True)
    except OSError:
        return None, "Tools gitlink src directory is missing"
    if not canonical_src.is_dir() or not _is_within(canonical_src, canonical_vendor):
        return None, "Tools gitlink src directory escapes vendor authority"
    return canonical_vendor, None


def _checkout_identity_error(
    canonical_repo: Path, canonical_vendor: Path, expected_sha: str
) -> str | None:
    """Return an error unless checkout identity matches the superproject pin."""

    submodule_code, submodule_output = _run_git_command(
        ("submodule", "status", "--", _TOOLS_GITLINK_PATH.as_posix()),
        cwd=canonical_repo,
    )
    expected_status = f" {expected_sha} {_TOOLS_GITLINK_PATH.as_posix()}"
    status_matches = submodule_output == expected_status or submodule_output.startswith(
        f"{expected_status} "
    )
    if submodule_code != 0 or not status_matches:
        return "Tools checkout is not synchronized to the tracked gitlink"

    head_code, checked_out_sha = _run_git_command(
        ("rev-parse", "--verify", "HEAD"), cwd=canonical_vendor
    )
    if head_code != 0 or checked_out_sha != expected_sha:
        found = checked_out_sha or "unknown"
        return f"Tools pin stale (expected {expected_sha}, found {found})"

    top_code, top_level = _run_git_command(
        ("rev-parse", "--show-toplevel"), cwd=canonical_vendor
    )
    if top_code != 0 or Path(top_level).resolve(strict=False) != canonical_vendor:
        return "Tools checkout is not rooted at the gitlink path"

    parent_code, superproject = _run_git_command(
        ("rev-parse", "--show-superproject-working-tree"), cwd=canonical_vendor
    )
    if (
        parent_code != 0
        or not superproject
        or Path(superproject).resolve(strict=False) != canonical_repo
    ):
        return "Tools checkout is not attached to this superproject"
    return None


def _checkout_cleanliness_error(
    canonical_repo: Path, canonical_vendor: Path
) -> str | None:
    """Return an error when either Git view reports mutable Tools state."""

    dirty_code, dirty_output = _run_git_command(
        ("status", "--porcelain=v1", "--untracked-files=all"), cwd=canonical_vendor
    )
    if dirty_code != 0 or dirty_output:
        return "Tools gitlink checkout is dirty"

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
        return "superproject reports a modified Tools gitlink"
    return None


def _unavailable(
    vendor_root: Path, reason: str, expected_sha: str = ""
) -> ToolsVendorAuthority:
    """Build one normalized fail-closed authority result."""
    return ToolsVendorAuthority(
        root=vendor_root,
        expected_sha=expected_sha,
        available=False,
        reason=reason,
    )


def inspect_tools_vendor_authority(repo_root: Path) -> ToolsVendorAuthority:
    """Validate the exact clean Tools gitlink checkout used in production.

    Precondition:
        ``repo_root`` is a ``pathlib.Path`` (the superproject checkout root).

    Postconditions:
        ``available`` is true only for a normal directory backed by the exact
        superproject gitlink and checked out at the tracked gitlink SHA with
        no dirty or untracked state. When ``available`` is false, ``reason``
        is a non-empty human-readable explanation; a checkout whose HEAD
        drifted from the gitlink reports
        ``"Tools pin stale (expected X, found Y)"`` explicitly rather than
        failing silently.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")

    canonical_repo = repo_root.resolve(strict=False)
    vendor_root = canonical_repo / _TOOLS_GITLINK_PATH
    if not canonical_repo.is_dir():
        return _unavailable(vendor_root, "UpstreamDrift repository root is unavailable")

    expected_sha, error = _tracked_gitlink_sha(canonical_repo)
    if error is not None or expected_sha is None:
        return _unavailable(
            vendor_root, error or "tracked Tools gitlink entry is missing"
        )
    canonical_vendor, error = _resolve_checkout(vendor_root)
    if error is not None or canonical_vendor is None:
        return _unavailable(
            vendor_root,
            error or "Tools checkout cannot be resolved",
            expected_sha,
        )
    error = _checkout_identity_error(canonical_repo, canonical_vendor, expected_sha)
    if error is not None:
        return _unavailable(vendor_root, error, expected_sha)
    error = _checkout_cleanliness_error(canonical_repo, canonical_vendor)
    if error is not None:
        return _unavailable(vendor_root, error, expected_sha)

    return ToolsVendorAuthority(
        root=canonical_vendor,
        expected_sha=expected_sha,
        available=True,
    )


__all__ = [
    "ProviderUnavailableError",
    "ToolsVendorAuthority",
    "expected_tools_gitlink_sha",
    "inspect_tools_vendor_authority",
]
