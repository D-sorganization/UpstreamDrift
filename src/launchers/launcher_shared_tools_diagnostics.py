"""Shared-Tools freshness probing for launcher diagnostics.

This module owns the repository-state comparison while
``LauncherDiagnostics`` retains result recording and its public method surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class GitCommandRunner(Protocol):
    """Callable boundary used to query local Git state."""

    def __call__(self, cmd: list[str], cwd: Path | None = None) -> str: ...


class SiblingRootFinder(Protocol):
    """Callable boundary used to locate a sibling Tools checkout."""

    def __call__(self) -> Path | None: ...


@dataclass(frozen=True)
class SharedToolsFreshness:
    """Value returned by the shared-Tools freshness probe."""

    status: str
    message: str
    details: dict[str, Any]

    def __post_init__(self) -> None:
        """Enforce the diagnostic result contract at the extraction boundary."""
        if self.status not in {"pass", "warning"}:
            msg = f"Unsupported shared-Tools diagnostic status: {self.status!r}"
            raise ValueError(msg)


def _read_pinned_sha(run_git_cmd: GitCommandRunner) -> str:
    """Return the locally recorded ``vendor/ud-tools`` Gitlink SHA."""
    output = run_git_cmd(["git", "ls-files", "--stage", "vendor/ud-tools"])
    for line in output.splitlines():
        if "vendor/ud-tools" not in line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            return parts[1]
    return ""


def _read_checkout_sha(
    repo_root: Path, run_git_cmd: GitCommandRunner
) -> tuple[Path, str]:
    """Return the expected submodule path and its checked-out SHA, if any."""
    submodule_dir = repo_root / "vendor" / "ud-tools"
    if not submodule_dir.is_dir():
        return submodule_dir, ""
    return submodule_dir, run_git_cmd(["git", "rev-parse", "HEAD"], cwd=submodule_dir)


def inspect_shared_tools_freshness(
    *,
    repo_root: Path,
    run_git_cmd: GitCommandRunner,
    find_sibling_root: SiblingRootFinder,
) -> SharedToolsFreshness:
    """Compare the vendored Tools checkout, its pin, sibling, and remote."""
    details: dict[str, Any] = {
        "submodule_path": "vendor/ud-tools",
        "sibling_path": None,
        "pinned_sha": None,
        "checked_out_sha": None,
        "sibling_sha": None,
        "submodule_status": "unknown",
        "sibling_status": "unknown",
        "is_current": True,
    }

    pinned_sha = _read_pinned_sha(run_git_cmd)
    if pinned_sha:
        details["pinned_sha"] = pinned_sha

    submodule_dir, checked_out_sha = _read_checkout_sha(repo_root, run_git_cmd)
    if checked_out_sha:
        details["checked_out_sha"] = checked_out_sha

    sibling_root = find_sibling_root()
    sibling_sha = ""
    if sibling_root:
        details["sibling_path"] = str(sibling_root)
        sibling_sha = run_git_cmd(["git", "rev-parse", "HEAD"], cwd=sibling_root)
        if sibling_sha:
            details["sibling_sha"] = sibling_sha

    status = "pass"
    messages: list[str] = []
    if not checked_out_sha:
        status = "warning"
        details["submodule_status"] = "not_initialized"
        details["is_current"] = False
        messages.append("vendor/ud-tools submodule is not initialized or checked out.")
    elif pinned_sha and checked_out_sha != pinned_sha:
        status = "warning"
        details["submodule_status"] = "out_of_sync_with_pin"
        details["is_current"] = False
        messages.append(
            f"vendor/ud-tools checked-out commit ({checked_out_sha[:8]}) "
            f"differs from pinned commit ({pinned_sha[:8]})."
        )
    else:
        details["submodule_status"] = "synchronized"

    if sibling_sha:
        compare_target = pinned_sha or checked_out_sha
        if compare_target and sibling_sha != compare_target:
            status = "warning"
            details["sibling_status"] = "out_of_sync_with_submodule"
            details["is_current"] = False
            messages.append(
                f"Sibling Tools repository commit ({sibling_sha[:8]}) "
                "differs from UpstreamDrift's expected/pinned version "
                f"({compare_target[:8]})."
            )
        else:
            details["sibling_status"] = "synchronized"
    elif details["sibling_path"]:
        details["sibling_status"] = "unreadable"
    else:
        details["sibling_status"] = "not_found"

    if checked_out_sha and submodule_dir.is_dir():
        tracking_branch = (
            run_git_cmd(["git", "config", "submodule.vendor/ud-tools.branch"]) or "main"
        )
        remote_sha = run_git_cmd(
            ["git", "rev-parse", f"origin/{tracking_branch}"], cwd=submodule_dir
        )
        if remote_sha and remote_sha != checked_out_sha:
            merge_base = run_git_cmd(
                ["git", "merge-base", "HEAD", f"origin/{tracking_branch}"],
                cwd=submodule_dir,
            )
            if merge_base == checked_out_sha:
                status = "warning"
                details["is_current"] = False
                details["remote_sha"] = remote_sha
                messages.append(
                    "vendor/ud-tools submodule is behind remote "
                    f"origin/{tracking_branch} ({remote_sha[:8]})."
                )

    message = (
        " ".join(messages)
        if messages
        else "Shared folders and submodules are fully up to date."
    )
    return SharedToolsFreshness(status=status, message=message, details=details)
