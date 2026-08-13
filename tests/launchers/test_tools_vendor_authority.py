"""Tests for the immutable Tools gitlink authority boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.shared.python.config import tools_vendor_authority as authority

GitResponseKey = tuple[Path, tuple[str, ...]]


def _valid_authority_responses(
    repo_root: Path,
) -> dict[GitResponseKey, tuple[int, str]]:
    vendor_root = repo_root / "vendor" / "ud-tools"
    pin = authority.TOOLS_GITLINK_SHA
    return {
        (
            repo_root,
            ("ls-files", "--stage", "--", "vendor/ud-tools"),
        ): (0, f"160000 {pin} 0\tvendor/ud-tools"),
        (
            repo_root,
            ("submodule", "status", "--", "vendor/ud-tools"),
        ): (0, f" {pin} vendor/ud-tools (heads/main)"),
        (vendor_root, ("rev-parse", "--verify", "HEAD")): (0, pin),
        (vendor_root, ("rev-parse", "--show-toplevel")): (0, str(vendor_root)),
        (
            vendor_root,
            ("rev-parse", "--show-superproject-working-tree"),
        ): (0, str(repo_root)),
        (
            vendor_root,
            ("status", "--porcelain=v1", "--untracked-files=all"),
        ): (0, ""),
        (
            repo_root,
            (
                "status",
                "--porcelain=v1",
                "--ignore-submodules=none",
                "--",
                "vendor/ud-tools",
            ),
        ): (0, ""),
    }


@pytest.fixture
def authority_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[GitResponseKey, tuple[int, str]]]:
    repo_root = tmp_path / "UpstreamDrift"
    vendor_root = repo_root / "vendor" / "ud-tools"
    (vendor_root / "src").mkdir(parents=True)
    (vendor_root / ".git").write_text(
        "gitdir: ../../../.git/modules/vendor/ud-tools\n",
        encoding="utf-8",
    )
    responses = _valid_authority_responses(repo_root)

    def fake_git(args: tuple[str, ...], *, cwd: Path) -> tuple[int, str]:
        return responses.get((cwd, args), (1, ""))

    monkeypatch.setattr(authority, "_run_git_command", fake_git)
    return repo_root, vendor_root, responses


def test_exact_clean_tracked_gitlink_is_available(
    authority_checkout: tuple[Path, Path, dict[GitResponseKey, tuple[int, str]]],
) -> None:
    repo_root, vendor_root, _responses = authority_checkout

    result = authority.inspect_tools_vendor_authority(repo_root)

    assert result.available is True
    assert result.root == vendor_root
    assert result.expected_sha == authority.TOOLS_GITLINK_SHA
    assert result.reason is None


@pytest.mark.parametrize(
    ("index_output", "reason"),
    [
        ("", "gitlink entry is missing"),
        (
            "100644 " + authority.TOOLS_GITLINK_SHA + " 0\tvendor/ud-tools",
            "not the declared gitlink",
        ),
        (
            "160000 0000000000000000000000000000000000000000 0\tvendor/ud-tools",
            "does not match the declared pin",
        ),
        (
            "160000 "
            + authority.TOOLS_GITLINK_SHA
            + " 0\tvendor/ud-tools\n160000 "
            + authority.TOOLS_GITLINK_SHA
            + " 0\tvendor/ud-tools-copy",
            "gitlink entry is missing",
        ),
    ],
)
def test_tracked_entry_must_be_one_exact_declared_gitlink(
    authority_checkout: tuple[Path, Path, dict[GitResponseKey, tuple[int, str]]],
    index_output: str,
    reason: str,
) -> None:
    repo_root, _vendor_root, responses = authority_checkout
    responses[(repo_root, ("ls-files", "--stage", "--", "vendor/ud-tools"))] = (
        0,
        index_output,
    )

    result = authority.inspect_tools_vendor_authority(repo_root)

    assert result.available is False
    assert reason in (result.reason or "")


def test_checkout_head_must_match_declared_pin(
    authority_checkout: tuple[Path, Path, dict[GitResponseKey, tuple[int, str]]],
) -> None:
    repo_root, vendor_root, responses = authority_checkout
    responses[(vendor_root, ("rev-parse", "--verify", "HEAD"))] = (
        0,
        "0000000000000000000000000000000000000000",
    )

    result = authority.inspect_tools_vendor_authority(repo_root)

    assert result.available is False
    assert "HEAD does not match" in (result.reason or "")


@pytest.mark.parametrize(
    ("command", "cwd_selector", "output", "reason"),
    [
        (
            ("status", "--porcelain=v1", "--untracked-files=all"),
            "vendor",
            "?? rogue.py",
            "checkout is dirty",
        ),
        (
            (
                "status",
                "--porcelain=v1",
                "--ignore-submodules=none",
                "--",
                "vendor/ud-tools",
            ),
            "repo",
            " M vendor/ud-tools",
            "modified Tools gitlink",
        ),
        (
            ("rev-parse", "--show-superproject-working-tree"),
            "vendor",
            "",
            "not attached to this superproject",
        ),
    ],
)
def test_dirty_or_detached_checkout_is_unavailable(
    authority_checkout: tuple[Path, Path, dict[GitResponseKey, tuple[int, str]]],
    command: tuple[str, ...],
    cwd_selector: str,
    output: str,
    reason: str,
) -> None:
    repo_root, vendor_root, responses = authority_checkout
    cwd = vendor_root if cwd_selector == "vendor" else repo_root
    responses[(cwd, command)] = (0, output)

    result = authority.inspect_tools_vendor_authority(repo_root)

    assert result.available is False
    assert reason in (result.reason or "")


def test_reparse_point_checkout_is_unavailable(
    authority_checkout: tuple[Path, Path, dict[GitResponseKey, tuple[int, str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, vendor_root, _responses = authority_checkout
    real_check: Callable[[Path], bool] = authority._is_reparse_point
    monkeypatch.setattr(
        authority,
        "_is_reparse_point",
        lambda path: path == vendor_root or real_check(path),
    )

    result = authority.inspect_tools_vendor_authority(repo_root)

    assert result.available is False
    assert "missing or replaced" in (result.reason or "")


def test_independent_repository_directory_cannot_replace_gitlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "UpstreamDrift"
    vendor_root = repo_root / "vendor" / "ud-tools"
    (vendor_root / "src").mkdir(parents=True)
    (vendor_root / ".git").mkdir()
    responses = _valid_authority_responses(repo_root)
    monkeypatch.setattr(
        authority,
        "_run_git_command",
        lambda args, *, cwd: responses.get((cwd, args), (1, "")),
    )

    result = authority.inspect_tools_vendor_authority(repo_root)

    assert result.available is False
    assert "not an initialized gitlink worktree" in (result.reason or "")
