"""Tests for the immutable Tools gitlink authority boundary."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from src.shared.python.config import tools_vendor_authority as authority

pytestmark = pytest.mark.unit

GitResponseKey = tuple[Path, tuple[str, ...]]

# Deliberately fake pin used by the fixture repo: the authority derives the
# expected SHA from the tracked gitlink, so tests pin whatever the fixture
# index declares (no duplicated hand-maintained constant to go stale, #8852).
_PIN = "aa" * 20
_OTHER_SHA = "bb" * 20

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_expected_sha_is_derived_from_the_real_tracked_gitlink() -> None:
    """The pin authority must equal the actual vendor/ud-tools gitlink.

    This is the guard issue #8852 asked for: it compares the module's
    expected SHA against ``git ls-tree HEAD vendor/ud-tools`` in this very
    repository, so a governed submodule bump is validated automatically and
    a drifting duplicate constant cannot exist.
    """
    completed = subprocess.run(  # noqa: S603  # nosec B603
        ["git", "ls-tree", "HEAD", "--", "vendor/ud-tools"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        pytest.skip("git or the vendor/ud-tools gitlink is unavailable")
    mode, _obj_type, rest = completed.stdout.split(maxsplit=2)
    tree_sha = rest.split()[0]
    assert mode == "160000"

    derived = authority.expected_tools_gitlink_sha(REPO_ROOT)

    assert derived == tree_sha


def test_expected_sha_requires_a_path() -> None:
    with pytest.raises(TypeError):
        authority.expected_tools_gitlink_sha("not-a-path")  # type: ignore[arg-type]


def _valid_authority_responses(
    repo_root: Path,
) -> dict[GitResponseKey, tuple[int, str]]:
    vendor_root = repo_root / "vendor" / "ud-tools"
    pin = _PIN
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
    assert result.expected_sha == _PIN
    assert result.reason is None


@pytest.mark.parametrize(
    ("index_output", "reason"),
    [
        ("", "gitlink entry is missing"),
        (
            "100644 " + _PIN + " 0\tvendor/ud-tools",
            "not the declared gitlink",
        ),
        (
            "160000 not-forty-hex-characters 0\tvendor/ud-tools",
            "gitlink SHA is malformed",
        ),
        (
            "160000 "
            + _PIN
            + " 0\tvendor/ud-tools\n160000 "
            + _PIN
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


def test_bumped_gitlink_without_synced_checkout_is_not_silently_available(
    authority_checkout: tuple[Path, Path, dict[GitResponseKey, tuple[int, str]]],
) -> None:
    """A one-line gitlink bump re-pins everything; an unsynced checkout fails."""
    repo_root, _vendor_root, responses = authority_checkout
    responses[(repo_root, ("ls-files", "--stage", "--", "vendor/ud-tools"))] = (
        0,
        f"160000 {_OTHER_SHA} 0\tvendor/ud-tools",
    )

    result = authority.inspect_tools_vendor_authority(repo_root)

    assert result.available is False
    assert result.expected_sha == _OTHER_SHA
    assert "not synchronized to the tracked gitlink" in (result.reason or "")


def test_stale_checkout_head_reports_expected_and_found_shas(
    authority_checkout: tuple[Path, Path, dict[GitResponseKey, tuple[int, str]]],
) -> None:
    """A stale pin must be explicit — never a silent fail-closed (#8852)."""
    repo_root, vendor_root, responses = authority_checkout
    responses[(vendor_root, ("rev-parse", "--verify", "HEAD"))] = (0, _OTHER_SHA)

    result = authority.inspect_tools_vendor_authority(repo_root)

    assert result.available is False
    assert result.reason == (f"Tools pin stale (expected {_PIN}, found {_OTHER_SHA})")


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
