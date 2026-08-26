#!/usr/bin/env python3
"""Restore bounded Docker inputs from the exact checked-out Git commit.

The helper fails closed before changing the worktree when the checkout identity
or requested path set cannot be proven.  After restoration it verifies that all
requested tracked content matches ``HEAD``.  It never synthesizes replacement
files and it deliberately ignores submodule worktree dirt when checking a
superproject-owned directory such as ``src``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

RUNTIME_DOCKER_CONTEXT_PATHS = (
    "Dockerfile",
    ".dockerignore",
    "Cargo.toml",
    "rust_core",
    "requirements.lock",
    "scripts/config/pip_audit_waivers.json",
    "scripts/ci/check_pip_audit_waivers.py",
    "src",
    "pyproject.toml",
    "launch_golf_suite.py",
    "scripts/ci/start_api_server.py",
    ".env.example",
    "docker/entrypoint.sh",
)

_PROFILES = {"runtime": RUNTIME_DOCKER_CONTEXT_PATHS}
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class RehydrationError(RuntimeError):
    """Raised when exact tracked-path restoration cannot be proven."""


def _run_git(
    repo_root: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RehydrationError(f"git {' '.join(args)} failed: {detail}")
    return result


def _validate_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if not paths:
        raise ValueError("paths must contain at least one relative POSIX path")

    validated: list[str] = []
    for path in paths:
        if not isinstance(path, str):
            raise TypeError("each path must be a relative POSIX path string")
        posix = PurePosixPath(path)
        is_unsafe = (
            not path
            or path == "."
            or path.startswith(":")
            or "\\" in path
            or posix.is_absolute()
            or ".." in posix.parts
            or posix.as_posix() != path
        )
        if is_unsafe:
            raise ValueError(f"path must be a bounded relative POSIX path: {path!r}")
        if path in validated:
            raise ValueError(f"paths must be unique relative POSIX paths: {path!r}")
        validated.append(path)
    return tuple(validated)


def _verify_repo_root(repo_root: Path) -> None:
    top_level = _run_git(repo_root, ("rev-parse", "--show-toplevel"))
    actual_root = Path(top_level.stdout.strip()).resolve()
    if actual_root != repo_root:
        raise RehydrationError(
            f"repository root {actual_root} does not match requested root {repo_root}"
        )


def _verify_head(repo_root: Path, expected_head: str) -> None:
    if not isinstance(expected_head, str) or not _COMMIT_PATTERN.fullmatch(
        expected_head
    ):
        raise ValueError("expected_head must be a lowercase 40-character Git SHA")
    actual_head = _run_git(repo_root, ("rev-parse", "HEAD")).stdout.strip()
    if actual_head != expected_head:
        raise RehydrationError(
            f"checked-out HEAD {actual_head} does not match expected {expected_head}"
        )


def _preflight_tracked_paths(repo_root: Path, paths: Sequence[str]) -> None:
    missing: list[str] = []
    for path in paths:
        result = _run_git(repo_root, ("cat-file", "-e", f"HEAD:{path}"), check=False)
        if result.returncode != 0:
            missing.append(path)
    if missing:
        raise RehydrationError(f"not tracked at HEAD: {', '.join(missing)}")


def _verify_restored_paths(repo_root: Path, paths: Sequence[str]) -> None:
    deleted = _run_git(
        repo_root,
        ("--literal-pathspecs", "ls-files", "--deleted", "-z", "--", *paths),
    )
    if deleted.stdout:
        missing = deleted.stdout.replace("\0", ", ").rstrip(", ")
        raise RehydrationError(f"tracked paths remain absent after restore: {missing}")

    diff = _run_git(
        repo_root,
        (
            "--literal-pathspecs",
            "diff",
            "--quiet",
            "--no-ext-diff",
            "--ignore-submodules=all",
            "HEAD",
            "--",
            *paths,
        ),
        check=False,
    )
    if diff.returncode == 1:
        raise RehydrationError("restored tracked paths do not match HEAD")
    if diff.returncode != 0:
        detail = diff.stderr.strip() or "unknown git diff failure"
        raise RehydrationError(f"could not verify restored tracked paths: {detail}")


def rehydrate_tracked_paths(
    repo_root: Path,
    *,
    expected_head: str,
    paths: Sequence[str],
) -> tuple[str, ...]:
    """Restore and verify a bounded tracked path set from ``HEAD``.

    Preconditions:
        ``repo_root`` is the exact Git worktree root, ``expected_head`` is the
        checked-out 40-character commit SHA, and every path is a unique,
        tracked, relative POSIX path.
    Postcondition:
        Every requested superproject-owned path exists and matches ``HEAD``;
        no untracked path is created as a substitute.
    """

    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")
    resolved_root = repo_root.resolve()
    validated_paths = _validate_paths(paths)
    _verify_repo_root(resolved_root)
    _verify_head(resolved_root, expected_head)

    # Complete preflight before the first worktree mutation.  A miss therefore
    # cannot leave a partially restored build context.
    _preflight_tracked_paths(resolved_root, validated_paths)
    _run_git(
        resolved_root,
        (
            "--literal-pathspecs",
            "checkout",
            "--force",
            "HEAD",
            "--",
            *validated_paths,
        ),
    )
    _verify_head(resolved_root, expected_head)
    _verify_restored_paths(resolved_root, validated_paths)
    return validated_paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a bounded Docker build-context profile from HEAD."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--profile", choices=tuple(_PROFILES), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    restored = rehydrate_tracked_paths(
        args.repo_root,
        expected_head=args.expected_head,
        paths=_PROFILES[args.profile],
    )
    print(
        f"Verified {args.profile} context at {args.expected_head}: {', '.join(restored)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
