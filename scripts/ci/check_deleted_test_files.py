#!/usr/bin/env python3
"""Check for deleted Python test files in PRs against the merge base (#10751).

This check computes deletions relative to the merge base (common ancestor)
between the base branch and HEAD, rather than the tip of the base branch.
This prevents false positives when new tests were added to the base branch
after the PR branch was created.

Design by Contract (DbC), Law of Demeter (LoD), and DRY principles are enforced.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_contract_helpers() -> tuple[Any, Any]:
    """Load DbC helpers without importing ``src.shared.python`` package init."""
    path = REPO_ROOT / "src" / "shared" / "python" / "contracts.py"
    if path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("contracts_dbc", path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.precondition, module.postcondition
        except (ImportError, AttributeError, OSError) as exc:
            logger.debug("Failed to load contracts module: %s", exc)

    def _noop(pred: Any, desc: str = "") -> Any:
        def _wrapper(fn: Any) -> Any:
            return fn

        return _wrapper

    return _noop, _noop


precondition, postcondition = _load_contract_helpers()


class MergeBaseResolutionError(RuntimeError):
    """Raised when git cannot resolve the merge-base between base and head refs."""


@precondition(
    lambda args, repo_root: (
        isinstance(args, list)
        and all(isinstance(a, str) for a in args)
        and isinstance(repo_root, Path)
    )
)
@postcondition(
    lambda result: (
        isinstance(result, tuple)
        and len(result) == 3
        and isinstance(result[0], int)
        and isinstance(result[1], str)
        and isinstance(result[2], str)
    )
)
def run_git_command(args: list[str], repo_root: Path) -> tuple[int, str, str]:
    """Execute a git command within repo_root, returning (code, stdout, stderr)."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


@precondition(
    lambda base_ref, head_ref, repo_root: (
        isinstance(base_ref, str)
        and bool(base_ref.strip())
        and isinstance(head_ref, str)
        and bool(head_ref.strip())
        and isinstance(repo_root, Path)
    )
)
@postcondition(
    lambda result: result is None or (isinstance(result, str) and len(result) >= 4)
)
def resolve_merge_base(base_ref: str, head_ref: str, repo_root: Path) -> str | None:
    """Resolve the common ancestor (merge-base) commit SHA between two refs."""
    code, stdout, stderr = run_git_command(
        ["merge-base", base_ref, head_ref], repo_root
    )
    if code != 0:
        logger.debug(
            "git merge-base failed between %s and %s: %s",
            base_ref,
            head_ref,
            stderr.strip(),
        )
        return None
    sha = stdout.strip()
    return sha or None


@precondition(
    lambda base_ref, head_ref, test_pattern, repo_root, fallback_to_base=False: (
        isinstance(base_ref, str)
        and bool(base_ref.strip())
        and isinstance(head_ref, str)
        and bool(head_ref.strip())
        and isinstance(test_pattern, str)
        and bool(test_pattern.strip())
        and (repo_root is None or isinstance(repo_root, Path))
        and isinstance(fallback_to_base, bool)
    )
)
@postcondition(
    lambda result: isinstance(result, list) and all(isinstance(p, str) for p in result)
)
def detect_deleted_test_files(
    base_ref: str,
    head_ref: str = "HEAD",
    test_pattern: str = "tests/**/*.py",
    repo_root: Path | None = None,
    fallback_to_base: bool = False,
) -> list[str]:
    """Detect test files deleted between the merge-base of base_ref and head_ref.

    Raises:
        MergeBaseResolutionError: If merge-base cannot be resolved and fallback_to_base is False.
    """
    root = repo_root or REPO_ROOT
    merge_base = resolve_merge_base(base_ref, head_ref, root)

    if merge_base is None:
        if fallback_to_base:
            logger.warning(
                "Cannot resolve merge-base between '%s' and '%s'; falling back to base ref",
                base_ref,
                head_ref,
            )
            diff_target = base_ref
        else:
            raise MergeBaseResolutionError(
                f"Cannot resolve merge-base between '{base_ref}' and '{head_ref}'. "
                "Ensure sufficient git fetch depth is available in the checkout."
            )
    else:
        diff_target = merge_base

    code, stdout, stderr = run_git_command(
        [
            "diff",
            "--name-only",
            "--diff-filter=D",
            diff_target,
            head_ref,
            "--",
            test_pattern,
        ],
        root,
    )
    if code != 0:
        raise RuntimeError(
            f"git diff failed between {diff_target} and {head_ref}: {stderr.strip()}"
        )

    deleted_files = [
        line.strip().replace("\\", "/") for line in stdout.splitlines() if line.strip()
    ]
    return sorted(deleted_files)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for checking deleted Python test files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base branch or commit to compute merge-base against (default: origin/main)",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head commit or branch (default: HEAD)",
    )
    parser.add_argument(
        "--test-pattern",
        default="tests/**/*.py",
        help="Path glob pattern for test files (default: tests/**/*.py)",
    )
    parser.add_argument(
        "--fallback-to-base",
        action="store_true",
        help="Fall back to base-ref if merge-base cannot be resolved",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write list of deleted test files",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory (default: current repository root)",
    )

    args = parser.parse_args(argv)

    try:
        deleted = detect_deleted_test_files(
            base_ref=args.base_ref,
            head_ref=args.head,
            test_pattern=args.test_pattern,
            repo_root=args.repo_root,
            fallback_to_base=args.fallback_to_base,
        )
    except MergeBaseResolutionError as err:
        sys.stderr.write(f"::error::{err}\n")
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(deleted) + ("\n" if deleted else ""), encoding="utf-8"
        )

    if deleted:
        sys.stderr.write(
            "::error::Deleted Python test files require review before CI can proceed.\n"
        )
        for file_path in deleted:
            sys.stderr.write(f"  {file_path}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
