#!/usr/bin/env python3
"""
Setup script for installing pre-commit and pre-push hooks.

This script:
1. Installs pre-commit if not present
2. Installs pre-commit hooks
3. Installs pre-push hooks
4. Verifies the installation

Usage:
    python scripts/setup_hooks.py
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    if not (cmd is not None):
        raise ValueError("cmd must be provided")
    if not isinstance(cmd, list):
        raise ValueError("cmd must be a list")
    if not isinstance(check, bool):
        raise ValueError("check must be a boolean")
    logger.info("  Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def check_pre_commit_installed() -> bool:
    """Check if pre-commit is installed."""
    try:
        result = run_command(["pre-commit", "--version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_pre_commit() -> None:
    """Install pre-commit via pip."""
    logger.info("\n[1/4] Installing pre-commit...")
    if check_pre_commit_installed():
        logger.info("  pre-commit is already installed")
    else:
        run_command([sys.executable, "-m", "pip", "install", "pre-commit"])
        logger.info("  pre-commit installed successfully")


def install_hooks() -> None:
    """Install pre-commit hooks."""
    logger.info("\n[2/4] Installing pre-commit hooks...")
    run_command(["pre-commit", "install"])
    logger.info("  pre-commit hooks installed")


def install_push_hooks() -> None:
    """Install pre-push hooks using a Python wrapper for cross-platform compatibility.

    Replaces the bare ``pytest`` call (which fails on Windows/venvs) with a
    ``python -m pytest`` invocation so the correct environment's pytest is always used.
    Fixes: https://github.com/D-sorganization/UpstreamDrift/issues/2037
    """
    logger.info("\n[3/4] Installing pre-push hooks...")
    hook_src = Path("scripts/pre_push_hook.py")
    hook_dst = Path(".git/hooks/pre-push")

    if not hook_src.exists():
        logger.warning("  pre_push_hook.py not found — skipping pre-push hook install")
        return

    hook_content = (
        "#!/bin/sh\n"
        f'"{sys.executable}" "$(git rev-parse --show-toplevel)/scripts/pre_push_hook.py"\n'
    )
    hook_dst.write_text(hook_content)
    # Make executable on Unix
    if sys.platform != "win32":
        import stat

        hook_dst.chmod(
            hook_dst.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        )
    logger.info("  pre-push hook installed (uses python -m pytest)")


def install_dev_dependencies() -> None:
    """Install development dependencies for hooks."""
    logger.info("\n[4/4] Installing hook dependencies...")
    deps = [
        "ruff>=0.14.0",
        "black>=26.0.0",
        "mypy>=1.13.0",
        "bandit>=1.7.0",
        "types-requests",
        "types-PyYAML",
        "pydantic",
    ]
    run_command([sys.executable, "-m", "pip", "install"] + deps)
    logger.info("  Dependencies installed")


def verify_installation() -> None:
    """Verify hooks are installed correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    git_hooks_dir = Path(".git/hooks")
    pre_commit_hook = git_hooks_dir / "pre-commit"
    pre_push_hook = git_hooks_dir / "pre-push"

    if pre_commit_hook.exists():
        logger.info("  [OK] pre-commit hook: %s", pre_commit_hook)
    else:
        logger.warning("  [MISSING] pre-commit hook: %s", pre_commit_hook)

    if pre_push_hook.exists():
        logger.info("  [OK] pre-push hook: %s", pre_push_hook)
    else:
        logger.warning("  [MISSING] pre-push hook: %s", pre_push_hook)


def log_summary() -> None:
    """Log usage summary."""
    logger.info("\n" + "=" * 60)
    logger.info("HOOK SUMMARY")
    logger.info("=" * 60)
    logger.info(
        """
PRE-COMMIT (runs on every commit, <15 seconds):
  - ruff (lint + auto-fix)
  - black (format)
  - no-wildcard-imports
  - quality-check (no TODOs/FIXMEs)
  - no-debug-statements
  - no-print-in-src
  - prettier (yaml/json/md)

PRE-PUSH (runs before push, ~60 seconds):
  - mypy (type check)
  - bandit (security scan)
  - pytest (unit tests)

MANUAL COMMANDS:
  pre-commit run --all-files      # Run all pre-commit hooks
  pre-commit run --hook-stage pre-push  # Run pre-push hooks manually
  pre-commit autoupdate           # Update hook versions
"""
    )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 60)
    logger.info("INSTALLING GIT HOOKS")
    logger.info("=" * 60)

    try:
        install_pre_commit()
        install_hooks()
        install_push_hooks()
        install_dev_dependencies()
        verify_installation()
        log_summary()
        logger.info("\n[SUCCESS] All hooks installed successfully!")
        logger.info("Your commits will now be checked locally before reaching CI.")

    except subprocess.CalledProcessError as e:
        logger.error("\n[ERROR] Command failed: %s", e)
        logger.error("  stdout: %s", e.stdout)
        logger.error("  stderr: %s", e.stderr)
        sys.exit(1)
    except (RuntimeError, OSError) as e:
        logger.error("\n[ERROR] Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
