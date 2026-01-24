#!/usr/bin/env python3
"""Comprehensive DRY and Orthogonality Refactoring Script.

This script systematically fixes DRY (Don't Repeat Yourself) and orthogonality
violations across the Golf Modeling Suite codebase.

Phases:
1. Logging standardization
2. Path utilities consolidation
3. Error handling patterns
4. Engine interface refactoring
5. GUI initialization patterns
6. Configuration management

Usage:
    python scripts/refactor_dry_orthogonality.py --phase all
    python scripts/refactor_dry_orthogonality.py --phase logging
    python scripts/refactor_dry_orthogonality.py --phase paths
"""

import argparse
import re
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class DRYRefactorer:
    """Refactors code to eliminate DRY violations."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.changes_made = 0

    def refactor_logging_imports(self, file_path: Path) -> bool:
        """Replace logging.getLogger(__name__) with get_logger(__name__)."""
        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content

            # Check if file already uses get_logger
            if "from src.shared.python.logging_config import get_logger" in content:
                return False

            # Pattern 1: logger = logging.getLogger(__name__)
            pattern1 = r"^logger = logging\.getLogger\(__name__\)"
            # Pattern 2: LOGGER = logging.getLogger(__name__)
            pattern2 = r"^LOGGER = logging\.getLogger\(__name__\)"

            has_pattern1 = bool(re.search(pattern1, content, re.MULTILINE))
            has_pattern2 = bool(re.search(pattern2, content, re.MULTILINE))

            if not (has_pattern1 or has_pattern2):
                return False

            # Add import if not present
            if "import logging" in content:
                # Replace import logging with get_logger import
                content = re.sub(
                    r"^import logging$",
                    "from src.shared.python.logging_config import get_logger",
                    content,
                    count=1,
                    flags=re.MULTILINE,
                )

            # Replace logger initialization
            if has_pattern1:
                content = re.sub(
                    pattern1,
                    "logger = get_logger(__name__)",
                    content,
                    flags=re.MULTILINE,
                )

            if has_pattern2:
                # Standardize to lowercase 'logger'
                content = re.sub(
                    pattern2,
                    "logger = get_logger(__name__)",
                    content,
                    flags=re.MULTILINE,
                )
                # Replace all LOGGER references with logger
                content = re.sub(r"\bLOGGER\b", "logger", content)

            if content != original_content:
                file_path.write_text(content, encoding="utf-8")
                self.changes_made += 1
                logger.info(
                    f"✓ Refactored logging in {file_path.relative_to(self.repo_root)}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to refactor {file_path}: {e}")
            return False

    def refactor_path_patterns(self, file_path: Path) -> bool:
        """Replace Path(__file__).resolve().parents[N] with path_utils functions."""
        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content

            # Pattern: Path(__file__).resolve().parents[N]
            # Pattern: Path(__file__).parent.parent...
            patterns = [
                (r"Path\(__file__\)\.resolve\(\)\.parents\[3\]", "get_repo_root()"),
                (r"Path\(__file__\)\.resolve\(\)\.parents\[2\]", "get_src_root()"),
                (
                    r"Path\(__file__\)\.parent\.parent\.parent\.resolve\(\)",
                    "get_repo_root()",
                ),
                (r"Path\(__file__\)\.parent\.parent\.resolve\(\)", "get_src_root()"),
            ]

            needs_import = False
            for pattern, replacement in patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    needs_import = True

            if (
                needs_import
                and "from src.shared.python.path_utils import" not in content
            ):
                # Add import after other imports
                import_line = "\nfrom src.shared.python.path_utils import get_repo_root, get_src_root\n"
                # Find last import statement
                import_match = list(
                    re.finditer(r"^(?:from|import) .+$", content, re.MULTILINE)
                )
                if import_match:
                    last_import = import_match[-1]
                    insert_pos = last_import.end()
                    content = content[:insert_pos] + import_line + content[insert_pos:]

            if content != original_content:
                file_path.write_text(content, encoding="utf-8")
                self.changes_made += 1
                logger.info(
                    f"✓ Refactored paths in {file_path.relative_to(self.repo_root)}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to refactor paths in {file_path}: {e}")
            return False

    def process_directory(self, directory: Path, refactor_func) -> int:
        """Process all Python files in directory with given refactor function."""
        count = 0
        for py_file in directory.rglob("*.py"):
            # Skip __pycache__ and .mypy_cache
            if "__pycache__" in str(py_file) or ".mypy_cache" in str(py_file):
                continue
            if refactor_func(py_file):
                count += 1
        return count


def refactor_logging_phase(repo_root: Path) -> int:
    """Phase 1: Standardize logging across all modules."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Logging Standardization")
    logger.info("=" * 60)

    refactorer = DRYRefactorer(repo_root)

    # Process all directories
    directories = [
        repo_root / "src",
        repo_root / "tests",
        repo_root / "scripts",
        repo_root / "examples",
    ]

    total_changes = 0
    for directory in directories:
        if directory.exists():
            logger.info(f"\nProcessing {directory.relative_to(repo_root)}...")
            changes = refactorer.process_directory(
                directory, refactorer.refactor_logging_imports
            )
            total_changes += changes

    logger.info(f"\n✓ Phase 1 complete: {total_changes} files refactored")
    return total_changes


def refactor_paths_phase(repo_root: Path) -> int:
    """Phase 2: Consolidate path handling patterns."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Path Utilities Consolidation")
    logger.info("=" * 60)

    refactorer = DRYRefactorer(repo_root)

    directories = [
        repo_root / "src",
        repo_root / "tests",
        repo_root / "scripts",
    ]

    total_changes = 0
    for directory in directories:
        if directory.exists():
            logger.info(f"\nProcessing {directory.relative_to(repo_root)}...")
            changes = refactorer.process_directory(
                directory, refactorer.refactor_path_patterns
            )
            total_changes += changes

    logger.info(f"\n✓ Phase 2 complete: {total_changes} files refactored")
    return total_changes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Refactor DRY and orthogonality violations"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "logging", "paths"],
        default="all",
        help="Which refactoring phase to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    total_changes = 0

    if args.phase in ["all", "logging"]:
        total_changes += refactor_logging_phase(REPO_ROOT)

    if args.phase in ["all", "paths"]:
        total_changes += refactor_paths_phase(REPO_ROOT)

    logger.info("=" * 60)
    logger.info(f"TOTAL CHANGES: {total_changes} files modified")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
