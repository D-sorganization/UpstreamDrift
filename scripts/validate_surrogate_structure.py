#!/usr/bin/env python3
"""Validate Option-2 surrogate structure and dependencies (Issue #4075)."""

import json
import sys
from pathlib import Path


def check_file_exists(path: str | Path, description: str) -> bool:
    """Check if a file exists and report."""
    p = Path(path)
    if p.exists():
        print(f"✓ {description}: {p}")
        return True
    else:
        print(f"✗ {description} missing: {p}")
        return False


def check_directory_exists(path: str | Path, description: str) -> bool:
    """Check if a directory exists and report."""
    p = Path(path)
    if p.is_dir():
        print(f"✓ {description}: {p}")
        return True
    else:
        print(f"✗ {description} missing: {p}")
        return False


def check_json_valid(path: str | Path) -> bool:
    """Check if a JSON file is valid."""
    p = Path(path)
    if not p.exists():
        print(f"✗ JSON file not found: {p}")
        return False

    try:
        with open(p) as f:
            json.load(f)
        print(f"✓ JSON valid: {p}")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ JSON invalid: {p} - {e}")
        return False


def main() -> int:
    """Validate surrogate structure."""
    print("=" * 70)
    print("OPTION-2 SURROGATE STRUCTURE VALIDATION")
    print("=" * 70)

    checks = []

    # Check core surrogate modules
    print("\nCore Modules:")
    checks.append(
        check_file_exists(
            "src/shared/python/motion_matching/surrogate/__init__.py",
            "Surrogate package init",
        )
    )
    checks.append(
        check_file_exists(
            "src/shared/python/motion_matching/surrogate/model.py",
            "SwingSurrogate model",
        )
    )
    checks.append(
        check_file_exists(
            "src/shared/python/motion_matching/surrogate/train.py",
            "Training loop",
        )
    )

    # Check training entry points
    print("\nTraining Entry Points:")
    checks.append(
        check_file_exists(
            "src/shared/python/motion_matching/surrogate/train_10k.py",
            "train_10k.py (training script)",
        )
    )

    # Check evaluation notebook
    print("\nEvaluation Notebook:")
    checks.append(
        check_json_valid(
            "notebooks/evaluate_surrogate.ipynb",
        )
    )

    # Check dataset loader
    print("\nDataset Support:")
    checks.append(
        check_file_exists(
            "src/shared/python/motion_matching/dataset/__init__.py",
            "Dataset package",
        )
    )

    # Check tests
    print("\nTest Suite:")
    checks.append(
        check_file_exists(
            "tests/unit/motion_matching/test_surrogate_train_10k.py",
            "Surrogate training tests",
        )
    )

    # Check documentation
    print("\nDocumentation:")
    checks.append(
        check_file_exists(
            "docs/motion_matching/SURROGATE_TRAINING_GUIDE.md",
            "Surrogate training guide",
        )
    )

    # Summary
    print("\n" + "=" * 70)
    passed = sum(checks)
    total = len(checks)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 70)

    if passed == total:
        print("\n✓ All structure checks passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
