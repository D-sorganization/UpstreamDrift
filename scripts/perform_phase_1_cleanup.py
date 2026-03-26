import contextlib
from pathlib import Path


def cleanup() -> None:
    """Delete redundant requirements.txt and duplicate quality check files."""
    root = Path(".")

    # 1. Delete redundant requirements.txt
    req_files = list(root.rglob("requirements.txt"))
    for f in req_files:
        if f.resolve() == (root / "requirements.txt").resolve():
            continue

        with contextlib.suppress(OSError):
            f.unlink()

    # 2. Delete duplicate matlab_quality_check.py
    quality_checks = list(root.rglob("matlab_quality_check.py"))
    canonical_path = (
        root / "tools/matlab_utilities/scripts/matlab_quality_check.py"
    ).resolve()

    for f in quality_checks:
        if f.resolve() == canonical_path:
            continue

        with contextlib.suppress(OSError):
            f.unlink()


if __name__ == "__main__":
    cleanup()
