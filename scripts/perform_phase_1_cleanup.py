from pathlib import Path


def cleanup() -> None:
    root = Path(".")

    # 1. Delete redundant requirements.txt
    print("--- Scanning for redundant requirements.txt files ---")
    req_files = list(root.rglob("requirements.txt"))
    for f in req_files:
        if f.resolve() == (root / "requirements.txt").resolve():
            continue

        print(f"Deleting: {f}")
        try:
            f.unlink()
        except OSError as e:
            print(f"Error deleting {f}: {e}")

    # 2. Delete duplicate matlab_quality_check.py
    print("\n--- Scanning for duplicate matlab_quality_check.py files ---")
    quality_checks = list(root.rglob("matlab_quality_check.py"))
    canonical_path = (
        root / "tools/matlab_utilities/scripts/matlab_quality_check.py"
    ).resolve()

    for f in quality_checks:
        if f.resolve() == canonical_path:
            print(f"Keeping canonical: {f}")
            continue

        print(f"Deleting duplicate: {f}")
        try:
            f.unlink()
        except OSError as e:
            print(f"Error deleting {f}: {e}")


if __name__ == "__main__":
    cleanup()
