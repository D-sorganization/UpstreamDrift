#!/usr/bin/env python3
"""Check ADR numbering for duplicate IDs.

Reads ADR filenames from this script's directory, parses the leading
four-digit number from each filename, and exits with a non-zero status
code if any duplicate numbers are found.

Usage::

    python3 docs/adr/check_adr_numbering.py

Exit codes:
    0 — no duplicates found
    1 — one or more duplicate ADR numbers detected
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_adr_number(filename: str) -> int | None:
    """Extract the leading ADR number from a filename.

    Args:
        filename: The bare filename (e.g. ``0005-some-title.md``).

    Returns:
        The integer ADR number, or ``None`` if the filename does not
        match the ``NNNN-`` prefix pattern.
    """
    match = re.match(r"^(\d{4})-", filename)
    if match is None:
        return None
    return int(match.group(1))


def check_adr_numbering(adr_dir: Path) -> int:
    """Scan *adr_dir* for ADR files and report duplicate numbers.

    Args:
        adr_dir: Path to the directory containing ADR markdown files.

    Returns:
        Exit code: ``0`` if no duplicates, ``1`` if duplicates exist.
    """
    number_to_files: dict[int, list[str]] = defaultdict(list)

    for path in sorted(adr_dir.iterdir()):
        if not path.is_file() or path.suffix != ".md":
            continue
        adr_number = parse_adr_number(path.name)
        if adr_number is None:
            # Skip files that don't follow the NNNN- naming convention
            # (e.g. README.md, ADR_TEMPLATE.md, api-versioning.md).
            continue
        number_to_files[adr_number].append(path.name)

    duplicates = {
        num: files for num, files in number_to_files.items() if len(files) > 1
    }

    if not duplicates:
        print("ADR numbering OK — no duplicate IDs found.")
        return 0

    print("ERROR: Duplicate ADR numbers detected:", file=sys.stderr)
    for num, files in sorted(duplicates.items()):
        print(f"  ADR-{num:04d}: {', '.join(sorted(files))}", file=sys.stderr)
    print(
        "\nRename the conflicting files to use the next available ID.",
        file=sys.stderr,
    )
    return 1


def main() -> None:
    """Entry point — resolves ADR directory and runs the check."""
    adr_dir = Path(__file__).parent
    sys.exit(check_adr_numbering(adr_dir))


if __name__ == "__main__":
    main()
