#!/usr/bin/env python3
"""Fail CI if error-handling anti-pattern counts grow against the baseline.

Patterns tracked (see ``scripts/config/error_handling_baseline.json``):

* ``# noqa: BLE001`` — grandfathered blind ``except Exception``
* ``# noqa: F841``   — grandfathered unused local variable
* ``# noqa: F401``   — grandfathered unused import
* raw ``subprocess.Popen(`` — should use
  ``src.shared.python.core.process_safety.managed_popen``
* ``asyncio.gather(`` without ``return_exceptions=`` — should use
  ``src.shared.python.core.process_safety.safe_gather``

Exit codes:
    0 — counts are equal to or below baseline (CI passes)
    1 — at least one count exceeds baseline (CI fails)
    2 — script invocation error (missing baseline, bad arg, etc.)

Usage:
    python3 scripts/ci/check_error_handling_ratchet.py
    python3 scripts/ci/check_error_handling_ratchet.py --update-baseline  # lower-only

This script is invoked from ci-standard.yml after ``ruff check``.

Filed as part of issue #5911 (Adversarial review category D).
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys

logger = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
BASELINE_PATH = REPO_ROOT / "scripts" / "config" / "error_handling_baseline.json"

PATTERNS: dict[str, re.Pattern[str]] = {
    "noqa_BLE001": re.compile(r"#\s*noqa\s*:[^#]*BLE001"),
    "noqa_F841": re.compile(r"#\s*noqa\s*:[^#]*F841"),
    "noqa_F401": re.compile(r"#\s*noqa\s*:[^#]*F401"),
    "raw_popen": re.compile(r"subprocess\.Popen\("),
    "gather_no_return_exceptions": re.compile(r"asyncio\.gather\("),
}

# Files that legitimately implement or test the ratchet/helpers themselves.
# These would otherwise create false positives.
SELF_EXEMPT = {
    REPO_ROOT / "src" / "shared" / "python" / "core" / "process_safety.py",
    REPO_ROOT / "scripts" / "ci" / "check_error_handling_ratchet.py",
}


def _count_gather_without_return_exceptions(text: str) -> int:
    """Count ``asyncio.gather(...)`` calls that omit ``return_exceptions``."""
    count = 0
    needle = "asyncio.gather("
    search_start = 0
    while True:
        gather_start = text.find(needle, search_start)
        if gather_start == -1:
            return count

        cursor = gather_start + len(needle)
        depth = 1
        while cursor < len(text) and depth > 0:
            char = text[cursor]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            cursor += 1

        args_end = cursor - 1 if depth == 0 else len(text)
        args_text = text[gather_start + len(needle) : args_end]
        if "return_exceptions" not in args_text:
            count += 1
        search_start = max(cursor, gather_start + len(needle))


def _count_patterns(src_dir: pathlib.Path) -> dict[str, int]:
    """Walk ``src_dir`` and count each pattern. LOD: returns dict, no side effects."""
    if not src_dir.exists():
        raise FileNotFoundError(f"source directory not found: {src_dir}")
    counts = dict.fromkeys(PATTERNS, 0)
    for py_path in src_dir.rglob("*.py"):
        if py_path in SELF_EXEMPT:
            continue
        try:
            text = py_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Tolerate odd encodings rather than failing the whole CI run
            logger.warning("skipping non-utf8 file: %s", py_path)
            continue
        for name, pat in PATTERNS.items():
            if name == "gather_no_return_exceptions":
                counts[name] += _count_gather_without_return_exceptions(text)
            else:
                counts[name] += len(pat.findall(text))
    return counts


def _load_baseline(path: pathlib.Path) -> dict[str, int]:
    """Load baseline counts. Validates that all PATTERNS keys are present."""
    if not path.exists():
        raise FileNotFoundError(f"baseline file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    counts = data.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"baseline {path} is missing required 'counts' object")
    missing = set(PATTERNS) - set(counts)
    if missing:
        raise ValueError(f"baseline {path} is missing required keys: {sorted(missing)}")
    return {k: int(counts[k]) for k in PATTERNS}


def _write_baseline_counts(path: pathlib.Path, new_counts: dict[str, int]) -> None:
    """Update only the 'counts' field of an existing baseline JSON.

    Preserves comments and policy text by reading the JSON, mutating in place,
    and writing back. (Standard json drops comments, but this baseline uses
    underscore-prefixed keys as comments which round-trip cleanly.)
    """
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data["counts"] = new_counts
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def _report(
    baseline: dict[str, int], current: dict[str, int]
) -> tuple[list[str], list[str]]:
    """Compare counts. Returns (failures, improvements) as printable strings."""
    failures: list[str] = []
    improvements: list[str] = []
    for name in PATTERNS:
        base = baseline[name]
        cur = current[name]
        if cur > base:
            failures.append(
                f"  REGRESSION  {name}: was {base}, now {cur} (+{cur - base})"
            )
        elif cur < base:
            improvements.append(
                f"  IMPROVED    {name}: was {base}, now {cur} (-{base - cur})"
            )
    return failures, improvements


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the desired process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline with the current counts. Only allowed when "
        "every count is at or below baseline (i.e. lowering only).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        baseline = _load_baseline(BASELINE_PATH)
        current = _count_patterns(SRC_DIR)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("ratchet check failed to run: %s", exc)
        return 2

    failures, improvements = _report(baseline, current)

    if improvements:
        logger.info("Error-handling ratchet improvements:")
        for line in improvements:
            logger.info(line)

    if failures:
        logger.error("Error-handling ratchet violations:")
        for line in failures:
            logger.error(line)
        logger.error(
            "\n"
            "How to fix:\n"
            "  - Replace `except Exception:` with `narrow_catch(SpecificError,"
            " ...)` from src.shared.python.core.process_safety.\n"
            "  - Replace raw `subprocess.Popen(...)` with `managed_popen(...)`"
            " from the same module.\n"
            "  - Replace `asyncio.gather(...)` with `safe_gather(...)` (also"
            " from process_safety).\n"
            "  - To intentionally raise the baseline, edit "
            f"{BASELINE_PATH.relative_to(REPO_ROOT)} in the same PR with a"
            " justification in the PR description."
        )
        return 1

    if args.update_baseline:
        _write_baseline_counts(BASELINE_PATH, current)
        logger.info(
            "Baseline updated to current counts (lowering-only): %s",
            current,
        )
    else:
        logger.info("Error-handling ratchet: OK (counts at or below baseline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
