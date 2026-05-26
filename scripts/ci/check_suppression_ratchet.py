#!/usr/bin/env python3
"""Fail CI if bare suppression directives in src/ grow against the baseline.

Tracked directives:

* ``# type: ignore`` without an explicit error code list
* ``# noqa`` without explicit rule codes

Exit codes:
    0 — counts are equal to or below baseline (CI passes)
    1 — at least one count exceeds baseline (CI fails)
    2 — script invocation error (missing baseline, bad arg, etc.)

Usage:
    python3 scripts/ci/check_suppression_ratchet.py
    python3 scripts/ci/check_suppression_ratchet.py --update-baseline  # lower-only

Filed as part of issue #5916 (Adversarial review category I).
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
BASELINE_PATH = REPO_ROOT / "scripts" / "config" / "suppression_ratchet_baseline.json"

PATTERNS: dict[str, re.Pattern[str]] = {
    "bare_type_ignore": re.compile(r"#\s*type:\s*ignore(?!\[)"),
    "bare_noqa": re.compile(r"#\s*noqa(?!:)"),
}


def _count_patterns(src_dir: pathlib.Path) -> dict[str, int]:
    """Walk ``src_dir`` and count each tracked suppression pattern."""
    if not src_dir.exists():
        raise FileNotFoundError(f"source directory not found: {src_dir}")
    counts = dict.fromkeys(PATTERNS, 0)
    for py_path in src_dir.rglob("*.py"):
        try:
            text = py_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("skipping non-utf8 file: %s", py_path)
            continue
        for name, pattern in PATTERNS.items():
            counts[name] += len(pattern.findall(text))
    return counts


def _load_baseline(path: pathlib.Path) -> dict[str, int]:
    """Load baseline counts and validate the expected shape."""
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
    return {name: int(counts[name]) for name in PATTERNS}


def _write_baseline_counts(path: pathlib.Path, new_counts: dict[str, int]) -> None:
    """Rewrite only the counts field in the existing baseline file."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data["counts"] = new_counts
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def _report(
    baseline: dict[str, int], current: dict[str, int]
) -> tuple[list[str], list[str]]:
    """Compare baseline to current counts."""
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
        logger.error("suppression ratchet failed to run: %s", exc)
        return 2

    failures, improvements = _report(baseline, current)

    if improvements:
        logger.info("Suppression-discipline ratchet improvements:")
        for line in improvements:
            logger.info(line)

    if failures:
        logger.error("Suppression-discipline ratchet violations:")
        for line in failures:
            logger.error(line)
        logger.error(
            "\n"
            "How to fix:\n"
            "  - Replace bare `# type: ignore` with a coded form such as\n"
            "    `# type: ignore[attr-defined]`.\n"
            "  - Replace bare `# noqa` with explicit rule codes such as\n"
            "    `# noqa: F821`.\n"
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
        logger.info("Suppression-discipline ratchet: OK (counts at or below baseline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
