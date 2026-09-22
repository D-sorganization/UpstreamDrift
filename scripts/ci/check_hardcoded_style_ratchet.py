#!/usr/bin/env python3
"""Fail CI if hardcoded-hex-color counts under ``src/tools`` grow.

Mirrors the web app's ``ui/src/utils/colorGuard`` guard and this repo's
``check_error_handling_ratchet.py`` pattern: ``src/tools`` GUIs should style
themselves from ``src.shared.python.theme`` (see
``src/shared/python/theme/tool_stylesheet.py``), not by hardcoding hex
literals in ``setStyleSheet`` strings. 45 such call sites existed at the time
of issue #8885; this ratchet seeds that count as a baseline (see
``scripts/config/hardcoded_style_baseline.json``) and fails if it grows, so
new code cannot add to the debt while the remaining sites are migrated
incrementally.

A file only counts once it actually calls ``setStyleSheet(`` -- a hex literal
used for something else (e.g. an OpenCV BGR tuple, a docstring example) does
not trip this guard unless the same file also builds a Qt stylesheet.

Exit codes:
    0 — counts are equal to or below baseline (CI passes)
    1 — the count exceeds baseline (CI fails)
    2 — script invocation error (missing baseline, bad arg, etc.)

Usage:
    python3 scripts/ci/check_hardcoded_style_ratchet.py
    python3 scripts/ci/check_hardcoded_style_ratchet.py --update-baseline  # lower-only

Filed as part of issue #8885.
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
TOOLS_DIR = REPO_ROOT / "src" / "tools"
BASELINE_PATH = REPO_ROOT / "scripts" / "config" / "hardcoded_style_baseline.json"

SET_STYLESHEET_PATTERN = re.compile(r"setStyleSheet\(")
HEX_COLOR_PATTERN = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")

# Files that legitimately implement or test the ratchet itself, or the
# shared stylesheet module it steers callers toward -- these would otherwise
# create false positives / circular exemptions.
SELF_EXEMPT = {
    REPO_ROOT / "scripts" / "ci" / "check_hardcoded_style_ratchet.py",
}


def _count_hex_literals(text: str) -> int:
    return len(HEX_COLOR_PATTERN.findall(text))


def _count_hardcoded_style_literals(tools_dir: pathlib.Path) -> int:
    """Walk *tools_dir* and count hex literals in files that call setStyleSheet.

    LOD: returns a single int, no side effects.
    """
    if not tools_dir.exists():
        raise FileNotFoundError(f"tools directory not found: {tools_dir}")
    total = 0
    for py_path in tools_dir.rglob("*.py"):
        if py_path in SELF_EXEMPT:
            continue
        try:
            text = py_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("skipping non-utf8 file: %s", py_path)
            continue
        if not SET_STYLESHEET_PATTERN.search(text):
            continue
        total += _count_hex_literals(text)
    return total


def _load_baseline(path: pathlib.Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"baseline file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    count = data.get("count")
    if not isinstance(count, int):
        raise ValueError(f"baseline {path} is missing required int 'count'")
    return count


def _write_baseline_count(path: pathlib.Path, new_count: int) -> None:
    """Update only the 'count' field of an existing baseline JSON."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data["count"] = new_count
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the desired process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline with the current count. Only allowed when "
        "the count is at or below baseline (i.e. lowering only).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        baseline = _load_baseline(BASELINE_PATH)
        current = _count_hardcoded_style_literals(TOOLS_DIR)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("hardcoded-style ratchet failed to run: %s", exc)
        return 2

    if current > baseline:
        logger.error(
            "Hardcoded-style ratchet violation: hex literals in "
            "setStyleSheet-bearing files under src/tools grew from %d to %d "
            "(+%d).",
            baseline,
            current,
            current - baseline,
        )
        logger.error(
            "\n"
            "How to fix:\n"
            "  - Use src.shared.python.theme.tool_stylesheet.apply_tool_theme()"
            " / primary_button_style() (or get_tool_colors()) instead of a"
            " hardcoded hex literal in setStyleSheet().\n"
            "  - To intentionally raise the baseline, edit "
            f"{BASELINE_PATH.relative_to(REPO_ROOT)} in the same PR with a"
            " justification in the PR description."
        )
        return 1

    if current < baseline:
        logger.info(
            "Hardcoded-style ratchet improved: %d -> %d (-%d).",
            baseline,
            current,
            baseline - current,
        )

    if args.update_baseline:
        _write_baseline_count(BASELINE_PATH, current)
        logger.info("Baseline updated to current count (lowering-only): %d", current)
    else:
        logger.info("Hardcoded-style ratchet: OK (count at or below baseline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
