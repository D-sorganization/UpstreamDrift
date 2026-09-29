#!/usr/bin/env python3
"""Fail CI if bare-input counts grow against the UX baseline.

Tracks input widgets that are NOT wrapped in the HelpfulField helpers
introduced in Phase 0 of epic #5968.  Modelled on
``scripts/ci/check_error_handling_ratchet.py`` — same baseline-file
shape, same exit codes, same ``--update-baseline`` (lower-only) flag.

Patterns counted (per file under ``src/`` and ``ui/src/``):

* Python: bare ``QSpinBox(``, ``QDoubleSpinBox(``, ``QComboBox(``,
  ``QSlider(``, ``QLineEdit(`` — flagged unless the file has a
  ``# noqa: ux/no-bare-field`` marker on the same line or imports
  ``HelpfulField``.
* TypeScript/TSX: bare ``<input``, ``<select``, ``<textarea`` —
  flagged unless the line carries a ``// ux:noqa`` comment or the
  file imports ``HelpfulField``.

The baseline at ``scripts/config/ux_field_coverage_baseline.json``
records the pre-existing count for each pattern.  The ratchet only
allows the count to **stay flat or shrink**, never grow.

Usage::

    python3 scripts/ci/check_ux_coverage_ratchet.py
    python3 scripts/ci/check_ux_coverage_ratchet.py --update-baseline

Exit codes:
    0 — counts ≤ baseline (CI passes)
    1 — at least one count exceeds baseline (CI fails)
    2 — script invocation error
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
PY_ROOT = REPO_ROOT / "src"
WEB_ROOT = REPO_ROOT / "ui" / "src"
BASELINE_PATH = REPO_ROOT / "scripts" / "config" / "ux_field_coverage_baseline.json"

PY_PATTERNS: dict[str, re.Pattern[str]] = {
    "bare_qspinbox": re.compile(r"\bQSpinBox\("),
    "bare_qdoublespinbox": re.compile(r"\bQDoubleSpinBox\("),
    "bare_qcombobox": re.compile(r"\bQComboBox\("),
    "bare_qslider": re.compile(r"\bQSlider\("),
    "bare_qlineedit": re.compile(r"\bQLineEdit\("),
}

WEB_PATTERNS: dict[str, re.Pattern[str]] = {
    "bare_html_input": re.compile(r"<input\b"),
    "bare_html_select": re.compile(r"<select\b"),
    "bare_html_textarea": re.compile(r"<textarea\b"),
}

PY_NOQA = re.compile(r"#\s*noqa:\s*ux/no-bare-field")
WEB_NOQA = re.compile(r"//\s*ux:noqa")
PY_IMPORT_OK = re.compile(r"from\s+src\.shared\.python\.ui\.helpful_field\b")
WEB_IMPORT_OK = re.compile(r"from\s+['\"][^'\"]*ux/HelpfulField['\"]")

SELF_EXEMPT: set[pathlib.Path] = {
    REPO_ROOT / "scripts" / "ci" / "check_ux_coverage_ratchet.py",
}


def _count_in_file(
    path: pathlib.Path,
    patterns: dict[str, re.Pattern[str]],
    noqa_pattern: re.Pattern[str],
    import_ok_pattern: re.Pattern[str],
) -> dict[str, int]:
    counts = dict.fromkeys(patterns, 0)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("skipping non-utf8 file: %s", path)
        return counts
    file_is_wrapped = bool(import_ok_pattern.search(text))
    for line in text.splitlines():
        if noqa_pattern.search(line):
            continue
        if file_is_wrapped:
            # The file already adopts HelpfulField; we tolerate a few
            # bare-input lines (often within HelpfulField's internals).
            continue
        for name, pat in patterns.items():
            counts[name] += len(pat.findall(line))
    return counts


def _walk(
    root: pathlib.Path,
    suffixes: tuple[str, ...],
    patterns: dict[str, re.Pattern[str]],
    noqa: re.Pattern[str],
    import_ok: re.Pattern[str],
) -> dict[str, int]:
    totals = dict.fromkeys(patterns, 0)
    if not root.exists():
        return totals
    for path in root.rglob("*"):
        if path.suffix not in suffixes or not path.is_file():
            continue
        if path in SELF_EXEMPT:
            continue
        for name, count in _count_in_file(path, patterns, noqa, import_ok).items():
            totals[name] += count
    return totals


def count_all() -> dict[str, int]:
    py_totals = _walk(PY_ROOT, (".py",), PY_PATTERNS, PY_NOQA, PY_IMPORT_OK)
    web_totals = _walk(WEB_ROOT, (".tsx", ".ts"), WEB_PATTERNS, WEB_NOQA, WEB_IMPORT_OK)
    return {**py_totals, **web_totals}


def _load_baseline(path: pathlib.Path) -> dict[str, int]:
    if not path.exists():
        return dict.fromkeys({**PY_PATTERNS, **WEB_PATTERNS}, 0)
    with path.open(encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"baseline {path} must be a JSON object")
    return {str(k): int(v) for k, v in loaded.items()}


def _write_baseline(path: pathlib.Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(counts, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Lower the baseline to the current counts (never raise).",
    )
    args = parser.parse_args(argv)

    current = count_all()
    baseline = _load_baseline(BASELINE_PATH)

    if args.update_baseline:
        # If no baseline exists yet, seed with current counts.  After
        # that, only allow the baseline to shrink (lower-only ratchet
        # — never grow), matching check_error_handling_ratchet.py.
        if not BASELINE_PATH.exists():
            merged = dict(current)
        else:
            merged = {
                k: min(current.get(k, 0), baseline.get(k, current.get(k, 0)))
                for k in {*current, *baseline}
            }
        _write_baseline(BASELINE_PATH, merged)
        logger.info("baseline updated: %s", merged)
        return 0

    failed: list[str] = []
    for name in sorted({*current, *baseline}):
        cur = current.get(name, 0)
        base = baseline.get(name, 0)
        if cur > base:
            failed.append(f"{name}: {cur} > baseline {base}")
    if failed:
        for line in failed:
            sys.stderr.write(f"UX coverage ratchet FAILED: {line}\n")
        sys.stderr.write(
            "Hint: wrap the new widget in HelpfulField "
            "(src/shared/python/ui/helpful_field.py or "
            "ui/src/components/ux/HelpfulField.tsx) "
            "or annotate with `# noqa: ux/no-bare-field` / `// ux:noqa` "
            "with a justification.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
