#!/usr/bin/env python3
"""Fail CI if the count of un-migrated blocking GUI handlers grows.

Issue #8880: exactly one file under ``src/tools/`` used ``QThread`` before
the shared :mod:`src.tools.async_action` helper landed, and everything else
wired a button's ``clicked`` signal straight to a handler that ran its
compute inline on the GUI thread. Migrating all ~25 tools is not a single
PR (`bunker_shot_gui`, `simulation_backends_launcher` and
`launch_monitor_analytics` are the ones done so far); this ratchet is the
guard that stops the remaining tools from growing in number while future
PRs migrate them one at a time.

Modelled on ``scripts/ci/check_ux_coverage_ratchet.py`` and
``scripts/ci/check_error_handling_ratchet.py`` -- same baseline-file shape,
same exit codes, same ``--update-baseline`` (lower-only) flag.

Heuristic (deliberately coarse -- a file-level signal, not a per-handler
one): a ``src/tools/*/gui*.py`` file counts as "needs migration" when it
wires at least one ``QPushButton`` to a ``.clicked.connect(...)`` handler
and does **not** import :mod:`src.tools.async_action`. A file that already
runs its background work another legitimate way (e.g. ``QProcess``, as
``motion_matching`` and ``golf_simulator`` do) can suppress the false
positive with a ``# noqa: gui-thread/ok`` comment anywhere in the file.

This does not (and cannot, at file granularity) prove every handler in a
migrated file is actually threaded -- that is what
``tests/tools/<tool>/test_async_actions.py`` proves per tool, following the
pattern in ``tests/ui/tools/simulation_backends/test_async_actions.py``.
It only stops the fleet-wide count of obviously-unmigrated files from
growing silently.

Usage::

    python3 scripts/ci/check_gui_thread_blocking_ratchet.py
    python3 scripts/ci/check_gui_thread_blocking_ratchet.py --update-baseline

Exit codes:
    0 -- count <= baseline (CI passes)
    1 -- count exceeds baseline (CI fails)
    2 -- script invocation error
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
TOOLS_ROOT = REPO_ROOT / "src" / "tools"
BASELINE_PATH = REPO_ROOT / "scripts" / "config" / "gui_thread_blocking_baseline.json"

_METRIC = "unmigrated_gui_files"

_HAS_BUTTON = re.compile(r"\bQPushButton\(")
_HAS_CLICK_CONNECT = re.compile(r"\.clicked\.connect\(")
_IMPORTS_ASYNC_ACTION = re.compile(r"\bsrc\.tools\.async_action\b")
_NOQA = re.compile(r"#\s*noqa:\s*gui-thread/ok")


def _needs_migration(text: str) -> bool:
    """Return whether *text* looks like an un-migrated blocking GUI file."""
    if _NOQA.search(text):
        return False
    if _IMPORTS_ASYNC_ACTION.search(text):
        return False
    return bool(_HAS_BUTTON.search(text) and _HAS_CLICK_CONNECT.search(text))


def _iter_gui_files() -> list[pathlib.Path]:
    if not TOOLS_ROOT.exists():
        return []
    return sorted(TOOLS_ROOT.glob("*/gui*.py"))


def find_unmigrated_files() -> list[pathlib.Path]:
    """Return every ``src/tools/*/gui*.py`` file flagged as needing migration."""
    offenders = []
    for path in _iter_gui_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("skipping non-utf8 file: %s", path)
            continue
        if _needs_migration(text):
            offenders.append(path)
    return offenders


def _load_baseline(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict) or _METRIC not in loaded:
        raise ValueError(f"baseline {path} must be a JSON object with {_METRIC!r}")
    return int(loaded[_METRIC])


def _write_baseline(path: pathlib.Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({_METRIC: count}, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Lower the baseline to the current count (never raise).",
    )
    args = parser.parse_args(argv)

    offenders = find_unmigrated_files()
    current = len(offenders)

    if args.update_baseline:
        if not BASELINE_PATH.exists():
            merged = current
        else:
            merged = min(current, _load_baseline(BASELINE_PATH))
        _write_baseline(BASELINE_PATH, merged)
        logger.info("baseline updated: %s=%d", _METRIC, merged)
        return 0

    baseline = _load_baseline(BASELINE_PATH)
    if current > baseline:
        sys.stderr.write(
            f"GUI thread-blocking ratchet FAILED: {_METRIC} {current} > "
            f"baseline {baseline}\n"
        )
        sys.stderr.write("Newly-flagged files:\n")
        for path in offenders:
            try:
                display = path.relative_to(REPO_ROOT)
            except ValueError:
                display = path
            sys.stderr.write(f"  {display}\n")
        sys.stderr.write(
            "\nHint: route the button's compute through "
            "src.tools.async_action.AsyncActionBar (see "
            "src/tools/simulation_backends_launcher/gui.py for the pattern), "
            "or if the file already backgrounds its work another legitimate "
            "way (e.g. QProcess), annotate it with `# noqa: gui-thread/ok` "
            "and explain why in the PR description. See issue #8880.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
