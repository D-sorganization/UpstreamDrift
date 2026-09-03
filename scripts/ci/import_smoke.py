#!/usr/bin/env python3
"""Import smoke for the always-on ``unit-core-always`` lane (#9409, RM #1507).

Imports every shared alias root declared by
``src.shared.python.import_aliases._SHARED_ROOTS`` in both spellings the
alias installer advertises -- ``src.shared.python.<root>`` and the bare
``<root>`` -- plus a handful of UpstreamDrift entry-point modules that every
launcher, API server and test session depends on. A broken alias installer,
a missing pinned Tools checkout (``vendor/ud-tools``) or a syntax error in a
core module surfaces here within seconds instead of as hundreds of collection
errors in the full test suite.

Failure policy (deliberately strict):

* A ``ModuleNotFoundError`` whose missing name IS the alias root itself means
  the root simply does not exist in this checkout (neither UpstreamDrift's
  ``src/shared/python`` nor the pinned Tools tree ships it). Such roots must
  be listed in ``ABSENT_ALIAS_ROOTS`` with the reason; unlisted absences fail.
* Any other exception (an ``ImportError`` from inside the package, a
  ``ModuleNotFoundError`` for a *dependency* of the root, a ``SyntaxError``,
  ...) fails the run with the root name and the full traceback.

Run from the repository root::

    python scripts/ci/import_smoke.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Alias roots declared in ``_SHARED_ROOTS`` that are known not to exist in
# either UpstreamDrift's ``src/shared/python`` or the pinned Tools tree
# (``vendor/ud-tools/src/shared/python``). ``import_aliases`` reserves these
# names so a later Tools release can ship them without a UD change; until it
# does, ``import <root>`` legitimately raises ``ModuleNotFoundError`` for the
# root itself. Keep each entry commented -- and remove it the moment the
# package appears, so the smoke starts covering it.
ABSENT_ALIAS_ROOTS: dict[str, str] = {
    # Empty as of #9409: every declared root resolves (locally in
    # ``src/shared/python`` or via the pinned Tools tree). Add an entry here
    # ONLY when a root is genuinely absent from both trees, e.g.
    #     "logging_pkg": "reserved alias root; not shipped by Tools@<sha>",
    # and the script fails if an entry goes stale (the root imports again).
}

# UpstreamDrift entry points that must import headlessly with only
# ``requirements.lock`` installed (no GUI extras, no optional engines).
UD_ROOTS: tuple[str, ...] = (
    "src.config.launcher_manifest_loader",
    "src.config.feature_parity_loader",
    "src.api.server",
    "src.api.local_server",
    # NOT listed: src.launchers.launcher_orchestrator imports PyQt6 at module
    # scope and PyQt6 is a GUI extra absent from requirements.lock (#9409).
    "src.shared.python.engine_core.engine_registry",
)


def _ensure_repo_on_path() -> None:
    """Mirror the pytest ``pythonpath`` entries the alias installer relies on."""
    for candidate in (
        _REPO_ROOT,
        _REPO_ROOT / "src",
        _REPO_ROOT / "src" / "shared" / "python",
        _REPO_ROOT / "vendor" / "ud-tools" / "src",
        _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python",
    ):
        text = str(candidate)
        if candidate.is_dir() and text not in sys.path:
            sys.path.append(text)


def _shared_roots() -> tuple[str, ...]:
    from src.shared.python import import_aliases

    return tuple(sorted(import_aliases._SHARED_ROOTS))


def _try_import(module_name: str, root: str) -> tuple[str, str]:
    """Return ``(status, detail)`` for one import attempt.

    ``status`` is ``"ok"``, ``"absent"`` (root itself is missing) or
    ``"fail"``; ``detail`` carries the traceback for failures.
    """
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing in {root, module_name}:
            return "absent", f"{exc.__class__.__name__}: {exc}"
        return "fail", traceback.format_exc()
    except Exception:  # noqa: BLE001 - any failure must be reported, not hidden
        return "fail", traceback.format_exc()
    return "ok", ""


def main() -> int:
    """Import every root; return 0 only when nothing unexpected happened."""
    _ensure_repo_on_path()

    try:
        shared_roots = _shared_roots()
    except Exception:  # noqa: BLE001 - the alias installer itself is under test
        print("::error::could not import src.shared.python.import_aliases")
        traceback.print_exc()
        return 1

    targets: list[tuple[str, str]] = []
    for root in shared_roots:
        targets.append((root, f"src.shared.python.{root}"))
        targets.append((root, root))
    targets.extend((name, name) for name in UD_ROOTS)

    rows: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str]] = []
    unexpected_absent: list[str] = []
    for root, module_name in targets:
        status, detail = _try_import(module_name, root)
        if status == "absent":
            if root in ABSENT_ALIAS_ROOTS:
                rows.append((module_name, "absent (allowlisted)", detail))
            else:
                rows.append((module_name, "ABSENT", detail))
                unexpected_absent.append(module_name)
        elif status == "fail":
            rows.append((module_name, "FAIL", detail.strip().splitlines()[-1]))
            failures.append((module_name, detail))
        else:
            rows.append((module_name, "ok", ""))

    width = max(len(row[0]) for row in rows)
    print(f"{'module':<{width}}  status")
    print(f"{'-' * width}  {'-' * 20}")
    for module_name, status, detail in rows:
        suffix = f"  {detail}" if detail and status != "ok" else ""
        print(f"{module_name:<{width}}  {status}{suffix}")

    stale_allowlist = sorted(
        root
        for root in ABSENT_ALIAS_ROOTS
        if root not in shared_roots
        or all(
            status != "absent (allowlisted)"
            for module_name, status, _ in rows
            if module_name in {root, f"src.shared.python.{root}"}
        )
    )

    print()
    ok_count = sum(1 for _, status, _ in rows if status == "ok")
    print(
        f"imported {ok_count}/{len(rows)} targets; "
        f"{len(ABSENT_ALIAS_ROOTS)} allowlisted absent roots; "
        f"{len(unexpected_absent)} unexpected absent; {len(failures)} failed"
    )

    exit_code = 0
    for module_name in unexpected_absent:
        print(
            f"::error::alias root '{module_name}' does not exist and is not in "
            "ABSENT_ALIAS_ROOTS (scripts/ci/import_smoke.py). Either restore the "
            "package or allowlist it with a reason."
        )
        exit_code = 1
    for module_name, detail in failures:
        print(f"::error::import of '{module_name}' failed")
        print(f"--- traceback for {module_name} ---")
        print(detail)
        exit_code = 1
    for root in stale_allowlist:
        print(
            f"::error::ABSENT_ALIAS_ROOTS entry '{root}' is stale: the root now "
            "imports (or is no longer a shared alias root). Remove it so the "
            "smoke covers the package."
        )
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
