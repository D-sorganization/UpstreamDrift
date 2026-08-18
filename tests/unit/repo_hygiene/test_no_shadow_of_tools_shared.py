"""Forbid UD-side modules from shadowing Tools-shared modules without a ledger entry.

See UpstreamDrift issue #5623. Tools is the source of truth for shared modules
under ``src/shared/python/``. UD consumes them via the ``vendor/ud-tools``
submodule. A UD-side module of the same name silently shadows the vendor copy
and causes work to be lost on the next vendor bump.

This test enumerates every top-level entry under
``vendor/ud-tools/src/shared/python/`` and asserts that no same-named entry
exists under ``src/shared/python/`` unless it is explicitly listed in
``scripts/config/shadow_modules.yaml``.

Design-by-contract:

- The ledger is a **ratchet**. Entries may disappear (debt paid down); a new
  shadow that is not in the ledger fails this test.
- Every ledger entry MUST carry a positive-int ``tracking_issue`` and a
  parseable ISO ``sunset_date`` that has not yet passed. Silent ledger rot is
  the primary failure mode this guard exists to prevent.
- A missing vendor tree **fails closed in CI**. It only degrades to a skip on a
  developer machine that has not run ``git submodule update --init``. This
  mirrors ``test_tools_child_copy_contract.py`` and exists because the previous
  unconditional ``pytest.skip`` made this guard vacuous on every CI run.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# Repo root is three parents up: tests/unit/repo_hygiene/<this file>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_SHARED = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
_UD_SHARED = _REPO_ROOT / "src" / "shared" / "python"
_LEDGER_PATH = _REPO_ROOT / "scripts" / "config" / "shadow_modules.yaml"

# Names that are not modules and must be ignored on either side.
_IGNORED_NAMES = frozenset(
    {
        "__init__.py",
        "__pycache__",
        "README.md",
        "README_PACKAGE.md",
        "tests",
    }
)

_MISSING_VENDOR_HINT = (
    f"The vendored Tools tree is missing at {_VENDOR_SHARED}. "
    "Run `git submodule update --init vendor/ud-tools` to materialise it. "
    "In CI this is a hard failure: the shadow guard must never pass "
    "vacuously (see UpstreamDrift issue #5623)."
)


def _module_names(root: Path) -> set[str]:
    """Return top-level module names (packages and ``.py`` modules) under *root*.

    Hidden/dunder entries and the items in :data:`_IGNORED_NAMES` are skipped.
    """
    if not root.is_dir():
        return set()
    names: set[str] = set()
    for entry in root.iterdir():
        if entry.name.startswith((".", "_")):
            continue
        if entry.name in _IGNORED_NAMES:
            continue
        if entry.is_dir() or (entry.is_file() and entry.suffix == ".py"):
            names.add(entry.name)
    return names


def _require_vendor_shared() -> Path:
    """Return the vendored Tools shared tree, or fail closed in CI.

    A guard that silently skips is worse than no guard, because everyone
    believes it ran. CI must never reach the skip branch.
    """
    if _VENDOR_SHARED.is_dir():
        return _VENDOR_SHARED
    if os.environ.get("CI"):
        raise AssertionError(_MISSING_VENDOR_HINT)
    pytest.skip(_MISSING_VENDOR_HINT)


def _load_ledger(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load and validate the shadow-module ledger.

    Returns an empty dict when the file declares ``shadows: {}``. Raises
    :class:`AssertionError` on malformed or expired entries.
    """
    ledger_path = _LEDGER_PATH if path is None else path
    assert ledger_path.is_file(), (
        f"Shadow ledger missing at {ledger_path}. See UpstreamDrift issue #5623."
    )
    raw = yaml.safe_load(ledger_path.read_text(encoding="utf-8")) or {}
    shadows = raw.get("shadows") or {}
    assert isinstance(shadows, dict), (
        f"shadow_modules.yaml: 'shadows' must be a mapping, got {type(shadows)!r}"
    )

    for name, entry in shadows.items():
        assert isinstance(entry, dict), (
            f"shadow_modules.yaml entry {name!r}: must be a mapping, "
            f"got {type(entry)!r}"
        )
        issue = entry.get("tracking_issue")
        sunset = entry.get("sunset_date")
        assert isinstance(issue, int) and issue > 0, (
            f"shadow_modules.yaml entry {name!r}: 'tracking_issue' must be a "
            f"positive int, got {issue!r}"
        )
        assert isinstance(sunset, str) and sunset, (
            f"shadow_modules.yaml entry {name!r}: 'sunset_date' must be a "
            f"non-empty ISO date string, got {sunset!r}"
        )
        # date.fromisoformat raises ValueError on bad input — surface as
        # AssertionError so the failure mode is uniform.
        try:
            parsed_sunset = date.fromisoformat(sunset)
        except ValueError as exc:
            raise AssertionError(
                f"shadow_modules.yaml entry {name!r}: 'sunset_date' "
                f"{sunset!r} is not a valid ISO date — {exc}"
            ) from exc
        # Enforce that the sunset date has not yet passed. Once it does the
        # entry must be removed (shadow resolved) or renegotiated with a new
        # tracking issue. Fixes UD issue #5627.
        assert parsed_sunset >= date.today(), (
            f"shadow_modules.yaml entry {name!r}: 'sunset_date' {sunset!r} "
            f"has passed (today is {date.today().isoformat()}). The shadow "
            f"must be resolved — remove the UD-side copy, or update the entry "
            f"with a new sunset_date and tracking_issue."
        )
    return shadows


def _unapproved_shadows(
    vendor_names: set[str],
    ud_names: set[str],
    allowed: set[str],
) -> list[str]:
    """Return shadowed module names that are absent from the ledger."""
    return sorted((vendor_names & ud_names) - allowed)


# ── ledger well-formedness ───────────────────────────────────────────────────


def test_ledger_is_well_formed() -> None:
    """DbC: every ledger entry has a tracking issue and an unexpired sunset date."""
    # Calling the loader exercises every assertion above.
    _load_ledger()


def test_load_ledger_rejects_expired_sunset_date(tmp_path: Path) -> None:
    """Regression UD#5627: an expired sunset_date must fail, not pass silently.

    The policy comment in shadow_modules.yaml says the guard fails once a
    sunset date passes. Before UD#5627 the date was parsed but never compared
    to today(), so expired entries silently passed.
    """
    ledger = tmp_path / "shadow_modules.yaml"
    ledger.write_text(
        "shadows:\n  some_module:\n    tracking_issue: 9999\n"
        '    sunset_date: "2000-01-01"\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="sunset_date.*has passed"):
        _load_ledger(ledger)


def test_load_ledger_rejects_entry_without_tracking_issue(tmp_path: Path) -> None:
    """A bare module name with no justification must not be accepted.

    This is the shape the ledger was reduced to by #8322 (`allowed_shadows: []`
    with no per-entry metadata). Rejecting it keeps the ledger a shrinking,
    accountable record rather than a permanent amnesty list.
    """
    ledger = tmp_path / "shadow_modules.yaml"
    ledger.write_text(
        'shadows:\n  some_module:\n    sunset_date: "2099-01-01"\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="tracking_issue"):
        _load_ledger(ledger)


# ── the guard itself ─────────────────────────────────────────────────────────


def test_no_unapproved_shadows_of_tools_shared() -> None:
    """Fail if any UD module shadows a Tools-shared module without a ledger entry.

    A "shadow" is a top-level entry (package directory or ``.py`` module) that
    exists under both ``vendor/ud-tools/src/shared/python/`` and
    ``src/shared/python/`` with the same name.
    """
    vendor_shared = _require_vendor_shared()

    unapproved = _unapproved_shadows(
        _module_names(vendor_shared),
        _module_names(_UD_SHARED),
        set(_load_ledger()),
    )

    assert not unapproved, (
        "Unapproved shadow modules detected under src/shared/python/ — these "
        "duplicate Tools-shared modules and will lose work on the next vendor "
        "bump. Either remove the UD-side copy and import from Tools via the "
        "vendor tree, or add an entry to scripts/config/shadow_modules.yaml "
        "with a tracking issue and sunset date. See UpstreamDrift issue "
        f"#5623.\n\nUnapproved shadows: {unapproved}"
    )


# ── no-growth ratchet behaviour ──────────────────────────────────────────────


def test_new_shadow_is_rejected_even_though_baseline_is_large() -> None:
    """A newly introduced shadow must fail even with 32 grandfathered entries.

    This is the property that makes the ledger a ratchet rather than an
    amnesty: the size of the baseline must not weaken detection of the next
    regression. Regression cover for #8322, which added four shadows while the
    guard was disabled.
    """
    allowed = set(_load_ledger())
    vendor_names = allowed | {"brand_new_module"}
    ud_names = allowed | {"brand_new_module"}

    assert _unapproved_shadows(vendor_names, ud_names, allowed) == ["brand_new_module"]


def test_removing_a_shadow_is_always_allowed() -> None:
    """The ratchet must never block paying debt down."""
    allowed = set(_load_ledger())
    assert _unapproved_shadows(set(), set(), allowed) == []


def test_ledger_has_no_entries_for_modules_that_are_not_shadowed() -> None:
    """Stale ledger entries must be pruned so the count reflects real debt.

    Without this the ledger can only grow: a module removed from Tools (or
    from UD) would leave a permanent unjustified line behind, and the "debt
    should only go down" claim in the file header would quietly become false.
    """
    vendor_shared = _require_vendor_shared()

    actual_shadows = _module_names(vendor_shared) & _module_names(_UD_SHARED)
    stale = sorted(set(_load_ledger()) - actual_shadows)

    assert not stale, (
        "scripts/config/shadow_modules.yaml lists modules that are no longer "
        "shadowed. Delete these lines — the ledger must shrink as debt is "
        f"paid down.\n\nStale entries: {stale}"
    )
