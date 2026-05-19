"""Forbid UD-side modules from shadowing Tools-shared modules without an allow-list entry.

See UpstreamDrift issue #5623. Tools is the source of truth for shared
modules under ``shared/python/``. UD consumes them via the
``vendor/ud-tools`` submodule. A UD-side module of the same name silently
shadows the vendor copy and causes work to be lost on the next vendor bump.

This test enumerates every top-level entry under
``vendor/ud-tools/src/shared/python/`` and asserts that no same-named
entry exists under ``src/shared/python/`` unless it is explicitly listed
in ``scripts/config/shadow_modules.yaml``.

Design-by-contract:

- Every allow-list entry MUST carry a non-empty integer ``tracking_issue``
  and a parseable ISO ``sunset_date``. Missing or malformed metadata fails
  the test loudly — silent allow-list rot is the primary failure mode this
  test exists to prevent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

# Repo root is three parents up: tests/unit/repo_hygiene/<this file>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_SHARED = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
_UD_SHARED = _REPO_ROOT / "src" / "shared" / "python"
_ALLOW_LIST_PATH = _REPO_ROOT / "scripts" / "config" / "shadow_modules.yaml"

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


def _module_names(root: Path) -> set[str]:
    """Return top-level module names (packages and .py modules) under ``root``.

    Hidden entries and the items in :data:`_IGNORED_NAMES` are skipped.
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


def _load_allow_list() -> dict[str, dict[str, Any]]:
    """Load and validate the shadow-module allow-list.

    Returns an empty dict when the file declares ``shadows: {}`` or is
    otherwise empty. Raises :class:`AssertionError` on malformed entries.
    """
    assert _ALLOW_LIST_PATH.is_file(), (
        f"Allow-list config missing at {_ALLOW_LIST_PATH}. "
        "See UpstreamDrift issue #5623."
    )
    raw = yaml.safe_load(_ALLOW_LIST_PATH.read_text(encoding="utf-8")) or {}
    shadows = raw.get("shadows") or {}
    assert isinstance(shadows, dict), (
        f"shadow_modules.yaml: 'shadows' must be a mapping, got {type(shadows)!r}"
    )

    for name, entry in shadows.items():
        assert isinstance(entry, dict), (
            f"shadow_modules.yaml entry {name!r}: must be a mapping, got {type(entry)!r}"
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
        # allow-list entry must be removed (shadow resolved) or the sunset date
        # extended with a new tracking issue. Fixes UD issue #5627.
        assert parsed_sunset >= date.today(), (
            f"shadow_modules.yaml entry {name!r}: 'sunset_date' {sunset!r} "
            f"has passed (today is {date.today().isoformat()}). The shadow "
            f"must be resolved — remove the UD-side copy or update the "
            f"entry with a new sunset_date and tracking_issue."
        )
    return shadows


def test_load_allow_list_rejects_expired_sunset_date() -> None:
    """Regression UD#5627: _load_allow_list must fail on an expired sunset_date.

    The policy comment in shadow_modules.yaml says "Once the sunset date
    passes, the allow-list test fails." Before the fix the date was parsed
    but never compared to today(), so expired entries silently passed.
    """
    import textwrap
    import tempfile
    import importlib
    import sys

    expired_yaml = textwrap.dedent(
        """\
        shadows:
          some_module:
            tracking_issue: 9999
            sunset_date: "2000-01-01"
        """
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(expired_yaml)
        tmp_path = Path(tmp.name)

    try:
        # Patch the module-level path constant and call the loader directly.
        import tests.unit.repo_hygiene.test_no_shadow_of_tools_shared as _mod

        original = _mod._ALLOW_LIST_PATH
        _mod._ALLOW_LIST_PATH = tmp_path
        try:
            with pytest.raises(AssertionError, match="sunset_date.*has passed"):
                _mod._load_allow_list()
        finally:
            _mod._ALLOW_LIST_PATH = original
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.unit
def test_allow_list_is_well_formed() -> None:
    """DbC: every allow-list entry has a tracking issue and a parseable sunset date."""
    # Just calling the loader exercises every assertion above.
    _load_allow_list()


@pytest.mark.unit
def test_no_unapproved_shadows_of_tools_shared() -> None:
    """Fail if any UD module shadows a Tools-shared module without approval.

    A "shadow" is a top-level entry (package directory or ``.py`` module)
    that exists under both ``vendor/ud-tools/src/shared/python/`` and
    ``src/shared/python/`` with the same name.
    """
    assert _VENDOR_SHARED.is_dir(), (
        f"Expected vendored Tools tree at {_VENDOR_SHARED}. "
        "Is the submodule initialised? Run `git submodule update --init`."
    )

    vendor_names = _module_names(_VENDOR_SHARED)
    ud_names = _module_names(_UD_SHARED)
    allowed = set(_load_allow_list().keys())

    shadows = vendor_names & ud_names
    unapproved = sorted(shadows - allowed)

    assert not unapproved, (
        "Unapproved shadow modules detected under src/shared/python/ — these "
        "duplicate Tools-shared modules and will lose work on the next vendor "
        "bump. Either remove the UD-side copy and import from "
        "shared.python.<name> via the vendor tree, or add an entry to "
        f"scripts/config/shadow_modules.yaml with a tracking issue and sunset "
        f"date. See UpstreamDrift issue #5623.\n\nUnapproved shadows: {unapproved}"
    )
