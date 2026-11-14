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
- Every ledger entry MUST carry a positive-int ``tracking_issue`` and either
  a parseable ISO ``sunset_date`` that has not yet passed OR a verified
  ``adr_ruling`` path under ``docs/adr/`` (never both). Silent ledger rot is
  the primary failure mode this guard exists to prevent.
- A missing vendor tree **fails closed in CI**. It only degrades to a skip on a
  developer machine that has not run ``git submodule update --init``. This
  mirrors ``test_tools_child_copy_contract.py`` and exists because the previous
  unconditional ``pytest.skip`` made this guard vacuous on every CI run.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from tests.helpers.seam_guards import require_vendor_path

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# Repo root is three parents up: tests/unit/repo_hygiene/<this file>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_SHARED = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
_UD_SHARED = _REPO_ROOT / "src" / "shared" / "python"
_LEDGER_PATH = _REPO_ROOT / "scripts" / "config" / "shadow_modules.yaml"
_SEAM_RULINGS = _REPO_ROOT / "docs" / "shared_tools" / "seam_rulings.v1.json"

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
    """Return the vendored Tools shared tree, or fail closed.

    A guard that silently skips is worse than no guard, because everyone
    believes it ran (see issue #9501).
    """
    return require_vendor_path(_VENDOR_SHARED)


def _validate_adr_ruling(name: str, adr_ruling: Any, repo_root: Path) -> None:
    """Validate an ``adr_ruling`` field referencing an existing ADR under ``docs/adr/``."""
    assert isinstance(adr_ruling, str) and adr_ruling, (
        f"shadow_modules.yaml entry {name!r}: 'adr_ruling' must be a non-empty "
        f"relative POSIX path under 'docs/adr/', got {adr_ruling!r}"
    )
    rel = PurePosixPath(adr_ruling)
    assert (
        not rel.is_absolute()
        and adr_ruling == rel.as_posix()
        and ".." not in rel.parts
        and len(rel.parts) == 3
        and rel.parts[:2] == ("docs", "adr")
        and rel.suffix == ".md"
    ), (
        f"shadow_modules.yaml entry {name!r}: 'adr_ruling' must be a canonical "
        f"POSIX path 'docs/adr/<file>.md', got {adr_ruling!r}"
    )
    adr_path = repo_root / rel
    assert adr_path.is_file(), (
        f"shadow_modules.yaml entry {name!r}: 'adr_ruling' target {adr_ruling!r} "
        f"does not exist at {adr_path}"
    )


def _load_ledger(
    path: Path | None = None, *, repo_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Load and validate the shadow-module ledger.

    Returns an empty dict when the file declares ``shadows: {}``. Raises
    :class:`AssertionError` on malformed, expired, or unbacked entries.
    """
    ledger_path = _LEDGER_PATH if path is None else path
    effective_repo_root = _REPO_ROOT if repo_root is None else repo_root
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
        assert isinstance(issue, int) and issue > 0, (
            f"shadow_modules.yaml entry {name!r}: 'tracking_issue' must be a "
            f"positive int, got {issue!r}"
        )
        has_sunset = "sunset_date" in entry
        has_adr = "adr_ruling" in entry
        assert has_sunset != has_adr, (
            f"shadow_modules.yaml entry {name!r}: must declare exclusively one "
            "of 'sunset_date' or 'adr_ruling'"
        )
        if has_adr:
            _validate_adr_ruling(name, entry.get("adr_ruling"), effective_repo_root)
            continue

        sunset = entry.get("sunset_date")
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
        # tracking issue. Compare against UTC date so UTC midnight rollovers
        # are caught identically on local developer machines and UTC CI runners
        # (regression #14928). Fixes UD issue #5627.
        current_utc_date = datetime.now(timezone.utc).date()
        assert parsed_sunset >= current_utc_date, (
            f"shadow_modules.yaml entry {name!r}: 'sunset_date' {sunset!r} "
            f"has passed (today is {current_utc_date.isoformat()} UTC). The shadow "
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
    return sorted(
        name
        for name in (vendor_names & ud_names) - allowed
        if not _is_reviewed_split_extension(name)
    )


def _is_reviewed_split_extension(name: str) -> bool:
    """Accept only a cleaned namespace split's exact reviewed UD-only files."""
    if not _SEAM_RULINGS.is_file():
        return False
    rulings = json.loads(_SEAM_RULINGS.read_text(encoding="utf-8")).get("rulings", {})
    ruling = rulings.get(name, {})
    if ruling.get("ruling") != "split" or ruling.get("status") != "cleaned":
        return False
    declared = ruling.get("ud_only_paths")
    if not isinstance(declared, list) or not declared:
        return False
    reviewed: set[str] = set()
    for raw in declared:
        if not isinstance(raw, str):
            return False
        relative = PurePosixPath(raw)
        if (
            not raw
            or relative.is_absolute()
            or raw != relative.as_posix()
            or ".." in relative.parts
            or relative.suffix != ".py"
            or relative.name == "__init__.py"
        ):
            return False
        reviewed.add(raw)
    if len(reviewed) != len(declared):
        return False
    local = _UD_SHARED / name
    vendor = _VENDOR_SHARED / name
    if (
        not local.is_dir()
        or not vendor.is_dir()
        or local.is_symlink()
        or vendor.is_symlink()
    ):
        return False
    if any(path.is_symlink() for path in local.rglob("*")):
        return False
    local_python = {path.relative_to(local).as_posix() for path in local.rglob("*.py")}
    if local_python != reviewed:
        return False
    if any(PurePosixPath(path).name == "__init__.py" for path in local_python):
        return False
    provider_names = {
        path.relative_to(vendor).as_posix()
        for path in vendor.rglob("*")
        if path.is_dir()
    }
    provider_names.update(
        (path.parent if path.name == "__init__.py" else path.with_suffix(""))
        .relative_to(vendor)
        .as_posix()
        for path in vendor.rglob("*.py")
    )
    local_names = {
        PurePosixPath(path).with_suffix("").as_posix() for path in local_python
    }
    return not bool(local_names & provider_names)


# ── ledger well-formedness ───────────────────────────────────────────────────


def test_ledger_is_well_formed() -> None:
    """DbC: every ledger entry has a tracking issue and an unexpired sunset date."""
    # Calling the loader exercises every assertion above.
    _load_ledger()


def test_realtime_shadow_ledger_is_governed_by_adr_split_ruling() -> None:
    """UD #12046: realtime carries an ADR-0061 split ruling without a sunset date."""
    entry = _load_ledger()["realtime"]
    assert entry["tracking_issue"] == 12046
    assert "sunset_date" not in entry
    assert entry["adr_ruling"] == "docs/adr/0061-realtime-provider-split-ruling.md"
    assert (_REPO_ROOT / entry["adr_ruling"]).is_file()


def test_load_ledger_succeeds_against_utc_and_local_dates() -> None:
    """Regression #14928: _load_ledger() validates sunset dates and ADR rulings."""
    shadows = _load_ledger()
    today_utc = datetime.now(timezone.utc).date()
    today_local = date.today()
    for name, entry in shadows.items():
        if "adr_ruling" in entry:
            assert "sunset_date" not in entry
            assert (_REPO_ROOT / entry["adr_ruling"]).is_file()
            continue
        parsed_sunset = date.fromisoformat(entry["sunset_date"])
        assert parsed_sunset >= today_utc, (
            f"shadow_modules.yaml entry {name!r}: 'sunset_date' "
            f"{entry['sunset_date']!r} expired in UTC ({parsed_sunset} < {today_utc})"
        )
        assert parsed_sunset >= today_local, (
            f"shadow_modules.yaml entry {name!r}: 'sunset_date' "
            f"{entry['sunset_date']!r} expired locally ({parsed_sunset} < {today_local})"
        )


def test_load_ledger_accepts_valid_adr_ruling_without_sunset_date(
    tmp_path: Path,
) -> None:
    """An entry backed by an existing docs/adr/*.md file passes without sunset_date."""
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    (adr_dir / "0099-split-ruling.md").write_text("# ADR-0099\n", encoding="utf-8")
    ledger = tmp_path / "shadow_modules.yaml"
    ledger.write_text(
        "shadows:\n"
        "  some_module:\n"
        "    tracking_issue: 12046\n"
        '    adr_ruling: "docs/adr/0099-split-ruling.md"\n',
        encoding="utf-8",
    )

    loaded = _load_ledger(ledger, repo_root=tmp_path)
    assert loaded["some_module"]["adr_ruling"] == "docs/adr/0099-split-ruling.md"


def test_load_ledger_rejects_adr_ruling_missing_file_or_outside_docs_adr(
    tmp_path: Path,
) -> None:
    """Invalid or nonexistent adr_ruling paths must fail closed."""
    outside = tmp_path / "notes.md"
    outside.write_text("# Not an ADR\n", encoding="utf-8")
    ledger = tmp_path / "shadow_modules.yaml"

    ledger.write_text(
        "shadows:\n"
        "  some_module:\n"
        "    tracking_issue: 12046\n"
        '    adr_ruling: "docs/adr/9999-missing.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="does not exist"):
        _load_ledger(ledger, repo_root=tmp_path)

    ledger.write_text(
        "shadows:\n"
        "  some_module:\n"
        "    tracking_issue: 12046\n"
        '    adr_ruling: "notes.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="docs/adr/"):
        _load_ledger(ledger, repo_root=tmp_path)


def test_load_ledger_rejects_entry_with_both_sunset_date_and_adr_ruling(
    tmp_path: Path,
) -> None:
    """An entry must not combine a temporary sunset_date with an adr_ruling."""
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    (adr_dir / "0099-split-ruling.md").write_text("# ADR-0099\n", encoding="utf-8")
    ledger = tmp_path / "shadow_modules.yaml"
    ledger.write_text(
        "shadows:\n"
        "  some_module:\n"
        "    tracking_issue: 12046\n"
        '    sunset_date: "2099-01-01"\n'
        '    adr_ruling: "docs/adr/0099-split-ruling.md"\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="exclusively one"):
        _load_ledger(ledger, repo_root=tmp_path)


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


def test_reviewed_disjoint_split_extension_is_not_a_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleaned split retains only its declared UD-only namespace extension."""
    vendor = tmp_path / "vendor"
    ud = tmp_path / "ud"
    (vendor / "signal_toolkit").mkdir(parents=True)
    (ud / "signal_toolkit").mkdir(parents=True)
    (vendor / "signal_toolkit" / "core.py").write_text("class Signal: pass\n")
    (ud / "signal_toolkit" / "signal_processing.py").write_text("PASS = True\n")
    rulings = tmp_path / "rulings.json"
    rulings.write_text(
        json.dumps(
            {
                "rulings": {
                    "signal_toolkit": {
                        "ruling": "split",
                        "status": "cleaned",
                        "ud_only_paths": ["signal_processing.py"],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(sys.modules[__name__], "_VENDOR_SHARED", vendor)
    monkeypatch.setattr(sys.modules[__name__], "_UD_SHARED", ud)
    monkeypatch.setattr(sys.modules[__name__], "_SEAM_RULINGS", rulings)
    assert _unapproved_shadows({"signal_toolkit"}, {"signal_toolkit"}, set()) == []


@pytest.mark.parametrize(
    ("extra_local", "extra_vendor"),
    [
        ("__init__.py", None),
        ("core.py", None),
        ("new_helper.py", None),
        ("remove-signal", None),
        (None, "signal_processing.py"),
        (None, "signal_processing/__init__.py"),
    ],
)
def test_split_extension_rejects_init_new_or_colliding_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_local: str | None,
    extra_vendor: str | None,
) -> None:
    vendor = tmp_path / "vendor"
    ud = tmp_path / "ud"
    (vendor / "signal_toolkit").mkdir(parents=True)
    (ud / "signal_toolkit").mkdir(parents=True)
    (vendor / "signal_toolkit" / "core.py").write_text("class Signal: pass\n")
    (ud / "signal_toolkit" / "signal_processing.py").write_text("PASS = True\n")
    if extra_local == "remove-signal":
        (ud / "signal_toolkit" / "signal_processing.py").unlink()
    elif extra_local is not None:
        path = ud / "signal_toolkit" / extra_local
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PASS = False\n")
    if extra_vendor is not None:
        path = vendor / "signal_toolkit" / extra_vendor
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PASS = False\n")
    rulings = tmp_path / "rulings.json"
    rulings.write_text(
        json.dumps(
            {
                "rulings": {
                    "signal_toolkit": {
                        "ruling": "split",
                        "status": "cleaned",
                        "ud_only_paths": ["signal_processing.py"],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(sys.modules[__name__], "_VENDOR_SHARED", vendor)
    monkeypatch.setattr(sys.modules[__name__], "_UD_SHARED", ud)
    monkeypatch.setattr(sys.modules[__name__], "_SEAM_RULINGS", rulings)
    assert _unapproved_shadows({"signal_toolkit"}, {"signal_toolkit"}, set()) == [
        "signal_toolkit"
    ]


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
