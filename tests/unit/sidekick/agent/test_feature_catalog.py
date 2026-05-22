"""Tests for sidekick.agent.feature_catalog (epic #5967 / S1 / #5970).

TDD: these tests are written first and pin the contract before the
implementation. Coverage targets:

* catalog is non-empty and deterministically ordered
* every FeatureEntry.module is importable
* lookup_feature raises KeyError with the closest matches on a miss
* search_features returns relevance-ranked entries with no LLM call
* FeatureEntry is frozen (DbC) and rejects invalid kinds
* discovery degrades gracefully when a single source raises
"""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from sidekick.agent.feature_catalog import (
    FeatureEntry,
    FeatureKind,
    build_feature_catalog,
    lookup_feature,
    search_features,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_catalog_cache():
    """Ensure each test starts with a clean catalog cache.

    The catalog memoises its result in a module-global; tests that
    monkey-patch a discovery source must not poison subsequent tests
    that expect the unpatched view. Resetting before *and* after means
    test order is irrelevant.
    """
    import sidekick.agent.feature_catalog as fc

    fc._CATALOG_CACHE = None
    try:
        yield
    finally:
        fc._CATALOG_CACHE = None


# ---------------------------------------------------------------------------
# FeatureEntry — Design by Contract
# ---------------------------------------------------------------------------


def test_feature_entry_is_frozen_and_slotted() -> None:
    entry = FeatureEntry(
        feature_id="calculator.example",
        kind="calculator",
        title="Example",
        summary="An example calculator.",
        module="sidekick.calculators.base",
        help_anchors=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.title = "Other"  # type: ignore[misc]
    assert "__dict__" not in dir(entry)  # slots=True


def test_feature_entry_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        FeatureEntry(
            feature_id="x.y",
            kind="not_a_kind",  # type: ignore[arg-type]
            title="t",
            summary="s",
            module="sidekick",
            help_anchors=(),
        )


def test_feature_entry_rejects_empty_feature_id() -> None:
    with pytest.raises(ValueError, match="feature_id"):
        FeatureEntry(
            feature_id="",
            kind="subtab",
            title="t",
            summary="s",
            module="sidekick",
            help_anchors=(),
        )


def test_feature_entry_id_must_match_kind_namespace() -> None:
    # Invariant: subtab entries must be feature_id="subtab.<name>".
    with pytest.raises(ValueError, match="namespace"):
        FeatureEntry(
            feature_id="calculator.wgs_reactor",
            kind="subtab",
            title="t",
            summary="s",
            module="sidekick",
            help_anchors=(),
        )


def test_feature_kind_enum_values() -> None:
    # Pin the closed set so downstream JSON schemas can rely on it.
    assert set(FeatureKind) == {
        FeatureKind.CALCULATOR,
        FeatureKind.PROCESS_CALCULATOR,
        FeatureKind.SUBTAB,
        FeatureKind.WORKFLOW,
        FeatureKind.THEME,
    }


# ---------------------------------------------------------------------------
# build_feature_catalog — discovery
# ---------------------------------------------------------------------------


def test_build_feature_catalog_is_non_empty() -> None:
    catalog = build_feature_catalog()
    assert catalog, "feature catalog must contain at least one entry"


def test_build_feature_catalog_is_deterministic() -> None:
    # force_refresh on both sides so we're comparing two real builds,
    # not the cached value against itself.
    a = build_feature_catalog(force_refresh=True)
    b = build_feature_catalog(force_refresh=True)
    assert list(a.keys()) == list(b.keys())


def test_build_feature_catalog_keys_match_feature_ids() -> None:
    catalog = build_feature_catalog()
    for key, entry in catalog.items():
        assert key == entry.feature_id, f"key {key!r} != entry.feature_id"


def test_build_feature_catalog_includes_subtabs() -> None:
    catalog = build_feature_catalog()
    subtab_ids = [fid for fid in catalog if fid.startswith("subtab.")]
    # DEFAULT_SIDEBAR_TAB_HELP exposes ~13 tabs; we expect at least 5.
    assert len(subtab_ids) >= 5, f"too few subtab entries: {subtab_ids}"


def test_build_feature_catalog_includes_known_subtab() -> None:
    catalog = build_feature_catalog()
    assert "subtab.calculator" in catalog
    entry = catalog["subtab.calculator"]
    assert entry.kind == "subtab"
    assert entry.title  # non-empty
    assert entry.summary  # non-empty


def test_build_feature_catalog_modules_are_importable() -> None:
    """Hygiene invariant: every advertised module path is real."""
    catalog = build_feature_catalog()
    failures = []
    for entry in catalog.values():
        try:
            importlib.import_module(entry.module)
        except Exception as exc:  # noqa: BLE001 - test isolating real importability
            failures.append(f"{entry.feature_id}: {entry.module} -> {exc!r}")
    assert not failures, "broken module references:\n  " + "\n  ".join(failures)


def test_build_feature_catalog_cached_call_is_fast() -> None:
    """Cache hit must be effectively free.

    The first call may take seconds on a fresh interpreter (pulls in
    pandas/scipy/matplotlib transitively through the calculator and
    process-calculator walks). After that, the cache must short-circuit
    so chat-turn-time callers never re-pay the import cost. We do not
    measure the cold build here — that's CI-runner-speed-dependent and
    out of scope for a unit test.
    """
    import time

    # Prime the cache.
    build_feature_catalog()
    start = time.perf_counter()
    build_feature_catalog()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 50.0, f"cached catalog call took {elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# lookup_feature — DbC
# ---------------------------------------------------------------------------


def test_lookup_feature_returns_matching_entry() -> None:
    catalog = build_feature_catalog()
    feature_id = next(iter(catalog))
    entry = lookup_feature(feature_id)
    assert entry.feature_id == feature_id


def test_lookup_feature_unknown_raises_key_error_with_suggestions() -> None:
    with pytest.raises(KeyError) as excinfo:
        lookup_feature("subtab.calculatr")  # typo
    msg = str(excinfo.value)
    # Suggestions must mention at least one real subtab.
    assert "calculator" in msg


def test_lookup_feature_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match="feature_id"):
        lookup_feature("")


# ---------------------------------------------------------------------------
# search_features — relevance ranking
# ---------------------------------------------------------------------------


def test_search_features_returns_empty_for_no_matches() -> None:
    assert search_features("zzzzzz_no_such_thing_zzzzzz") == ()


def test_search_features_respects_limit() -> None:
    results = search_features("calculator", limit=3)
    assert len(results) <= 3


def test_search_features_query_match_ranks_above_unrelated() -> None:
    """A query token that appears in title/summary outranks ones that do not."""
    results = search_features("calculator", limit=10)
    assert results, "expected at least one match for 'calculator'"
    # All returned entries should mention the token somewhere.
    for entry in results:
        haystack = f"{entry.feature_id} {entry.title} {entry.summary}".lower()
        assert "calculator" in haystack or "calc" in haystack


def test_search_features_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query"):
        search_features("   ")


def test_search_features_returns_tuple_not_list() -> None:
    # Immutability at the boundary (DbC postcondition).
    results = search_features("calculator", limit=1)
    assert isinstance(results, tuple)


# ---------------------------------------------------------------------------
# graceful degradation
# ---------------------------------------------------------------------------


def test_catalog_skips_broken_help_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the help-source parser returns nothing, build still succeeds.

    We patch the catalog's private parser rather than the underlying
    file — that parser is the seam the catalog uses precisely so it can
    degrade when the heavy ``tools_sidebar`` package is unavailable in a
    headless / partial install.
    """
    import sidekick.agent.feature_catalog as fc

    monkeypatch.setattr(fc, "_parse_help_content", dict)
    # force_refresh defeats the cached catalog so the monkeypatch
    # actually takes effect.
    catalog = build_feature_catalog(force_refresh=True)
    # Catalog still has theme/calculator entries even with no subtab help.
    assert catalog
    assert not any(fid.startswith("subtab.") for fid in catalog)
