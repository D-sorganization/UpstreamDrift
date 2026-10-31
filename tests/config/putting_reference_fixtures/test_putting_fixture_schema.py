"""Schema contract for the shared putting reference fixtures (#14141, step 2).

``src/config/putting_reference_fixtures.json`` is shared by the Python,
standalone HTML studio and PyQt6 tests. These tests are its schema gate:
every fixture must carry provenance and a tolerance basis, ids must be
unique, ``maps_to`` targets must resolve, and a fixture without an
external target or public mapping must say why (``pending_reason``).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.putting_dynamics import (
    DEFAULT_FIXTURES_PATH,
    ReferenceFixture,
    load_reference_fixtures,
)

pytestmark = pytest.mark.unit

_PLACEHOLDERS = {"todo", "tbd", "n/a", "unknown", "none"}


@pytest.fixture(scope="module")
def fixtures() -> tuple[ReferenceFixture, ...]:
    return load_reference_fixtures()


def test_committed_file_loads_with_unique_ids(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    ids = [fixture.fixture_id for fixture in fixtures]
    assert ids and len(ids) == len(set(ids))


def test_provenance_is_real_text_not_placeholders(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    for fixture in fixtures:
        for text in (fixture.source.citation, fixture.source.locator):
            assert text.strip().lower() not in _PLACEHOLDERS, fixture.fixture_id
        assert len(fixture.expected.tolerance_basis) >= 40, fixture.fixture_id


def test_maps_to_targets_resolve(fixtures: tuple[ReferenceFixture, ...]) -> None:
    for fixture in fixtures:
        if fixture.maps_to is not None:
            assert fixture.resolve_target() is not None, fixture.fixture_id


def test_unmapped_or_targetless_fixtures_are_pending(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    for fixture in fixtures:
        if fixture.maps_to is None or not fixture.expected.has_target:
            assert fixture.pending_reason, fixture.fixture_id


def test_fixtures_are_immutable(fixtures: tuple[ReferenceFixture, ...]) -> None:
    with pytest.raises(AttributeError):
        fixtures[0].fixture_id = "mutated"  # type: ignore[misc]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_reference_fixtures(tmp_path / "absent.json")


def _drop_value(fixture: dict[str, Any]) -> None:
    fixture.pop("pending_reason", None)
    for key in ("value", "min", "max"):
        fixture["expected"].pop(key, None)


_MUTATIONS: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "empty_citation": (lambda f: f["source"].update(citation=""), "citation"),
    "missing_locator": (lambda f: f["source"].pop("locator"), "locator"),
    "missing_tolerance_basis": (
        lambda f: f["expected"].pop("tolerance_basis"),
        "tolerance_basis",
    ),
    "unknown_runner": (lambda f: f.update(applies_to=["matlab"]), "unknown runner"),
    "string_verified": (lambda f: f["source"].update(verified="yes"), "verified"),
    "no_value_not_pending": (_drop_value, "pending_reason"),
}


@pytest.mark.parametrize("name", sorted(_MUTATIONS))
def test_schema_violations_are_rejected(name: str, tmp_path: Path) -> None:
    mutate, message = _MUTATIONS[name]
    payload = json.loads(DEFAULT_FIXTURES_PATH.read_text(encoding="utf-8"))
    mutate(payload["fixtures"][0])
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_reference_fixtures(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_FIXTURES_PATH.read_text(encoding="utf-8"))
    payload["fixtures"].append(copy.deepcopy(payload["fixtures"][0]))
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate fixture_id"):
        load_reference_fixtures(path)
