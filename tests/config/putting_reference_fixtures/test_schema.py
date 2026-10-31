"""Schema contract for the shared putting reference-fixture file (#14141 PR 2).

``src/config/putting_reference_fixtures.json`` is loaded identically by
Python, the future standalone JS putting studio, and PyQt6 tests. These
tests ARE the schema gate: they fail when a fixture is missing its
provenance, claims a numeric target with no tolerance basis, declares
an unknown runner, duplicates another fixture's id, or names a
``maps_to`` function that does not exist on the ``putting_dynamics``
façade.

Design rule enforced here, not by the loader: a fixture asserting a
real numeric target (no ``pending_reason``) must either map to a real
façade attribute or be one of the documented physics-constants
comparisons in ``tests/unit/putting_dynamics/test_reference_fixtures.py``
-- it must not silently assert nothing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import src.shared.python.putting_dynamics as pd
from src.shared.python.putting_dynamics.reference_fixtures import (
    DEFAULT_FIXTURES_PATH,
    VALID_APPLIES_TO,
    ReferenceFixture,
    load_reference_fixtures,
)

pytestmark = pytest.mark.unit

#: fixture_id values that compare against a module other than the
#: putting_dynamics façade (kept in sync with
#: tests/unit/putting_dynamics/test_reference_fixtures.py's
#: ``_PHYSICS_CONSTANTS_FIXTURES``). A fixture here is allowed to omit
#: ``maps_to`` even without a ``pending_reason``.
_NON_FACADE_TARGETS = {"usga_ball_mass_max_kg", "usga_ball_diameter_min_m"}


@pytest.fixture(scope="module")
def fixtures() -> tuple[ReferenceFixture, ...]:
    return load_reference_fixtures()


@pytest.fixture(scope="module")
def raw_fixtures() -> dict[str, Any]:
    return json.loads(DEFAULT_FIXTURES_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "putting_reference_fixtures.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The committed fixture file
# ---------------------------------------------------------------------------


def test_fixture_file_exists() -> None:
    assert DEFAULT_FIXTURES_PATH.exists()


def test_loads_at_least_fourteen_fixtures(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    assert len(fixtures) >= 14


def test_fixture_ids_are_unique(fixtures: tuple[ReferenceFixture, ...]) -> None:
    ids = [f.fixture_id for f in fixtures]
    assert len(set(ids)) == len(ids)


def test_every_fixture_has_citation_locator_and_tolerance_basis(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    for fixture in fixtures:
        assert fixture.source.citation.strip(), fixture.fixture_id
        assert fixture.source.locator.strip(), fixture.fixture_id
        assert fixture.expected.tolerance_basis.strip(), fixture.fixture_id


def test_verified_is_a_real_bool(fixtures: tuple[ReferenceFixture, ...]) -> None:
    for fixture in fixtures:
        assert isinstance(fixture.source.verified, bool), fixture.fixture_id


def test_applies_to_is_a_valid_nonempty_subset(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    for fixture in fixtures:
        assert fixture.applies_to, fixture.fixture_id
        assert set(fixture.applies_to) <= VALID_APPLIES_TO, fixture.fixture_id


def test_fixtures_with_no_resolvable_value_carry_pending_reason(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    """A fixture with neither a value/range nor a mapped function must say why."""
    for fixture in fixtures:
        has_value = fixture.expected.value is not None
        has_range = (
            fixture.expected.min is not None and fixture.expected.max is not None
        )
        has_target = fixture.maps_to is not None or fixture.fixture_id in (
            _NON_FACADE_TARGETS
        )
        if not (has_value or has_range) or not has_target:
            assert fixture.pending_reason, (
                f"{fixture.fixture_id} has no resolvable expected value/target "
                "and no pending_reason explaining the gap"
            )


def test_maps_to_resolves_to_a_real_facade_attribute(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    for fixture in fixtures:
        if fixture.maps_to is None:
            continue
        assert hasattr(pd, fixture.maps_to), (
            f"{fixture.fixture_id}: maps_to={fixture.maps_to!r} is not a "
            "putting_dynamics facade attribute"
        )


def test_no_fixture_value_is_unsourced_boilerplate(
    fixtures: tuple[ReferenceFixture, ...],
) -> None:
    """Citations must name a real source, not placeholder text."""
    banned = {"todo", "tbd", "n/a", "unknown"}
    for fixture in fixtures:
        citation_lower = fixture.source.citation.strip().lower()
        assert citation_lower not in banned, fixture.fixture_id


# ---------------------------------------------------------------------------
# Loader contract (DbC)
# ---------------------------------------------------------------------------


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_reference_fixtures(tmp_path / "absent.json")


def test_duplicate_fixture_id_rejected(
    tmp_path: Path, raw_fixtures: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_fixtures)
    payload["fixtures"].append(copy.deepcopy(payload["fixtures"][0]))
    with pytest.raises(ValueError, match="duplicate fixture_id"):
        load_reference_fixtures(_write(tmp_path, payload))


def test_missing_citation_rejected(
    tmp_path: Path, raw_fixtures: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_fixtures)
    payload["fixtures"][0]["source"]["citation"] = ""
    with pytest.raises(ValueError, match="citation"):
        load_reference_fixtures(_write(tmp_path, payload))


def test_missing_locator_rejected(tmp_path: Path, raw_fixtures: dict[str, Any]) -> None:
    payload = copy.deepcopy(raw_fixtures)
    del payload["fixtures"][0]["source"]["locator"]
    with pytest.raises(ValueError, match="locator"):
        load_reference_fixtures(_write(tmp_path, payload))


def test_missing_tolerance_basis_rejected(
    tmp_path: Path, raw_fixtures: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_fixtures)
    del payload["fixtures"][0]["expected"]["tolerance_basis"]
    with pytest.raises(ValueError, match="tolerance_basis"):
        load_reference_fixtures(_write(tmp_path, payload))


def test_unknown_applies_to_rejected(
    tmp_path: Path, raw_fixtures: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_fixtures)
    payload["fixtures"][0]["applies_to"] = ["matlab"]
    with pytest.raises(ValueError, match="unknown runner"):
        load_reference_fixtures(_write(tmp_path, payload))


def test_non_bool_verified_rejected(
    tmp_path: Path, raw_fixtures: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_fixtures)
    payload["fixtures"][0]["source"]["verified"] = "yes"
    with pytest.raises(ValueError, match="verified"):
        load_reference_fixtures(_write(tmp_path, payload))


def test_value_required_without_pending_reason(
    tmp_path: Path, raw_fixtures: dict[str, Any]
) -> None:
    payload = copy.deepcopy(raw_fixtures)
    fixture = payload["fixtures"][0]
    assert "pending_reason" not in fixture
    fixture["expected"].pop("value", None)
    fixture["expected"].pop("min", None)
    fixture["expected"].pop("max", None)
    with pytest.raises(ValueError, match="pending_reason"):
        load_reference_fixtures(_write(tmp_path, payload))


def test_fixtures_are_immutable(fixtures: tuple[ReferenceFixture, ...]) -> None:
    with pytest.raises((AttributeError, TypeError)):
        fixtures[0].fixture_id = "mutated"  # type: ignore[misc]
