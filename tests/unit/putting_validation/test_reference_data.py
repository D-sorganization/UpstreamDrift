"""Provenance completeness for the putting reference registry (#14176)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.putting_validation.reference_data import (
    AVAILABLE_SERIES,
    UNAVAILABLE_SERIES,
    License,
    UnavailableSeries,
    get_series,
    require_validation_grade,
)

pytestmark = pytest.mark.unit
ANCHORS = AVAILABLE_SERIES["sg_optimizer_documented_anchors"]


def test_every_series_has_complete_provenance_and_matching_key() -> None:
    assert not set(AVAILABLE_SERIES) & set(UNAVAILABLE_SERIES)
    pairs: list[tuple[str, Any]] = [
        *AVAILABLE_SERIES.items(),
        *UNAVAILABLE_SERIES.items(),
    ]
    for key, series in pairs:
        prov = series.provenance
        assert series.series_id == key
        assert prov.citation.strip() and prov.table.strip()
        assert prov.distance_units == "feet"
        assert prov.population in {"tour", "amateur"}
        assert isinstance(prov.license, License)


def test_unavailable_series_state_a_reason_and_cover_wanted_sources() -> None:
    for series in UNAVAILABLE_SERIES.values():
        assert len(series.reason.strip()) > 20
    provs = [s.provenance for s in UNAVAILABLE_SERIES.values()]
    assert {p.population for p in provs} == {"tour", "amateur"}
    assert {p.license for p in provs} == {License.PUBLIC, License.LICENSED}
    assert {"scratch", "hcp10", "hcp20"} <= {p.handicap_band for p in provs}


def test_documented_anchors_are_not_validation_grade() -> None:
    assert get_series("sg_optimizer_documented_anchors") is ANCHORS
    with pytest.raises(ContractViolationError):
        require_validation_grade(ANCHORS)
    require_validation_grade(replace(ANCHORS, validation_grade=True))


def test_get_series_refuses_unavailable_and_unknown() -> None:
    with pytest.raises(ContractViolationError, match="unavailable"):
        get_series("broadie_2014_tour_make_pct")
    with pytest.raises(ContractViolationError, match="unknown series"):
        get_series("nope")


def test_blank_provenance_and_reason_rejected() -> None:
    with pytest.raises(ContractViolationError):
        replace(ANCHORS.provenance, citation=" ")
    with pytest.raises(ContractViolationError):
        UnavailableSeries("x", ANCHORS.quantity, ANCHORS.provenance, " ")


@pytest.mark.parametrize(
    ("distances", "values"),
    [((3.0, 3.0), (0.9, 0.8)), ((3.0,), (0.9, 0.8)), ((0.0, 3.0), (1.0, 0.8))],
)
def test_series_shape_contracts(
    distances: tuple[float, ...], values: tuple[float, ...]
) -> None:
    with pytest.raises(ContractViolationError):
        replace(ANCHORS, distances_ft=distances, values=values)
