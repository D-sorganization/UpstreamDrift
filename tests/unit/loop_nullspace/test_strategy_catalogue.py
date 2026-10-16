"""NSX-95 (#12724): the strategy catalogue is data and never fakes a mechanism."""

from __future__ import annotations

import dataclasses

import pytest

from src.shared.python.loop_nullspace.strategy_spec import (
    CATALOGUE,
    NON_CLAIM,
    Availability,
    CostDeclaration,
    ModelClass,
    Representation,
    StrategyEntry,
    StrategyId,
    entry_for,
    resolve_evaluators,
)

pytestmark = pytest.mark.unit

IMPLEMENTED = {
    StrategyId.S1,
    StrategyId.S2,
    StrategyId.S3,
    StrategyId.S5,
    StrategyId.S7,
    StrategyId.S8,
}


def test_catalogue_has_exactly_s1_to_s12_in_order() -> None:
    assert [s.value for s in CATALOGUE] == [f"S{i}" for i in range(1, 13)]
    for sid, entry in CATALOGUE.items():
        assert entry.id is sid and entry_for(sid) is entry


def test_every_entry_declares_mechanism_cost_and_every_model_class() -> None:
    for entry in CATALOGUE.values():
        assert entry.name and entry.mechanism.strip()
        assert entry.cost.basis and entry.cost.description
        assert {r.model_class for r in entry.representations} == set(ModelClass)


def test_implemented_entries_point_to_real_callables() -> None:
    implemented = {s for s, e in CATALOGUE.items() if e.availability.implemented}
    assert implemented == IMPLEMENTED
    for sid in IMPLEMENTED:
        entry = CATALOGUE[sid]
        fns = resolve_evaluators(entry)
        assert fns and all(callable(f) for f in fns)
        assert entry.adapter
        assert any(r.availability.implemented for r in entry.representations)


def test_non_implemented_entries_say_why_and_have_no_evaluator() -> None:
    for sid, entry in CATALOGUE.items():
        if sid in IMPLEMENTED:
            continue
        assert entry.availability in (
            Availability.NOT_IMPLEMENTED,
            Availability.UNAVAILABLE,
        )
        assert entry.evaluators == () and entry.adapter is None
        assert all(r.reason.strip() for r in entry.representations)
        with pytest.raises(ValueError, match="no evaluator"):
            resolve_evaluators(entry)


def test_s10_is_unavailable_on_rigid_clubs_and_s6_is_not_implemented() -> None:
    s10 = CATALOGUE[StrategyId.S10]
    assert s10.availability is Availability.UNAVAILABLE
    assert "rigid" in s10.representations[0].reason.lower()
    assert CATALOGUE[StrategyId.S6].availability is Availability.NOT_IMPLEMENTED


def test_s2_has_both_a_rigid_and_a_muscle_representation() -> None:
    s2 = CATALOGUE[StrategyId.S2]
    by_class = {r.model_class: r for r in s2.representations}
    assert by_class[ModelClass.RIGID_JOINT_TORQUE].availability.implemented
    assert by_class[ModelClass.MUSCLE_DRIVEN].availability.implemented


def test_entries_are_frozen_and_the_registry_is_read_only() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        CATALOGUE[StrategyId.S1].name = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        CATALOGUE[StrategyId.S1] = CATALOGUE[StrategyId.S2]  # type: ignore[index]


def test_contract_violations_are_rejected() -> None:
    cost = CostDeclaration("basis", "text")
    rep_ok = tuple(
        Representation(m, Availability.NOT_IMPLEMENTED, "later") for m in ModelClass
    )
    base = {
        "id": StrategyId.S4,
        "name": "n",
        "mechanism": "m",
        "mechanism_ids": (),
        "representations": rep_ok,
        "cost": cost,
        "availability": Availability.NOT_IMPLEMENTED,
    }
    StrategyEntry(**base)
    with pytest.raises(ValueError, match="evaluator"):  # IMPLEMENTED without one
        StrategyEntry(**{**base, "availability": Availability.IMPLEMENTED})
    with pytest.raises(ValueError, match="reason"):
        Representation(ModelClass.MUSCLE_DRIVEN, Availability.UNAVAILABLE, "")
    with pytest.raises(ValueError, match="availability"):  # disagrees with reps
        StrategyEntry(**{**base, "availability": Availability.UNAVAILABLE})
    with pytest.raises(ValueError, match="model class"):
        StrategyEntry(**{**base, "representations": rep_ok[:1]})


def test_non_claim_is_stated() -> None:
    assert "not coaching advice" in NON_CLAIM
