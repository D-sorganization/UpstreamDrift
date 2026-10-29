"""Contract tests for interface-ellipsoid kinds, provenance and parity ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.spatial_algebra.interface_ellipsoids import (
    EllipsoidProvenance,
    InterfaceEllipsoidKind,
    load_parity_ledger,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ENGINES_DIR = ROOT / "src" / "engines" / "physics_engines"
K = InterfaceEllipsoidKind


def test_kind_values_and_units():
    assert [k.value for k in K] == [
        "mobility",
        "effective-mass",
        "acceleration-capacity",
        "force-capacity",
    ]
    assert K.MOBILITY.radius_unit == "kg^-1/2"
    assert K.EFFECTIVE_MASS.radius_unit == "kg^1/2"
    assert K.ACCELERATION_CAPACITY.radius_unit == "m/s^2"
    assert K.FORCE_CAPACITY.radius_unit == "N"


def test_capacity_flags():
    assert not K.MOBILITY.is_capacity and not K.MOBILITY.requires_torque_limits
    assert K.FORCE_CAPACITY.is_capacity and K.FORCE_CAPACITY.inner_approximation
    assert K.ACCELERATION_CAPACITY.requires_torque_limits
    assert not K.EFFECTIVE_MASS.inner_approximation
    assert all(k.label for k in K)


def _prov(**kw):
    base = {
        "engine": "mujoco",
        "topology": "welded_lead_hand",
        "closure": "constrained",
        "torque_limit_source": None,
        "assumed_slots": 0,
        "verified_slots": 0,
        "kind": K.MOBILITY,
    }
    base.update(kw)
    return EllipsoidProvenance(**base)


def test_provenance_label_inertial():
    assert "inner approx." not in _prov().label()


def test_provenance_label_capacity_mentions_inner_and_assumed():
    p = _prov(
        kind=K.FORCE_CAPACITY,
        torque_limit_source="loop_nullspace_registry",
        assumed_slots=3,
    )
    text = p.label()
    assert "inner approx." in text and "3 assumed" in text


@pytest.mark.parametrize(
    "kw, exc",
    [
        ({"engine": ""}, ValueError),
        ({"closure": "open"}, ValueError),
        ({"assumed_slots": -1}, ValueError),
        ({"verified_slots": 1.5}, TypeError),
        ({"kind": "mobility"}, TypeError),
        ({"kind": K.ACCELERATION_CAPACITY}, ValueError),  # needs limit source
        ({"kind": K.FORCE_CAPACITY, "torque_limit_source": " "}, ValueError),
    ],
)
def test_provenance_rejects_bad_input(kw, exc):
    with pytest.raises(exc):
        _prov(**kw)


def test_ledger_covers_every_engine_and_kind():
    ledger = load_parity_ledger()
    dirs = {
        p.name
        for p in ENGINES_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "tests"))
    }
    assert dirs | {"simscape"} <= set(ledger)
    for entry in ledger.values():
        assert set(entry) == {k.value for k in K}


def test_ledger_gaps_have_issues_and_exempt_have_reasons():
    for engine, entry in load_parity_ledger().items():
        for kind, rec in entry.items():
            if rec["status"] == "gap":
                assert isinstance(rec["issue"], int), (engine, kind)
            if rec["status"] == "exempt":
                assert rec["reason"].strip(), (engine, kind)


def _write(tmp_path, data):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(data))
    return p


def test_loader_rejects_unknown_kind_and_status(tmp_path):
    full = {k.value: {"status": "parity"} for k in K}
    bad_kind = {"e": {**full, "bogus": {"status": "parity"}}}
    with pytest.raises(ValueError, match="kind"):
        load_parity_ledger(_write(tmp_path, {"engines": bad_kind}))
    bad_status = {"e": {**full, "mobility": {"status": "maybe"}}}
    with pytest.raises(ValueError, match="status"):
        load_parity_ledger(_write(tmp_path, {"engines": bad_status}))
    no_issue = {"e": {**full, "mobility": {"status": "gap"}}}
    with pytest.raises(ValueError, match="issue"):
        load_parity_ledger(_write(tmp_path, {"engines": no_issue}))
