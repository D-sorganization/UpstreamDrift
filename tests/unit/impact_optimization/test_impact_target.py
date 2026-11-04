"""SHOW-C1 (#14557): ImpactTarget contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest


from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.impact_target import (
    ImpactTarget,
    ImpactWeights,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Measured:
    """Structural stand-in for impact_parameters.conditions.ImpactConditions (C2)."""

    face_angle_deg: float | None
    club_path_deg: float | None
    attack_angle_deg: float
    clubhead_speed_mph: float
    toe_mm: float | None = 0.0
    high_mm: float | None = 0.0


_ALIAS = {
    "face_deg": "face_angle_deg",
    "path_deg": "club_path_deg",
    "aoa_deg": "attack_angle_deg",
    "speed_mph": "clubhead_speed_mph",
    "strike_offset_x_m": "toe_mm",
    "strike_offset_y_m": "high_mm",
}


def _cond(**kw: float | None) -> _Measured:
    base: dict[str, float | None] = {
        "face_angle_deg": 0.0,
        "club_path_deg": 0.0,
        "attack_angle_deg": 0.0,
        "clubhead_speed_mph": 100.0,
        "toe_mm": 0.0,
        "high_mm": 0.0,
    }
    for k, v in kw.items():
        base[_ALIAS.get(k, k)] = v
    return _Measured(**base)


def test_default_target_is_straight_level_shot() -> None:
    t = ImpactTarget()
    assert (t.face_deg, t.path_deg, t.aoa_deg) == (0.0, 0.0, 0.0)
    assert t.speed_mph is None


def test_default_weights_weight_face_ten_to_one_over_path() -> None:
    w = ImpactWeights()
    assert w.face / w.path == pytest.approx(10.0)


def test_weights_are_configurable_but_must_be_positive() -> None:
    assert ImpactWeights(face=20.0, path=2.0).face == 20.0
    with pytest.raises(ContractViolationError):
        ImpactWeights(face=0.0)
    with pytest.raises(ContractViolationError):
        ImpactWeights(path=-1.0)


def test_deviation_zero_at_target() -> None:
    assert ImpactTarget().deviation(_cond()) == 0.0


@pytest.mark.parametrize("field", ["face_deg", "path_deg", "aoa_deg"])
def test_deviation_monotone_and_symmetric_in_each_angle(field: str) -> None:
    t = ImpactTarget()
    errs = [t.deviation(_cond(**{field: e})) for e in (0.5, 1.0, 2.0, 4.0)]
    assert errs == sorted(errs) and errs[0] > 0.0
    assert t.deviation(_cond(**{field: 2.0})) == pytest.approx(
        t.deviation(_cond(**{field: -2.0}))
    )


def test_one_degree_of_face_costs_ten_times_one_degree_of_path() -> None:
    t = ImpactTarget()
    assert t.deviation(_cond(face_deg=1.0)) == pytest.approx(
        10.0 * t.deviation(_cond(path_deg=1.0))
    )


def test_speed_term_only_when_requested_and_monotone() -> None:
    assert ImpactTarget().deviation(_cond(speed_mph=80.0)) == 0.0
    fast = ImpactTarget(speed_mph=125.0)
    d = [fast.deviation(_cond(speed_mph=s)) for s in (125.0, 120.0, 110.0)]
    assert d[0] == 0.0 and d[0] < d[1] < d[2]


def test_strike_offset_penalised() -> None:
    t = ImpactTarget()
    assert t.deviation(_cond(toe_mm=10.0)) > 0.0
    assert t.deviation(_cond(high_mm=-10.0)) > 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite_rejected(bad: float) -> None:
    with pytest.raises(ContractViolationError):
        ImpactTarget(face_deg=bad)
    with pytest.raises(ContractViolationError):
        ImpactTarget().deviation(_cond(path_deg=bad))


@pytest.mark.parametrize("missing", ["face_angle_deg", "club_path_deg"])
def test_unavailable_face_or_path_is_never_invented(missing: str) -> None:
    with pytest.raises(ContractViolationError):
        ImpactTarget().deviation(_cond(**{missing: None}))


def test_unavailable_strike_location_skips_strike_term() -> None:
    t = ImpactTarget()
    assert t.deviation(_cond(toe_mm=None, high_mm=None)) == 0.0


def test_nonpositive_target_speed_rejected() -> None:
    with pytest.raises(ContractViolationError):
        ImpactTarget(speed_mph=0.0)
