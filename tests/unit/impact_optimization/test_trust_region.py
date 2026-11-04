"""SHOW-C1 (#14557): TrustRegion around the matched trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.trust_region import TrustRegion

REF = np.linspace(0.0, 1.0, 50)[:, None] * np.array([[1.0, 2.0, 3.0]])


def _tr(radius: float = 1.0, w: tuple[float, ...] = (1.0, 1.0, 1.0)) -> TrustRegion:
    return TrustRegion(reference=REF, weights=np.array(w), radius=radius)


def test_zero_deviation_on_reference() -> None:
    tr = _tr()
    assert tr.deviation(REF) == 0.0
    assert tr.contains(REF)
    assert tr.penalty(REF) == 0.0


def test_deviation_monotone_and_weighted() -> None:
    tr = _tr()
    d = [tr.deviation(REF + s) for s in (0.1, 0.2, 0.4)]
    assert d[0] < d[1] < d[2]
    bump = np.zeros_like(REF)
    bump[:, 0] = 0.3
    heavy = _tr(w=(4.0, 1.0, 1.0))
    assert heavy.deviation(REF + bump) == pytest.approx(2.0 * tr.deviation(REF + bump))


def test_outside_radius_is_penalised() -> None:
    tr = _tr(radius=0.1)
    far = REF + 1.0
    assert not tr.contains(far)
    assert tr.penalty(far) > 0.0
    assert tr.slack(far) < 0.0


def test_reference_is_read_only_copy() -> None:
    ref = REF.copy()
    tr = TrustRegion(reference=ref, weights=np.ones(3), radius=1.0)
    ref[0, 0] = 99.0
    assert tr.reference[0, 0] == 0.0
    with pytest.raises(ValueError):
        tr.reference[0, 0] = 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0.0},
        {"radius": float("nan")},
        {"w": (1.0, 0.0, 1.0)},
        {"w": (1.0, 1.0)},
    ],
)
def test_invalid_construction_rejected(kwargs: dict) -> None:
    with pytest.raises(ContractViolationError):
        _tr(**kwargs)


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ContractViolationError):
        _tr().deviation(REF[:, :2])
