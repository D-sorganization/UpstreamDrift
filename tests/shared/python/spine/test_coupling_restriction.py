"""Restricting a coupling law to a model's level range (SPN-14a, #12769).

Every number is INVENTED except the shipped laws, which are read through
``coupling_data`` unchanged.  The tests check that restriction keeps the
retained matrices bit-identical, labels the result, and refuses ranges the law
does not cover.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.spine.coupling import LinearCouplingLaw, restrict_to_levels
from src.shared.python.spine.coupling_data import coupling_law
from src.shared.python.spine.levels import SpinalLevel, levels_between

pytestmark = pytest.mark.unit

LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
SIX = levels_between(SpinalLevel.L5_S1, SpinalLevel.T12_L1)
CITE = "Invented fixture law, doi:10.0000/test table 1"


def _law() -> LinearCouplingLaw:
    mats = {lv: np.eye(3) * (0.1 + 0.01 * k) for k, lv in enumerate(SIX)}
    return LinearCouplingLaw("inv", "3", mats, ("fe", "lb", "ar"), CITE)


def test_restriction_keeps_matrices_and_labels_version() -> None:
    law = _law()
    out = restrict_to_levels(law, LUMBAR)
    assert out.levels == LUMBAR
    assert out.law_id == law.law_id
    assert out.version == "3+levels:L5_S1..L1_L2"
    assert "restricted to L5_S1..L1_L2" in out.citation
    assert CITE in out.citation
    assert out.independent == law.independent
    for lv in LUMBAR:
        np.testing.assert_array_equal(out.matrices[lv], law.matrices[lv])
        assert not out.matrices[lv].flags.writeable


def test_full_range_returns_the_law_unchanged() -> None:
    law = _law()
    assert restrict_to_levels(law, SIX) is law


def test_level_order_is_irrelevant() -> None:
    law = _law()
    assert restrict_to_levels(law, tuple(reversed(LUMBAR))).levels == LUMBAR


def test_shipped_laws_restrict_to_lumbar() -> None:
    for law_id in ("bruno2015_regional", "fujii2007_lumbar", "uniform_reference"):
        out = restrict_to_levels(coupling_law(law_id), LUMBAR)
        assert out.levels == LUMBAR
        assert out.law_id == law_id


def test_preconditions() -> None:
    law = _law()
    with pytest.raises(TypeError, match="LinearCouplingLaw"):
        restrict_to_levels("law", LUMBAR)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one level"):
        restrict_to_levels(law, ())
    with pytest.raises(TypeError, match="SpinalLevel"):
        restrict_to_levels(law, ("L5_S1",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not covered"):
        restrict_to_levels(law, (SpinalLevel.T11_T12, SpinalLevel.T12_L1))
    with pytest.raises(ValueError, match="contiguous"):
        restrict_to_levels(law, (SpinalLevel.L5_S1, SpinalLevel.L3_L4))
