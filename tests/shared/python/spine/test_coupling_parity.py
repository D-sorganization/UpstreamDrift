"""Parity of the Fujii 2007 coupling law in trunk rotation (#12821).

Fujii measured coupled lumbar flexion for rotation to EITHER side, so the
flexion row must be even in the rotation; coupled lateral bend (reported
relative to the rotation direction) and axial rotation stay odd.  Magnitudes
are the extraction rows read through ``coupling_data``; nothing else is
transcribed.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.shared.python.spine.coupling import (
    CouplingLaw,
    ParityCouplingLaw,
    restrict_to_levels,
)
from src.shared.python.spine.coupling_data import (
    DEFAULT_PATH,
    coupling_law,
    load_coupling_laws,
)
from src.shared.python.spine.levels import SpinalLevel, levels_between

pytestmark = pytest.mark.unit

LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
REF = math.radians(45.0)
CITE = "Invented fixture law, doi:10.0000/test table 1"


def _rows() -> dict[SpinalLevel, dict]:
    entry = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))["laws"]
    fujii = next(e for e in entry if e["law_id"] == "fujii2007_lumbar")
    return {SpinalLevel[r["level"]]: r for r in fujii["distribution"]["rows"]}


def _phi(deg: float) -> np.ndarray:
    return np.array([math.radians(deg)])


def test_fujii_is_a_parity_law() -> None:
    law = coupling_law("fujii2007_lumbar")
    assert isinstance(law, ParityCouplingLaw)
    assert isinstance(law, CouplingLaw)
    assert law.version == "2"
    assert law.even_axes == (0,)


def test_left_and_right_rotation_give_identical_flexion() -> None:
    law = coupling_law("fujii2007_lumbar")
    left, right = law.expand(_phi(45.0)), law.expand(_phi(-45.0))
    for level in law.levels:
        assert left[level][0] == pytest.approx(right[level][0], abs=1e-15)
        assert left[level][1] == pytest.approx(-right[level][1], abs=1e-15)
        assert left[level][2] == pytest.approx(-right[level][2], abs=1e-15)


def test_flexion_is_isb_flexion_for_both_directions() -> None:
    """ISB flexion_extension is extension-positive: flexion is negative."""
    law = coupling_law("fujii2007_lumbar")
    for deg in (45.0, -45.0, 20.0, -20.0):
        for level, row in _rows().items():
            published = row["coupled_flexion"][0]
            fe = law.expand(_phi(deg))[level][0]
            assert fe * published < 0.0 or published == 0.0


def test_magnitudes_at_45_degrees_match_table_rows() -> None:
    law = coupling_law("fujii2007_lumbar")
    for sign in (1.0, -1.0):
        angles = law.expand(_phi(sign * 45.0))
        for level, row in _rows().items():
            assert angles[level][0] == pytest.approx(
                -math.radians(row["coupled_flexion"][0]), abs=1e-12
            )
            assert angles[level][1] == pytest.approx(
                -sign * math.radians(row["coupled_lateral_bend"][0]), abs=1e-12
            )
            assert angles[level][2] == pytest.approx(
                sign * math.radians(row["axial_rotation"][0]), abs=1e-12
            )


def test_jacobian_matches_finite_differences_away_from_zero() -> None:
    law = coupling_law("fujii2007_lumbar")
    for deg in (30.0, -30.0):
        phi, h = _phi(deg), 1e-6
        jac = law.jacobian(phi)
        up, dn = law.expand(phi + h), law.expand(phi - h)
        for level in law.levels:
            fd = (up[level] - dn[level]) / (2 * h)
            np.testing.assert_allclose(jac[level][:, 0], fd, atol=1e-9)


def test_jacobian_at_zero_uses_zero_subgradient_for_even_axes() -> None:
    law = coupling_law("fujii2007_lumbar")
    jac = law.jacobian(np.zeros(1))
    for level in law.levels:
        assert jac[level][0, 0] == 0.0
        assert jac[level][2, 0] == pytest.approx(law.matrices[level][2, 0])


def test_restriction_preserves_parity() -> None:
    law = coupling_law("fujii2007_lumbar")
    out = restrict_to_levels(law, LUMBAR)
    assert isinstance(out, ParityCouplingLaw)
    assert out.even_axes == law.even_axes
    for deg in (35.0, -35.0):
        for level in LUMBAR:
            np.testing.assert_array_equal(
                out.expand(_phi(deg))[level], law.expand(_phi(deg))[level]
            )
    assert not out.even_matrices[SpinalLevel.L5_S1].flags.writeable


def test_json_declares_parity_and_numbers_unchanged() -> None:
    entry = load_coupling_laws()["fujii2007_lumbar"]
    assert entry["version"] == "2"
    assert entry["distribution"]["even_in_rotation"] == ["coupled_flexion"]
    assert "Declared parity defect" not in entry["assumptions"]
    assert "|rotation|" in entry["assumptions"]


def _mats(n: int = 1) -> dict[SpinalLevel, np.ndarray]:
    return {SpinalLevel.L5_S1: np.zeros((3, n)), SpinalLevel.L4_L5: np.zeros((3, n))}


def _build(**over):
    odd, even = _mats(), _mats()
    even[SpinalLevel.L5_S1][0, 0] = 0.1
    args = {
        "law_id": "p",
        "version": "1",
        "matrices": odd,
        "even_matrices": even,
        "even_axes": (0,),
        "independent": ("r",),
        "citation": CITE,
    }
    args.update(over)
    return ParityCouplingLaw(**args)


def test_valid_construction_and_read_only() -> None:
    law = _build()
    assert not law.even_matrices[SpinalLevel.L5_S1].flags.writeable
    out = law.expand(np.array([-2.0]))[SpinalLevel.L5_S1]
    np.testing.assert_allclose(out, [0.2, 0.0, 0.0])


def test_even_term_outside_declared_axes_rejected() -> None:
    even = _mats()
    even[SpinalLevel.L5_S1][1, 0] = 0.1
    with pytest.raises(ValueError, match="even"):
        _build(even_matrices=even)


def test_odd_term_on_even_axis_rejected() -> None:
    odd = _mats()
    odd[SpinalLevel.L5_S1][0, 0] = 0.1
    with pytest.raises(ValueError, match="odd"):
        _build(matrices=odd)


def test_bad_axes_and_levels_rejected() -> None:
    with pytest.raises(ValueError, match="axis"):
        _build(even_axes=(3,))
    with pytest.raises(ValueError, match="even_axes"):
        _build(even_axes=())
    with pytest.raises(ValueError, match="levels"):
        _build(even_matrices={SpinalLevel.L5_S1: np.zeros((3, 1))})
    with pytest.raises(TypeError):
        _build(even_axes="a")
