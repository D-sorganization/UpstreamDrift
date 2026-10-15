"""Coupler extraction (SPN-8b, #12754): .osim couplers back to a CouplingLaw.

Round trips fixture laws through ``build_fixture_osim`` and edits the generated
XML to exercise spline, polynomial and unsupported couplers.  Every number is
invented.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep  # test-only parse of generated XML
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.coupler_extraction import (
    UnsupportedCouplerError,
    extract_coupling_law,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    build_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.inventory import inventory_spine
from src.shared.python.spine.coupling import (
    LinearCouplingLaw,
    PolynomialCouplingLaw,
    StateDependentCouplingLaw,
)
from src.shared.python.spine.levels import SpinalLevel, levels_between
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.requires_opensim, pytest.mark.integration]

LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
PHI = ("trunk_flex_ext", "trunk_lat_bend", "trunk_axial_rot")
CITE = "invented, SPN-8b test"


@pytest.fixture(scope="module", autouse=True)
def _osim():
    return require_opensim()


def _law() -> LinearCouplingLaw:
    mats = {}
    for k, lv in enumerate(LUMBAR):
        w = 0.1 + 0.05 * k
        mats[lv] = np.array(
            [[w, 0.03 * (k + 1), 0.0], [0.0, w, 0.02 * (k + 1)], [0.0, 0.0, w / 2]]
        )
    return LinearCouplingLaw("invented", "1", mats, PHI, CITE)


def _write(tmp_path: Path, edit=None, name: str = "m.osim") -> Path:
    root = ET.fromstring(build_fixture_osim(FixtureSpec(), coupling=_law()))  # noqa: S314  # nosemgrep
    if edit is not None:
        edit(root)
    path = tmp_path / name
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return path


def _couplers(root: ET.Element) -> list[ET.Element]:
    return root.findall(".//CoordinateCouplerConstraint")


def _set_function(item: ET.Element, xml: str, independent: str) -> None:
    item.find("independent_coordinate_names").text = independent
    old = item.find("coupled_coordinates_function")
    for child in list(old):
        old.remove(child)
    old.append(ET.fromstring(xml))  # noqa: S314  # nosemgrep  # test literal


def _extract(path: Path, **kw):
    inv = inventory_spine(path, "fixture")
    return extract_coupling_law(
        path, inv, law_id="ext", version="2", citation=CITE, **kw
    )


class TestLinearRoundTrip:
    def test_matrices_equal_original(self, tmp_path: Path) -> None:
        law = _law()
        got = _extract(_write(tmp_path))
        assert isinstance(got, LinearCouplingLaw)
        assert got.independent == law.independent
        assert got.levels == law.levels
        for lv in law.levels:
            np.testing.assert_allclose(got.matrices[lv], law.matrices[lv], atol=1e-12)

    def test_expand_equal_for_random_phi(self, tmp_path: Path) -> None:
        law, got = _law(), _extract(_write(tmp_path))
        rng = np.random.default_rng(7)
        for _ in range(5):
            phi = rng.uniform(-0.5, 0.5, 3)
            a, b = law.expand(phi), got.expand(phi)
            for lv in law.levels:
                np.testing.assert_allclose(b[lv], a[lv], atol=1e-12)

    def test_metadata_passthrough(self, tmp_path: Path) -> None:
        got = _extract(_write(tmp_path))
        assert (got.law_id, got.version, got.citation) == ("ext", "2", CITE)

    def test_accepts_loaded_model(self, tmp_path: Path) -> None:
        osim = require_opensim()
        path = _write(tmp_path)
        model = osim.Model(str(path))
        inv = inventory_spine(model, "fixture")
        got = extract_coupling_law(model, inv, law_id="x", version="1", citation=CITE)
        assert isinstance(got, LinearCouplingLaw)

    def test_disabled_constraint_is_skipped(self, tmp_path: Path) -> None:
        def edit(root):
            _couplers(root)[0].find("isEnforced").text = "false"

        got = _extract(_write(tmp_path, edit))
        first = LUMBAR[0]
        np.testing.assert_array_equal(got.matrices[first][0], np.zeros(3))
        np.testing.assert_allclose(
            got.matrices[first][1], _law().matrices[first][1], atol=1e-12
        )

    def test_multivariate_polynomial_order_one(self, tmp_path: Path) -> None:
        xml = (
            "<MultivariatePolynomialFunction><dimension>3</dimension>"
            "<order>1</order><coefficients>0 0.5 0.25 -0.125</coefficients>"
            "</MultivariatePolynomialFunction>"
        )

        def edit(root):
            _set_function(_couplers(root)[0], xml, " ".join(PHI))

        osim = require_opensim()
        path = _write(tmp_path, edit)
        fn = osim.MultivariatePolynomialFunction.safeDownCast(
            osim.CoordinateCouplerConstraint.safeDownCast(
                osim.Model(str(path)).getConstraintSet().get(0)
            ).getFunction()
        )
        assert fn is not None
        got = _extract(path)
        probe = np.array([0.3, -0.2, 0.7])
        vec = osim.Vector(3, 0.0)
        for i in range(3):
            vec.set(i, float(probe[i]))
        expected = fn.calcValue(vec)
        row = got.matrices[LUMBAR[0]][0]
        assert float(row @ probe) == pytest.approx(expected, abs=1e-12)


class TestTabulated:
    @pytest.mark.parametrize(
        ("kind", "tag"),
        [("SimmSpline", "SimmSpline"), ("PiecewiseLinear", "PiecewiseLinearFunction")],
    )
    def test_state_dependent_exact_at_knots(self, tmp_path: Path, kind, tag) -> None:
        xs, ys = (-0.4, 0.0, 0.3, 0.6), (-0.2, 0.0, 0.1, 0.35)

        def edit(root):
            xml = (
                f"<{tag}><x>{' '.join(map(str, xs))}</x>"
                f"<y>{' '.join(map(str, ys))}</y></{tag}>"
            )
            _set_function(_couplers(root)[0], xml, PHI[0])

        got = _extract(_write(tmp_path, edit))
        assert isinstance(got, StateDependentCouplingLaw)
        assert got.scheduling == PHI[0]
        np.testing.assert_allclose(got.breakpoints, xs)
        for x, y in zip(xs, ys, strict=True):
            phi = np.array([x, 0.0, 0.0])
            assert got.expand(phi)[LUMBAR[0]][0] == pytest.approx(y, abs=1e-12)
        # An untouched multivariate-linear coupler stays exact between knots.
        phi = np.array([0.1, 0.2, -0.3])
        np.testing.assert_allclose(
            got.expand(phi)[LUMBAR[1]], _law().expand(phi)[LUMBAR[1]], atol=1e-12
        )

    def test_nonzero_value_at_origin_rejected(self, tmp_path: Path) -> None:
        def edit(root):
            xml = "<SimmSpline><x>-0.5 0.5</x><y>0.1 0.2</y></SimmSpline>"
            _set_function(_couplers(root)[0], xml, PHI[0])

        with pytest.raises(ValueError, match="l5_s1_flex_ext_coupler"):
            _extract(_write(tmp_path, edit))


class TestPolynomial:
    def _all_polynomial(self, root: ET.Element) -> None:
        for item in _couplers(root):
            xml = "<PolynomialFunction><coefficients>0.5 0.25 0</coefficients></PolynomialFunction>"
            _set_function(item, xml, PHI[0])

    def test_quadratic_gives_polynomial_law(self, tmp_path: Path) -> None:
        got = _extract(_write(tmp_path, self._all_polynomial))
        assert isinstance(got, PolynomialCouplingLaw)
        for x in (-0.3, 0.0, 0.4):
            phi = np.array([x, 0.0, 0.0])
            for angles in got.expand(phi).values():
                np.testing.assert_allclose(angles, [0.5 * x * x + 0.25 * x] * 3)

    def test_mixed_polynomial_and_linear_rejected(self, tmp_path: Path) -> None:
        def edit(root):
            xml = "<PolynomialFunction><coefficients>0.5 0.25 0</coefficients></PolynomialFunction>"
            _set_function(_couplers(root)[0], xml, PHI[0])

        with pytest.raises(UnsupportedCouplerError, match="l5_s1_flex_ext_coupler"):
            _extract(_write(tmp_path, edit))

    def test_degree_one_polynomial_is_linear(self, tmp_path: Path) -> None:
        def edit(root):
            for item in _couplers(root):
                xml = "<PolynomialFunction><coefficients>0.5 0</coefficients></PolynomialFunction>"
                _set_function(item, xml, PHI[1])

        got = _extract(_write(tmp_path, edit))
        assert isinstance(got, LinearCouplingLaw)
        assert got.independent == (PHI[1],)
        np.testing.assert_allclose(got.matrices[LUMBAR[0]], np.full((3, 1), 0.5))


class TestErrors:
    def test_unmapped_dependent_named(self, tmp_path: Path) -> None:
        def edit(root):
            _couplers(root)[0].find("dependent_coordinate_name").text = PHI[0]

        with pytest.raises(ValueError, match=PHI[0]):
            _extract(_write(tmp_path, edit))

    def test_unsupported_function_named(self, tmp_path: Path) -> None:
        def edit(root):
            _set_function(
                _couplers(root)[0], "<Constant><value>0.1</value></Constant>", PHI[0]
            )

        with pytest.raises(NotImplementedError, match="l5_s1_flex_ext_coupler"):
            _extract(_write(tmp_path, edit))

    def test_unsupported_is_also_value_error(self) -> None:
        assert issubclass(UnsupportedCouplerError, ValueError)

    def test_affine_linear_function_rejected(self, tmp_path: Path) -> None:
        def edit(root):
            xml = "<LinearFunction><coefficients>0.5 0.1</coefficients></LinearFunction>"
            _set_function(_couplers(root)[0], xml, PHI[0])

        with pytest.raises(ValueError, match="intercept"):
            _extract(_write(tmp_path, edit))

    def test_no_couplers_clear_message(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.osim"
        path.write_text(build_fixture_osim(FixtureSpec()), encoding="utf-8")
        inv = inventory_spine(path, "fixture")
        with pytest.raises(ValueError, match="no CoordinateCouplerConstraint"):
            extract_coupling_law(path, inv, law_id="x", version="1", citation=CITE)

    def test_bad_arguments(self, tmp_path: Path) -> None:
        path = _write(tmp_path)
        inv = inventory_spine(path, "fixture")
        with pytest.raises(TypeError, match="inventory"):
            extract_coupling_law(path, object(), law_id="x", version="1", citation=CITE)
        with pytest.raises(ValueError, match="law_id"):
            extract_coupling_law(path, inv, law_id=" ", version="1", citation=CITE)
        with pytest.raises(ValueError, match="not found"):
            extract_coupling_law(
                tmp_path / "nope.osim", inv, law_id="x", version="1", citation=CITE
            )
