"""Unit tests for :mod:`anthropometrics.estimators.from_inertia_calc`.

Verifies analytical correctness of all three inertia computations
plus the :func:`build_segment_properties_with_inertia` builder, and
asserts that every produced tensor is symmetric, positive-definite,
and satisfies the triangle inequality (the latter via the
:class:`SegmentProperties` constructor's own validation).
"""

from __future__ import annotations

import numpy as np
import pytest

from anthropometrics import SegmentProperties
from anthropometrics.estimators.from_inertia_calc import (
    build_segment_properties_with_inertia,
    inertia_from_cylinder,
    inertia_from_ellipsoid,
    inertia_from_gyration_radii,
)

ATOL = 1e-9


# --------------------------------------------------------------------------- #
# Helpers.                                                                    #
# --------------------------------------------------------------------------- #
def _assert_physical_inertia(tensor: np.ndarray) -> None:
    """Verify symmetry, positive-definiteness, and triangle inequality."""
    assert tensor.shape == (3, 3)
    assert np.allclose(tensor, tensor.T, atol=ATOL)

    eigenvalues = np.linalg.eigvalsh(tensor)
    assert np.all(eigenvalues > 0), f"non positive-definite: {eigenvalues}"

    ix, iy, iz = sorted(float(e) for e in eigenvalues)
    assert ix + iy + ATOL >= iz
    assert iy + iz + ATOL >= ix
    assert ix + iz + ATOL >= iy


def _default_kwargs() -> dict[str, object]:
    """Return a ``SegmentProperties`` kwargs dict valid except for inertia."""
    return {
        "name": "upper_arm_left",
        "body_part_id": "upper_arm",
        "length_m": 0.30,
        "proximal_marker": "L_SHO",
        "distal_marker": "L_ELB",
        "mass_kg": 2.0,
        "com_xyz_m": np.array([0.15, 0.0, 0.0]),
        "source_method": "de_leva",
        "source_subject_height_m": 1.80,
        "source_subject_mass_kg": 75.0,
    }


# --------------------------------------------------------------------------- #
# Cylinder.                                                                   #
# --------------------------------------------------------------------------- #
class TestInertiaFromCylinder:
    def test_canonical_values(self) -> None:
        """Cylinder (m=1, L=2, r=0.1) matches analytic Ix and Iy=Iz."""
        tensor = inertia_from_cylinder(mass_kg=1.0, length_m=2.0, radius_m=0.1)

        expected_ix = 1.0 * 0.1**2 / 2.0  # 0.005
        expected_iy = 1.0 * (3.0 * 0.1**2 + 2.0**2) / 12.0  # 0.33583333...

        assert tensor[0, 0] == pytest.approx(expected_ix, abs=ATOL)
        assert tensor[1, 1] == pytest.approx(expected_iy, abs=ATOL)
        assert tensor[2, 2] == pytest.approx(expected_iy, abs=ATOL)

        # Off-diagonals zero.
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert tensor[i, j] == pytest.approx(0.0, abs=ATOL)
            assert tensor[j, i] == pytest.approx(0.0, abs=ATOL)

        _assert_physical_inertia(tensor)

    def test_independent_scaling(self) -> None:
        """Doubling mass doubles every principal moment."""
        base = inertia_from_cylinder(mass_kg=1.0, length_m=2.0, radius_m=0.1)
        scaled = inertia_from_cylinder(mass_kg=2.0, length_m=2.0, radius_m=0.1)
        assert np.allclose(2.0 * base, scaled, atol=ATOL)

    @pytest.mark.parametrize(
        ("mass_kg", "length_m", "radius_m", "label"),
        [
            (0.0, 1.0, 0.1, "mass_kg"),
            (-1.0, 1.0, 0.1, "mass_kg"),
            (1.0, 0.0, 0.1, "length_m"),
            (1.0, -1.0, 0.1, "length_m"),
            (1.0, 1.0, 0.0, "radius_m"),
            (1.0, 1.0, -0.1, "radius_m"),
            (float("nan"), 1.0, 0.1, "mass_kg"),
            (float("inf"), 1.0, 0.1, "mass_kg"),
        ],
    )
    def test_rejects_non_positive(
        self, mass_kg: float, length_m: float, radius_m: float, label: str
    ) -> None:
        with pytest.raises(ValueError, match=label):
            inertia_from_cylinder(mass_kg=mass_kg, length_m=length_m, radius_m=radius_m)


# --------------------------------------------------------------------------- #
# Ellipsoid.                                                                  #
# --------------------------------------------------------------------------- #
class TestInertiaFromEllipsoid:
    def test_unit_sphere(self) -> None:
        """Unit sphere (m=1, a=b=c=1) gives 0.4 on every principal axis."""
        tensor = inertia_from_ellipsoid(mass_kg=1.0, a_m=1.0, b_m=1.0, c_m=1.0)
        for i in range(3):
            assert tensor[i, i] == pytest.approx(0.4, abs=ATOL)
        _assert_physical_inertia(tensor)

    def test_general_ellipsoid_principal_moments(self) -> None:
        """Non-degenerate ellipsoid gives the standard m/5 formula on each axis."""
        m, a, b, c = 3.0, 0.4, 0.2, 0.1
        tensor = inertia_from_ellipsoid(m, a, b, c)
        assert tensor[0, 0] == pytest.approx(m * (b**2 + c**2) / 5.0, abs=ATOL)
        assert tensor[1, 1] == pytest.approx(m * (a**2 + c**2) / 5.0, abs=ATOL)
        assert tensor[2, 2] == pytest.approx(m * (a**2 + b**2) / 5.0, abs=ATOL)
        _assert_physical_inertia(tensor)

    @pytest.mark.parametrize(
        ("mass_kg", "a_m", "b_m", "c_m", "label"),
        [
            (0.0, 1.0, 1.0, 1.0, "mass_kg"),
            (1.0, 0.0, 1.0, 1.0, "a_m"),
            (1.0, 1.0, 0.0, 1.0, "b_m"),
            (1.0, 1.0, 1.0, 0.0, "c_m"),
            (1.0, -1.0, 1.0, 1.0, "a_m"),
        ],
    )
    def test_rejects_non_positive(
        self,
        mass_kg: float,
        a_m: float,
        b_m: float,
        c_m: float,
        label: str,
    ) -> None:
        with pytest.raises(ValueError, match=label):
            inertia_from_ellipsoid(mass_kg, a_m, b_m, c_m)


# --------------------------------------------------------------------------- #
# Gyration-radii.                                                             #
# --------------------------------------------------------------------------- #
class TestInertiaFromGyrationRadii:
    def test_canonical_values(self) -> None:
        """k=(0.5, 0.3, 0.4), L=1, m=1 gives I=(0.25, 0.09, 0.16)."""
        tensor = inertia_from_gyration_radii(
            mass_kg=1.0, length_m=1.0, gyration_ratios=(0.5, 0.3, 0.4)
        )
        assert tensor[0, 0] == pytest.approx(0.25, abs=ATOL)
        assert tensor[1, 1] == pytest.approx(0.09, abs=ATOL)
        assert tensor[2, 2] == pytest.approx(0.16, abs=ATOL)
        _assert_physical_inertia(tensor)

    def test_quadratic_in_length(self) -> None:
        """Doubling length should quadruple every principal moment."""
        ratios = (0.4, 0.3, 0.35)
        base = inertia_from_gyration_radii(
            mass_kg=1.5, length_m=0.4, gyration_ratios=ratios
        )
        doubled = inertia_from_gyration_radii(
            mass_kg=1.5, length_m=0.8, gyration_ratios=ratios
        )
        assert np.allclose(4.0 * base, doubled, atol=ATOL)

    @pytest.mark.parametrize(
        ("mass_kg", "length_m", "ratios", "match"),
        [
            (0.0, 1.0, (0.5, 0.3, 0.4), "mass_kg"),
            (1.0, 0.0, (0.5, 0.3, 0.4), "length_m"),
            (1.0, 1.0, (0.0, 0.3, 0.4), "k_x"),
            (1.0, 1.0, (0.5, -0.1, 0.4), "k_y"),
            (1.0, 1.0, (0.5, 0.3, float("nan")), "k_z"),
        ],
    )
    def test_rejects_invalid_inputs(
        self,
        mass_kg: float,
        length_m: float,
        ratios: tuple[float, float, float],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            inertia_from_gyration_radii(
                mass_kg=mass_kg, length_m=length_m, gyration_ratios=ratios
            )

    def test_rejects_wrong_arity(self) -> None:
        with pytest.raises(ValueError, match="exactly 3 elements"):
            inertia_from_gyration_radii(
                mass_kg=1.0,
                length_m=1.0,
                gyration_ratios=(0.5, 0.3),  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------------- #
# Builder.                                                                    #
# --------------------------------------------------------------------------- #
class TestBuildSegmentPropertiesWithInertia:
    def test_cylinder_builder_matches_direct_call(self) -> None:
        """Builder's tensor matches a direct ``inertia_from_cylinder`` call."""
        kwargs = _default_kwargs()
        seg = build_segment_properties_with_inertia(
            kwargs["name"],
            kwargs["body_part_id"],
            mass_kg=kwargs["mass_kg"],
            length_m=kwargs["length_m"],
            com_xyz_m=kwargs["com_xyz_m"],
            method="cylinder",
            method_params={"radius_m": 0.04},
            source_method=kwargs["source_method"],
            source_subject_height_m=kwargs["source_subject_height_m"],
            source_subject_mass_kg=kwargs["source_subject_mass_kg"],
            proximal_marker=kwargs["proximal_marker"],
            distal_marker=kwargs["distal_marker"],
        )

        assert isinstance(seg, SegmentProperties)
        expected = inertia_from_cylinder(
            mass_kg=kwargs["mass_kg"],
            length_m=kwargs["length_m"],
            radius_m=0.04,
        )
        assert np.allclose(seg.inertia_tensor, expected, atol=ATOL)
        _assert_physical_inertia(seg.inertia_tensor)

    def test_ellipsoid_builder(self) -> None:
        kwargs = _default_kwargs()
        seg = build_segment_properties_with_inertia(
            kwargs["name"],
            kwargs["body_part_id"],
            mass_kg=kwargs["mass_kg"],
            length_m=kwargs["length_m"],
            com_xyz_m=kwargs["com_xyz_m"],
            method="ellipsoid",
            method_params={"a_m": 0.15, "b_m": 0.05, "c_m": 0.05},
            source_method=kwargs["source_method"],
            source_subject_height_m=kwargs["source_subject_height_m"],
            source_subject_mass_kg=kwargs["source_subject_mass_kg"],
        )

        expected = inertia_from_ellipsoid(
            mass_kg=kwargs["mass_kg"], a_m=0.15, b_m=0.05, c_m=0.05
        )
        assert np.allclose(seg.inertia_tensor, expected, atol=ATOL)
        _assert_physical_inertia(seg.inertia_tensor)

    def test_gyration_radii_builder(self) -> None:
        kwargs = _default_kwargs()
        ratios = (0.322, 0.303, 0.158)  # de Leva-ish for upper arm
        seg = build_segment_properties_with_inertia(
            kwargs["name"],
            kwargs["body_part_id"],
            mass_kg=kwargs["mass_kg"],
            length_m=kwargs["length_m"],
            com_xyz_m=kwargs["com_xyz_m"],
            method="gyration_radii",
            method_params={"gyration_ratios": ratios},
            source_method=kwargs["source_method"],
            source_subject_height_m=kwargs["source_subject_height_m"],
            source_subject_mass_kg=kwargs["source_subject_mass_kg"],
        )

        expected = inertia_from_gyration_radii(
            mass_kg=kwargs["mass_kg"],
            length_m=kwargs["length_m"],
            gyration_ratios=ratios,
        )
        assert np.allclose(seg.inertia_tensor, expected, atol=ATOL)
        _assert_physical_inertia(seg.inertia_tensor)

    def test_unknown_method_rejected(self) -> None:
        kwargs = _default_kwargs()
        with pytest.raises(ValueError, match="method must be one of"):
            build_segment_properties_with_inertia(
                kwargs["name"],
                kwargs["body_part_id"],
                mass_kg=kwargs["mass_kg"],
                length_m=kwargs["length_m"],
                com_xyz_m=kwargs["com_xyz_m"],
                method="bogus",  # type: ignore[arg-type]
                method_params={},
                source_method=kwargs["source_method"],
                source_subject_height_m=kwargs["source_subject_height_m"],
                source_subject_mass_kg=kwargs["source_subject_mass_kg"],
            )

    def test_cylinder_missing_param_rejected(self) -> None:
        kwargs = _default_kwargs()
        with pytest.raises(ValueError, match="radius_m"):
            build_segment_properties_with_inertia(
                kwargs["name"],
                kwargs["body_part_id"],
                mass_kg=kwargs["mass_kg"],
                length_m=kwargs["length_m"],
                com_xyz_m=kwargs["com_xyz_m"],
                method="cylinder",
                method_params={},
                source_method=kwargs["source_method"],
                source_subject_height_m=kwargs["source_subject_height_m"],
                source_subject_mass_kg=kwargs["source_subject_mass_kg"],
            )

    def test_ellipsoid_missing_param_rejected(self) -> None:
        kwargs = _default_kwargs()
        with pytest.raises(ValueError, match="a_m"):
            build_segment_properties_with_inertia(
                kwargs["name"],
                kwargs["body_part_id"],
                mass_kg=kwargs["mass_kg"],
                length_m=kwargs["length_m"],
                com_xyz_m=kwargs["com_xyz_m"],
                method="ellipsoid",
                method_params={"b_m": 0.1, "c_m": 0.1},
                source_method=kwargs["source_method"],
                source_subject_height_m=kwargs["source_subject_height_m"],
                source_subject_mass_kg=kwargs["source_subject_mass_kg"],
            )

    def test_gyration_missing_param_rejected(self) -> None:
        kwargs = _default_kwargs()
        with pytest.raises(ValueError, match="gyration_ratios"):
            build_segment_properties_with_inertia(
                kwargs["name"],
                kwargs["body_part_id"],
                mass_kg=kwargs["mass_kg"],
                length_m=kwargs["length_m"],
                com_xyz_m=kwargs["com_xyz_m"],
                method="gyration_radii",
                method_params={},
                source_method=kwargs["source_method"],
                source_subject_height_m=kwargs["source_subject_height_m"],
                source_subject_mass_kg=kwargs["source_subject_mass_kg"],
            )

    def test_segment_properties_validation_propagates(self) -> None:
        """A non-positive ``mass_kg`` is caught before the SP is constructed."""
        kwargs = _default_kwargs()
        with pytest.raises(ValueError, match="mass_kg"):
            build_segment_properties_with_inertia(
                kwargs["name"],
                kwargs["body_part_id"],
                mass_kg=-1.0,
                length_m=kwargs["length_m"],
                com_xyz_m=kwargs["com_xyz_m"],
                method="cylinder",
                method_params={"radius_m": 0.04},
                source_method=kwargs["source_method"],
                source_subject_height_m=kwargs["source_subject_height_m"],
                source_subject_mass_kg=kwargs["source_subject_mass_kg"],
            )
