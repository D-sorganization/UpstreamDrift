"""Unit test pinning the de Leva 1996 male table against the published paper.

Reference:
de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
Journal of Biomechanics, 29(9), 1223-1230.
Table 4 (males): adjusted segment mass, center of mass position (from proximal joint),
and radii of gyration about center of mass (sagittal, transverse, longitudinal).

Segment name mapping between paper and anthropometry module:
- head: "Head and neck" (vertex to cervicale)
- trunk: "Trunk" composite (cervicale to hip joint centre)
- upper_trunk: "Thorax" (cervicale to xiphion)
- middle_trunk: "Abdomen" (xiphion to omphalion)
- lower_trunk: "Pelvis" (omphalion to hip joint centre)
- upper_arm: "Upper arm" (shoulder to elbow joint centres)
- forearm: "Forearm" (elbow to wrist joint centres)
- hand: "Hand" (wrist joint centre to metacarpale III)
- thigh: "Thigh" (hip to knee joint centres)
- shank: "Shank" / "Calf" (knee to ankle joint centres)
- foot: "Foot" (heel to toe tip)
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_matching.anthropometry import DE_LEVA_MALE

pytestmark = pytest.mark.unit

# Literal values from de Leva (1996) Table 4 (males):
# (length_m for 1.741m reference subject, mass_fraction, com_fraction_from_proximal, (r_sag, r_tra, r_lon))
PAPER_DE_LEVA_MALE = {
    "head": (0.2429, 0.0694, 0.5002, (0.303, 0.315, 0.261)),
    "trunk": (0.5319, 0.4346, 0.5138, (0.328, 0.306, 0.169)),
    "upper_trunk": (0.1707, 0.1596, 0.2999, (0.505, 0.465, 0.418)),
    "middle_trunk": (0.2155, 0.1633, 0.4502, (0.482, 0.383, 0.468)),
    "lower_trunk": (0.1457, 0.1117, 0.6115, (0.615, 0.551, 0.587)),
    "upper_arm": (0.2817, 0.0271, 0.5772, (0.285, 0.269, 0.158)),
    "forearm": (0.2689, 0.0162, 0.4574, (0.276, 0.265, 0.121)),
    "hand": (0.0862, 0.0061, 0.7900, (0.628, 0.513, 0.401)),
    "thigh": (0.4222, 0.1416, 0.4095, (0.329, 0.329, 0.149)),
    "shank": (0.4340, 0.0433, 0.4395, (0.251, 0.246, 0.102)),
    "foot": (0.2581, 0.0137, 0.4415, (0.257, 0.245, 0.124)),
}


@pytest.mark.parametrize("segment", sorted(PAPER_DE_LEVA_MALE.keys()))
def test_de_leva_male_table_entry(segment: str) -> None:
    """Verify each entry of DE_LEVA_MALE against literal paper values."""
    assert segment in DE_LEVA_MALE, f"Missing segment {segment} in DE_LEVA_MALE"
    row = DE_LEVA_MALE[segment]
    expected_len, expected_mass, expected_com, expected_radii = PAPER_DE_LEVA_MALE[
        segment
    ]

    assert row.length_m == pytest.approx(expected_len, abs=1e-4), (
        f"{segment} length_m mismatch: {row.length_m} != {expected_len}"
    )
    assert row.mass_fraction == pytest.approx(expected_mass, abs=1e-4), (
        f"{segment} mass_fraction mismatch: {row.mass_fraction} != {expected_mass}"
    )
    assert row.com_fraction == pytest.approx(expected_com, abs=1e-4), (
        f"{segment} com_fraction mismatch: {row.com_fraction} != {expected_com}"
    )

    assert len(row.radii) == 3, f"{segment} radii tuple must have 3 elements"
    for i, (actual, expected) in enumerate(zip(row.radii, expected_radii, strict=True)):
        axis = ["sagittal", "transverse", "longitudinal"][i]
        assert actual == pytest.approx(expected, abs=1e-4), (
            f"{segment} {axis} radius mismatch: {actual} != {expected}"
        )
