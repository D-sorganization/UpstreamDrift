"""Unit test pinning the fit-path de Leva 1996 male table to the paper.

Every row is one whole Table 4 row (#13037; the trunk, upper_trunk and shank
rows mixed two endpoint rows before). The paper itself is checked in
``tests/unit/body_segment_parameters/``.

Reference:
de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
Journal of Biomechanics, 29(9), 1223-1230.
Table 4 (males): adjusted segment mass, center of mass position (from proximal joint),
and radii of gyration about center of mass (sagittal, transverse, longitudinal).

Segment name mapping between paper and anthropometry module:
- head: "Head and neck" (vertex to cervicale)
- trunk: "Trunk" composite (cervicale to mid-hip, CERV-MIDH)
- upper_trunk: "Thorax" (cervicale to xiphion)
- middle_trunk: "Abdomen" (xiphion to omphalion)
- lower_trunk: "Pelvis" (omphalion to hip joint centre)
- upper_arm: "Upper arm" (shoulder to elbow joint centres)
- forearm: "Forearm" (elbow to wrist joint centres)
- hand: "Hand" (wrist joint centre to metacarpale III)
- thigh: "Thigh" (hip to knee joint centres)
- shank: "Shank" / "Calf" (knee to ankle joint centres, KJC-AJC)
- foot: "Foot" (heel to toe tip)
"""

from __future__ import annotations

import pytest

from src.shared.python.motion_matching.anthropometry import DE_LEVA_MALE

pytestmark = pytest.mark.unit

# de Leva 1996 Table 4 (males, p. 1228), typed from the paper independently of
# the transcription the module derives from:
# (length_m for 1.741m reference subject, mass_fraction, com_fraction_from_proximal, (r_sag, r_tra, r_lon))
PAPER_DE_LEVA_MALE = {
    "head": (0.2429, 0.0694, 0.5002, (0.303, 0.315, 0.261)),
    "trunk": (0.6033, 0.4346, 0.5138, (0.328, 0.306, 0.169)),
    "upper_trunk": (0.2421, 0.1596, 0.5066, (0.505, 0.320, 0.465)),
    "middle_trunk": (0.2155, 0.1633, 0.4502, (0.482, 0.383, 0.468)),
    "lower_trunk": (0.1457, 0.1117, 0.6115, (0.615, 0.551, 0.587)),
    "upper_arm": (0.2817, 0.0271, 0.5772, (0.285, 0.269, 0.158)),
    "forearm": (0.2689, 0.0162, 0.4574, (0.276, 0.265, 0.121)),
    "hand": (0.0862, 0.0061, 0.7900, (0.628, 0.513, 0.401)),
    "thigh": (0.4222, 0.1416, 0.4095, (0.329, 0.329, 0.149)),
    "shank": (0.4403, 0.0433, 0.4395, (0.251, 0.246, 0.102)),
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


def test_every_row_is_derived_from_its_named_table4_row() -> None:
    """Each segment carries exactly the values of the Table 4 row it names."""
    from src.shared.python.body_segment_parameters.de_leva_1996 import table4_row
    from src.shared.python.motion_matching.anthropometry import DE_LEVA_ROWS

    assert set(DE_LEVA_ROWS) == set(DE_LEVA_MALE)
    for name, row_id in DE_LEVA_ROWS.items():
        row = table4_row(row_id, "male")
        values = DE_LEVA_MALE[name]
        assert (values.length_m, values.mass_fraction, values.com_fraction) == (
            row.length_m,
            row.mass_fraction,
            row.com_fraction,
        )
        assert values.radii == row.radii


def test_trunk_parts_add_up_to_the_trunk() -> None:
    """Upper, middle and lower trunk span cervicale to mid-hip like the trunk."""
    parts = ("upper_trunk", "middle_trunk", "lower_trunk")
    length = sum(DE_LEVA_MALE[p].length_m for p in parts)
    mass = sum(DE_LEVA_MALE[p].mass_fraction for p in parts)
    assert length == pytest.approx(DE_LEVA_MALE["trunk"].length_m, abs=1e-9)
    assert mass == pytest.approx(DE_LEVA_MALE["trunk"].mass_fraction, abs=1e-9)
