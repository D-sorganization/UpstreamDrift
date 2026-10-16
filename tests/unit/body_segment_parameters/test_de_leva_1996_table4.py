"""The de Leva (1996) Table 4 transcription is complete and internally consistent.

The transcription in ``body_segment_parameters/data/de_leva_1996_table4.json`` is
the single source of de Leva values (#13010). Two earlier hand-copied tables
paired a length from one endpoint row with the CM position or radii of another
(for example the shank length to the lateral malleolus with the CM and radii to
the ankle joint centre). These tests check the transcription against arithmetic
identities the paper's own numbers satisfy, so a mistyped cell fails here, and
expose rows only as whole records, so values cannot be mixed between rows.
"""

from __future__ import annotations

import json

import pytest

from src.shared.python.body_segment_parameters import de_leva_1996 as table4

pytestmark = pytest.mark.unit

SEXES = ("female", "male")
LIMBS = ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")

#: Recomputed whenever the transcription is edited; a change must be reviewed
#: against the paper (Table 4, p. 1228) and the verification record updated.
TABLE4_SHA256 = "c04fbc4f4dc8d77ffa3c7daa7d6b3f80df991413bff8335dacdd44c1a46e8f4f"


def _primary(segment: str, sex: str) -> table4.SegmentRow:
    (row,) = [
        r
        for r in table4.table4_rows(sex)
        if r.segment == segment and r.table_part == "primary"
    ]
    return row


def test_every_printed_row_is_transcribed_for_both_sexes() -> None:
    for sex in SEXES:
        rows = table4.table4_rows(sex)
        assert len(rows) == 21
        assert len({r.row_id for r in rows}) == 21
        assert sum(r.table_part == "primary" for r in rows) == 11


def test_whole_body_mass_is_one_hundred_percent() -> None:
    """Head, trunk and two of each limb segment make up the body (paper p. 1229)."""
    for sex in SEXES:
        total = _primary("head", sex).mass_fraction + _primary("trunk", sex).mass_fraction
        total += 2 * sum(_primary(s, sex).mass_fraction for s in LIMBS)
        assert total == pytest.approx(1.0, abs=2e-4), sex


def test_trunk_parts_add_up_to_the_trunk() -> None:
    for sex in SEXES:
        parts = ("upt_supr_xyph", "mpt_xyph_omph", "lpt_omph_midh")
        mass = sum(table4.table4_row(p, sex).mass_fraction for p in parts)
        length = sum(table4.table4_row(p, sex).length_m for p in parts)
        trunk = table4.table4_row("trunk_supr_midh", sex)
        assert mass == pytest.approx(trunk.mass_fraction, abs=1e-12)
        assert length == pytest.approx(trunk.length_m, abs=1e-12)
        # The same chain measured from the cervicale.
        cerv = ("upt_cerv_xyph", "mpt_xyph_omph", "lpt_omph_midh")
        length = sum(table4.table4_row(p, sex).length_m for p in cerv)
        assert length == pytest.approx(
            table4.table4_row("trunk_cerv_midh", sex).length_m, abs=1e-12
        )


def test_trunk_cm_follows_from_its_parts() -> None:
    """The mass-weighted CM of the three parts lies at the tabulated trunk CM.

    Female rows agree to 0.01 %; the male trunk CM differs by 0.55 % of trunk
    length, which the paper derives separately (p. 1229). One percent catches
    a mistyped digit in any of the twelve cells involved.
    """
    for sex in SEXES:
        offset, moment, mass = 0.0, 0.0, 0.0
        for part in ("upt_supr_xyph", "mpt_xyph_omph", "lpt_omph_midh"):
            row = table4.table4_row(part, sex)
            moment += row.mass_fraction * (offset + row.com_fraction * row.length_m)
            mass += row.mass_fraction
            offset += row.length_m
        trunk = table4.table4_row("trunk_supr_midh", sex)
        assert moment / mass / trunk.length_m == pytest.approx(
            trunk.com_fraction, abs=0.01
        ), sex


def test_endpoint_alternatives_keep_the_segment_mass() -> None:
    for sex in SEXES:
        for row in table4.table4_rows(sex):
            assert row.mass_fraction == _primary(row.segment, sex).mass_fraction


def test_values_are_physical() -> None:
    for sex in SEXES:
        for row in table4.table4_rows(sex):
            assert 0.0 < row.length_m < 1.0
            assert 0.0 < row.mass_fraction < 0.5
            assert 0.0 < row.com_fraction < 1.0
            assert all(0.0 < k < 1.0 for k in row.radii)


def test_rows_scale_to_the_reference_subject() -> None:
    row = table4.table4_row("shank_kjc_ajc", "male")
    assert (row.length_m, row.com_fraction, row.radii) == (
        0.4403,
        0.4395,
        (0.251, 0.246, 0.102),
    )
    assert row.length_ratio == pytest.approx(0.4403 / 1.741)
    assert table4.reference_subject("female") == (1.735, 61.9)


def test_unknown_rows_and_sexes_are_rejected() -> None:
    with pytest.raises(ValueError, match="row"):
        table4.table4_row("shin", "male")
    with pytest.raises(ValueError, match="sex"):
        table4.table4_rows("unspecified")


def test_transcription_is_pinned() -> None:
    assert table4.table4_sha256() == TABLE4_SHA256


def test_a_mistyped_cell_is_rejected_on_load(tmp_path) -> None:
    document = json.loads(table4.TABLE4_PATH.read_text(encoding="utf-8"))
    document["rows"][9]["male"]["mass_pct"] = 4.38
    path = tmp_path / "table4.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="mass"):
        table4.load_table4(path)


def test_match_row_finds_only_whole_rows() -> None:
    row = table4.table4_row("head_vert_cerv", "male")
    assert table4.match_row("male", row.mass_fraction, row.com_fraction, row.radii) == (
        "head_vert_cerv"
    )
    # CM of the cervicale row with the radii of the mid-gonion row: no row.
    midg = table4.table4_row("head_vert_midg", "male")
    assert table4.match_row("male", row.mass_fraction, row.com_fraction, midg.radii) is None
