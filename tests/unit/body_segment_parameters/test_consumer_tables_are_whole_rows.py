"""Every de Leva table in the codebase uses whole Table 4 rows (#13010).

A de Leva length, CM position and set of radii are only valid together: each
endpoint pair has its own row. Mixing rows went unnoticed twice, so every
consumer table is checked against the transcription here.
"""

from __future__ import annotations

import pytest

from src.shared.python.body_segment_parameters import de_leva_1996 as table4
from src.shared.python.motion_matching.anthropometry import DE_LEVA_MALE

pytestmark = pytest.mark.unit

#: Rows of the fit-path table known to mix endpoint rows; correcting them
#: changes the golfer geometry and the receipt hash, so they are fixed with a
#: refit in #13037. This set must equal the mixed rows: it can only shrink.
KNOWN_MIXED = {"trunk", "upper_trunk", "shank"}


def _whole_row(sex: str, length_m: float, mass: float, com: float, radii) -> bool:
    row_id = table4.match_row(sex, mass, com, radii)
    if row_id is None:
        return False
    return table4.table4_row(row_id, sex).length_m == pytest.approx(length_m, abs=1e-9)


def test_fit_path_table_mixes_only_the_known_rows() -> None:
    mixed = {
        name
        for name, row in DE_LEVA_MALE.items()
        if not _whole_row(
            "male", row.length_m, row.mass_fraction, row.com_fraction, row.radii
        )
    }
    assert mixed == KNOWN_MIXED
