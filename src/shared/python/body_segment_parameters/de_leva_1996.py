"""de Leva (1996) Table 4, as whole rows, for both sexes.

``data/de_leva_1996_table4.json`` transcribes every printed row of Table 4
(p. 1228): both parts of the table, female and male, exactly as printed. It is
the only place de Leva values are typed; every consumer derives from it.

Each endpoint pair has its own length, CM position and radii, and the values of
one row are only valid together. Earlier hand-copied tables mixed rows (a
length to one landmark with the CM and radii to another, #13010), so this
module hands out whole :class:`SegmentRow` records only. The loader checks the
identities the paper's numbers satisfy (whole-body mass, trunk parts adding up
to the trunk, constant mass across endpoint alternatives) and refuses a table
that breaks them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

Sex = Literal["female", "male"]

TABLE4_PATH = Path(__file__).resolve().parent / "data" / "de_leva_1996_table4.json"

_SEXES: tuple[Sex, ...] = ("female", "male")
_LIMBS = ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")
_TRUNK_PARTS = ("upt_supr_xyph", "mpt_xyph_omph", "lpt_omph_midh")
#: Printed values carry two decimals of a percent; sums may round by 0.01 %.
_MASS_SUM_TOL = 2e-4
_LENGTH_SUM_TOL_M = 1e-9


@dataclass(frozen=True)
class SegmentRow:
    """One printed row of Table 4 for one sex, in SI fractions.

    ``length_m`` is the reference subject's segment length; ``com_fraction``
    and ``radii`` (sagittal, transverse, longitudinal) are fractions of that
    length, measured from the ``origin`` endpoint.
    """

    row_id: str
    segment: str
    origin: str
    other: str
    table_part: str
    adjusted: bool
    sex: Sex
    length_m: float
    mass_fraction: float
    com_fraction: float
    radii: tuple[float, float, float]
    stature_m: float

    @property
    def length_ratio(self) -> float:
        """Segment length as a fraction of the reference subject's stature."""
        return self.length_m / self.stature_m


def _check_sex(sex: str) -> Sex:
    if sex not in _SEXES:
        raise ValueError(f"sex must be one of {_SEXES}, got {sex!r}")
    return sex  # type: ignore[return-value]


def _row(raw: dict[str, Any], sex: Sex, stature_m: float) -> SegmentRow:
    cells = raw[sex]
    return SegmentRow(
        row_id=raw["id"],
        segment=raw["segment"],
        origin=raw["origin"],
        other=raw["other"],
        table_part=raw["table_part"],
        adjusted=bool(raw["adjusted"]),
        sex=sex,
        length_m=round(cells["length_mm"] / 1000.0, 7),
        mass_fraction=round(cells["mass_pct"] / 100.0, 7),
        com_fraction=round(cells["com_pct"] / 100.0, 7),
        radii=(
            round(cells["r_sagittal_pct"] / 100.0, 7),
            round(cells["r_transverse_pct"] / 100.0, 7),
            round(cells["r_longitudinal_pct"] / 100.0, 7),
        ),
        stature_m=stature_m,
    )


def _check_invariants(rows: dict[str, SegmentRow], sex: Sex) -> None:
    """Raise ``ValueError`` unless the paper's own identities hold for ``sex``."""
    primary = {r.segment: r for r in rows.values() if r.table_part == "primary"}
    total = primary["head"].mass_fraction + primary["trunk"].mass_fraction
    total += 2 * sum(primary[s].mass_fraction for s in _LIMBS)
    if not math.isclose(total, 1.0, abs_tol=_MASS_SUM_TOL):
        raise ValueError(f"{sex} segment masses sum to {total:.4f} of body mass")
    parts = [rows[p] for p in _TRUNK_PARTS]
    trunk = rows["trunk_supr_midh"]
    if not math.isclose(
        sum(p.mass_fraction for p in parts), trunk.mass_fraction, abs_tol=1e-9
    ):
        raise ValueError(f"{sex} trunk part masses do not add up to the trunk")
    for chain, whole in (
        (_TRUNK_PARTS, "trunk_supr_midh"),
        (("upt_cerv_xyph", *_TRUNK_PARTS[1:]), "trunk_cerv_midh"),
    ):
        length = sum(rows[p].length_m for p in chain)
        if not math.isclose(length, rows[whole].length_m, abs_tol=_LENGTH_SUM_TOL_M):
            raise ValueError(f"{sex} trunk part lengths do not add up to {whole}")
    for row in rows.values():
        if row.mass_fraction != primary[row.segment].mass_fraction:
            raise ValueError(
                f"{sex} {row.row_id} mass differs from the {row.segment} row"
            )


def load_table4(path: Path = TABLE4_PATH) -> dict[Sex, dict[str, SegmentRow]]:
    """Load and check a Table 4 transcription.

    Postcondition: for each sex, the rows satisfy the identities in
    :func:`_check_invariants`; otherwise ``ValueError`` names the broken one.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    subjects = document["reference_subjects"]
    tables: dict[Sex, dict[str, SegmentRow]] = {}
    for sex in _SEXES:
        stature = float(subjects[sex]["stature_m"])
        rows = {raw["id"]: _row(raw, sex, stature) for raw in document["rows"]}
        if len(rows) != len(document["rows"]):
            raise ValueError("Table 4 row ids must be unique")
        _check_invariants(rows, sex)
        tables[sex] = rows
    return tables


@cache
def _tables() -> dict[Sex, dict[str, SegmentRow]]:
    return load_table4()


def table4_rows(sex: str) -> tuple[SegmentRow, ...]:
    """Every Table 4 row for ``sex`` in printed order."""
    return tuple(_tables()[_check_sex(sex)].values())


def table4_row(row_id: str, sex: str) -> SegmentRow:
    """One whole row by id (for example ``"shank_kjc_ajc"``)."""
    rows = _tables()[_check_sex(sex)]
    if row_id not in rows:
        raise ValueError(f"Unknown Table 4 row {row_id!r}; known: {sorted(rows)}")
    return rows[row_id]


def reference_subject(sex: str) -> tuple[float, float]:
    """``(stature_m, mass_kg)`` of the paper's reference subject for ``sex``."""
    document = json.loads(TABLE4_PATH.read_text(encoding="utf-8"))
    subject = document["reference_subjects"][_check_sex(sex)]
    return float(subject["stature_m"]), float(subject["mass_kg"])


def match_row(
    sex: str,
    mass_fraction: float,
    com_fraction: float,
    radii: Sequence[float],
    *,
    tol: float = 1e-9,
) -> str | None:
    """Id of the row whose mass, CM and radii all equal the given values.

    ``None`` when no single row carries all of them, which is how a table that
    mixes values from two endpoint rows is detected.
    """
    for row in table4_rows(sex):
        values = (row.mass_fraction, row.com_fraction, *row.radii)
        given = (mass_fraction, com_fraction, *radii)
        if len(given) == len(values) and all(
            math.isclose(a, b, abs_tol=tol) for a, b in zip(values, given, strict=True)
        ):
            return row.row_id
    return None


def table4_sha256() -> str:
    """SHA-256 of the transcription in canonical JSON form."""
    document = json.loads(TABLE4_PATH.read_text(encoding="utf-8"))
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
