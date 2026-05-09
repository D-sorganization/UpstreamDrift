"""Dempster (1955) :class:`~anthropometrics.Estimator` implementation.

Loads the published Dempster ratios from
:file:`ratios/dempster_1955.json` at construction time and emits
a :class:`SubjectAnthropometrics` driven by the shared
ratio-table algebra in :mod:`_base`.
"""

from __future__ import annotations

from pathlib import Path

from .._subject_anthropometrics import SubjectAnthropometrics
from .._types import Sex
from ._base import build_subject_from_ratio_table
from ._json_loader import LoadedRatioTable, load_ratio_table

_DEFAULT_RATIO_FILE = Path(__file__).resolve().parent / "ratios" / "dempster_1955.json"


class DempsterEstimator:
    """Estimate :class:`SubjectAnthropometrics` from Dempster (1955).

    The published Dempster ratios live entirely in
    :file:`ratios/dempster_1955.json` (loaded once at
    construction). The estimator is sex-agnostic — Dempster's
    cadaveric study did not publish a sex breakdown, so the same
    ratios are used for ``"M"``, ``"F"``, and ``"unspecified"``.
    """

    method_name: str

    def __init__(self, ratio_file: Path | None = None) -> None:
        """Load the ratio table (defaults to the bundled JSON)."""
        path = ratio_file if ratio_file is not None else _DEFAULT_RATIO_FILE
        self._table: LoadedRatioTable = load_ratio_table(path)
        self.method_name = self._table.method_name

    @property
    def citation(self) -> str:
        """Return the bibliographic citation for the loaded table."""
        return self._table.citation

    def estimate(
        self,
        *,
        subject_id: str,
        height_m: float,
        mass_kg: float,
        sex: str = Sex.UNSPECIFIED.value,
        age_years: float | None = None,
    ) -> SubjectAnthropometrics:
        """Return a fully-populated :class:`SubjectAnthropometrics`.

        Raises
        ------
        ValueError
            If any precondition is violated (see
            :func:`_base.build_subject_from_ratio_table`).
        """
        return build_subject_from_ratio_table(
            subject_id=subject_id,
            height_m=height_m,
            mass_kg=mass_kg,
            sex=sex,
            age_years=age_years,
            method_name=self.method_name,
            segment_classes=self._table.segment_classes,
            segment_name_map=self._table.segment_name_map,
        )
