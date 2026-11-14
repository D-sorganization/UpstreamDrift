"""Shared estimate path for JSON-backed ratio-table estimators."""

from __future__ import annotations

from pathlib import Path

from .._subject_anthropometrics import SubjectAnthropometrics
from .._types import Sex
from ._base import build_subject_from_ratio_table
from ._humanoid_expansion import resolve_humanoid_segment_table
from ._json_loader import LoadedRatioTable, load_ratio_table


class JsonRatioTableEstimator:
    """Mixin for estimators backed by a bundled JSON ratio table."""

    _table: LoadedRatioTable

    @property
    def method_name(self) -> str:
        """Estimator provenance key from the loaded table."""
        return self._table.method_name

    @property
    def citation(self) -> str:
        """Bibliographic citation from the loaded table."""
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
        """Return a fully-populated :class:`SubjectAnthropometrics`."""
        return estimate_from_loaded_ratio_table(
            self._table,
            subject_id=subject_id,
            height_m=height_m,
            mass_kg=mass_kg,
            sex=sex,
            age_years=age_years,
        )

    @classmethod
    def from_ratio_file(cls, path: Path) -> JsonRatioTableEstimator:
        """Construct from a ratio-table JSON path."""
        instance = cls.__new__(cls)
        instance._table = load_ratio_table(path)
        return instance


def estimate_from_loaded_ratio_table(
    table: LoadedRatioTable,
    *,
    subject_id: str,
    height_m: float,
    mass_kg: float,
    sex: str,
    age_years: float | None,
) -> SubjectAnthropometrics:
    """Materialise a subject from a parsed ratio-table file."""
    segment_classes, segment_name_map, normalize_mass = resolve_humanoid_segment_table(
        table.segment_classes,
        table.segment_name_map,
    )
    return build_subject_from_ratio_table(
        subject_id=subject_id,
        height_m=height_m,
        mass_kg=mass_kg,
        sex=sex,
        age_years=age_years,
        method_name=table.method_name,
        segment_classes=segment_classes,
        segment_name_map=segment_name_map,
        normalize_mass=normalize_mass,
    )
