"""de Leva (1996) :class:`~anthropometrics.Estimator` implementation.

This estimator is a **thin wrapper** around the canonical de Leva
ratio table that already lives in
:mod:`humanoid_character_builder.core.anthropometry`. The ratios
are NOT duplicated here — that module is the single source of
truth for de Leva data across the entire UpstreamDrift codebase.

If you need to update or extend the de Leva ratios, edit
``humanoid_character_builder/core/anthropometry.py`` and this
wrapper will pick up the change automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from humanoid_character_builder.core.anthropometry import (
    DE_LEVA_DATA,
    _SEGMENT_NAME_MAP,
    SegmentAnthropometry,
)

from .._subject_anthropometrics import SubjectAnthropometrics
from .._types import Sex
from ._base import SegmentRatios, build_subject_from_ratio_table

if TYPE_CHECKING:
    pass


METHOD_NAME = "de_leva_1996"


def _segment_classes_for_sex(sex: str) -> dict[str, SegmentRatios]:
    """Return a fresh ``class_id -> SegmentRatios`` mapping for *sex*.

    Reads directly from :data:`DE_LEVA_DATA` — the canonical
    ratio table. ``sex == "F"`` selects the female table,
    everything else selects the male table (the de Leva paper
    only published male/female tables; "unspecified" defaults to
    male, matching prior convention in the codebase).
    """
    table = DE_LEVA_DATA.female if sex == Sex.FEMALE.value else DE_LEVA_DATA.male
    return {
        class_id: _ratios_from_segment_anthropometry(seg)
        for class_id, seg in table.items()
    }


def _ratios_from_segment_anthropometry(
    seg: SegmentAnthropometry,
) -> SegmentRatios:
    """Adapt the existing dataclass to :class:`SegmentRatios`.

    This is purely a field rename; no numerical values are
    transformed. It keeps the de Leva module as the single source
    of truth for the ratios while letting the shared estimator
    driver consume one uniform shape across all three estimators.
    """
    return SegmentRatios(
        mass_ratio=seg.mass_ratio,
        length_ratio=seg.length_ratio,
        com_proximal_ratio=seg.com_proximal_ratio,
        gyration_sagittal=seg.gyration_sagittal,
        gyration_transverse=seg.gyration_transverse,
        gyration_longitudinal=seg.gyration_longitudinal,
    )


class DeLevaEstimator:
    """Estimate :class:`SubjectAnthropometrics` from de Leva (1996) ratios.

    Wraps :data:`humanoid_character_builder.core.anthropometry.DE_LEVA_DATA`
    so the ratios live in exactly one place in the codebase.
    """

    method_name: str = METHOD_NAME

    def __init__(self) -> None:
        # No state — the ratio table is looked up live on each
        # ``estimate`` call so updates to the source-of-truth
        # module are picked up without rebuilding the instance.
        pass

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

        Parameters
        ----------
        subject_id
            Non-empty subject identifier.
        height_m
            Subject standing height in metres; must be > 0.
        mass_kg
            Subject total body mass in kilograms; must be > 0.
        sex
            One of ``"M"``, ``"F"``, ``"unspecified"``. Selects
            between de Leva's male and female tables (defaults to
            male for unspecified, matching the published paper).
        age_years
            Optional, non-negative.

        Raises
        ------
        ValueError
            If any precondition is violated.
        """
        return build_subject_from_ratio_table(
            subject_id=subject_id,
            height_m=height_m,
            mass_kg=mass_kg,
            sex=sex,
            age_years=age_years,
            method_name=self.method_name,
            segment_classes=_segment_classes_for_sex(sex),
            segment_name_map=dict(_SEGMENT_NAME_MAP),
        )
