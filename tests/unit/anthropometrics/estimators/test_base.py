"""Unit tests for the shared ratio-table driver in :mod:`_base`.

These cover the structural edge cases that the concrete-estimator
tests don't exercise directly (empty ratio table, name map
referencing an unknown class id, all-zero ratio sum).
"""

from __future__ import annotations

import pytest

from anthropometrics.estimators._base import (
    SegmentRatios,
    build_subject_from_ratio_table,
)


def _ratios(mass: float = 0.5) -> SegmentRatios:
    """Build a single :class:`SegmentRatios` for fixture use."""
    return SegmentRatios(
        mass_ratio=mass,
        length_ratio=0.2,
        com_proximal_ratio=0.5,
        gyration_sagittal=0.3,
        gyration_transverse=0.3,
        gyration_longitudinal=0.2,
    )


def test_empty_segment_classes_raises_value_error() -> None:
    """An empty segment_classes mapping must raise ValueError."""
    with pytest.raises(ValueError, match="segment_classes must be non-empty"):
        build_subject_from_ratio_table(
            subject_id="x",
            height_m=1.80,
            mass_kg=70.0,
            sex="M",
            age_years=None,
            method_name="test",
            segment_classes={},
            segment_name_map={"a": "b"},
        )


def test_empty_segment_name_map_raises_value_error() -> None:
    """An empty segment_name_map must raise ValueError."""
    with pytest.raises(ValueError, match="segment_name_map must be non-empty"):
        build_subject_from_ratio_table(
            subject_id="x",
            height_m=1.80,
            mass_kg=70.0,
            sex="M",
            age_years=None,
            method_name="test",
            segment_classes={"a": _ratios()},
            segment_name_map={},
        )


def test_unknown_class_in_name_map_raises_value_error() -> None:
    """Name map referencing an undefined class id must raise."""
    with pytest.raises(ValueError, match="unknown class"):
        build_subject_from_ratio_table(
            subject_id="x",
            height_m=1.80,
            mass_kg=70.0,
            sex="M",
            age_years=None,
            method_name="test",
            segment_classes={"a": _ratios()},
            segment_name_map={"head": "missing"},
        )


def test_zero_total_ratio_raises_value_error() -> None:
    """A ratio table whose mass ratios all sum to zero must raise."""
    zero = SegmentRatios(
        mass_ratio=0.0,
        length_ratio=0.2,
        com_proximal_ratio=0.5,
        gyration_sagittal=0.3,
        gyration_transverse=0.3,
        gyration_longitudinal=0.2,
    )
    with pytest.raises(ValueError, match="sum of mass ratios"):
        build_subject_from_ratio_table(
            subject_id="x",
            height_m=1.80,
            mass_kg=70.0,
            sex="M",
            age_years=None,
            method_name="test",
            segment_classes={"a": zero},
            segment_name_map={"head": "a"},
        )


def test_normalize_disabled_preserves_raw_ratio_mass() -> None:
    """With normalize_mass=False, mass equals raw_ratio * total_mass exactly."""
    subject = build_subject_from_ratio_table(
        subject_id="x",
        height_m=1.80,
        mass_kg=70.0,
        sex="M",
        age_years=None,
        method_name="test",
        segment_classes={"a": _ratios(mass=0.4)},
        segment_name_map={"head": "a"},
        normalize_mass=False,
    )
    assert subject.segments[0][1].mass_kg == pytest.approx(70.0 * 0.4)


def test_invalid_subject_id_in_base_raises_value_error() -> None:
    """The base helper must validate subject_id (non-empty string)."""
    with pytest.raises(ValueError, match="subject_id"):
        build_subject_from_ratio_table(
            subject_id="",
            height_m=1.80,
            mass_kg=70.0,
            sex="M",
            age_years=None,
            method_name="test",
            segment_classes={"a": _ratios()},
            segment_name_map={"head": "a"},
        )
