"""Shared internals for ratio-table driven anthropometric estimators.

The three concrete estimators (:mod:`from_de_leva`,
:mod:`from_dempster`, :mod:`from_zatsiorsky`) all share the same
algebra: given a published table of mass / length / CoM /
gyration ratios per segment class, produce one
:class:`~anthropometrics.SegmentProperties` per named anatomical
segment.

This module factors that algebra out into a single private
helper, :func:`build_subject_from_ratio_table`, so the concrete
estimators contain only their data source plumbing (a JSON file
path or a pre-existing dataclass table) and a thin adapter that
yields the canonical ratio dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._subject_anthropometrics import SubjectAnthropometrics
from .._types import Sex
from ..segment_properties import SegmentProperties

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class SegmentRatios:
    """Published ratios for one segment class.

    All ratios are dimensionless. ``mass_ratio`` is a fraction of
    total body mass; ``length_ratio`` is a fraction of standing
    height; ``com_proximal_ratio`` locates the centre of mass
    along the segment from the proximal end as a fraction of
    segment length; the three gyration radii are fractions of
    segment length about the principal axes of the segment frame
    (sagittal = mediolateral axis, transverse = anteroposterior
    axis, longitudinal = the long axis).
    """

    mass_ratio: float
    length_ratio: float
    com_proximal_ratio: float
    gyration_sagittal: float
    gyration_transverse: float
    gyration_longitudinal: float


def _validate_subject_inputs(
    subject_id: str,
    height_m: float,
    mass_kg: float,
    sex: str,
    age_years: float | None,
) -> None:
    """Raise ``ValueError`` if any subject-level input is invalid."""
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError(f"subject_id must be a non-empty string, got {subject_id!r}")
    if not (
        isinstance(height_m, (int, float)) and np.isfinite(height_m) and height_m > 0
    ):
        raise ValueError(f"height_m must be a positive finite number, got {height_m!r}")
    if not (isinstance(mass_kg, (int, float)) and np.isfinite(mass_kg) and mass_kg > 0):
        raise ValueError(f"mass_kg must be a positive finite number, got {mass_kg!r}")
    valid_sex = {member.value for member in Sex}
    if sex not in valid_sex:
        raise ValueError(f"sex must be one of {sorted(valid_sex)}, got {sex!r}")
    if age_years is not None and not (
        isinstance(age_years, (int, float))
        and np.isfinite(float(age_years))
        and age_years >= 0
    ):
        raise ValueError(
            f"age_years must be a non-negative finite number, got {age_years!r}"
        )


def _segment_properties_from_ratios(
    *,
    name: str,
    body_part_id: str,
    ratios: SegmentRatios,
    height_m: float,
    mass_kg: float,
    method_name: str,
) -> SegmentProperties:
    """Construct one :class:`SegmentProperties` from a ratio row.

    The inertia tensor is built as a diagonal matrix in the
    principal-axis frame using the gyration-radius identity
    ``I_i = m * (k_i * L)^2``. The result is then validated by
    the :class:`SegmentProperties` post-init contract (symmetry,
    positive-definite, triangle inequality).
    """
    length = float(height_m) * float(ratios.length_ratio)
    mass = float(mass_kg) * float(ratios.mass_ratio)
    if length <= 0:
        raise ValueError(
            f"derived length for segment {name!r} must be positive, got {length!r}"
        )
    if mass <= 0:
        raise ValueError(
            f"derived mass for segment {name!r} must be positive, got {mass!r}"
        )

    com_z = length * float(ratios.com_proximal_ratio)
    com_xyz = np.array([0.0, 0.0, com_z], dtype=float)

    ix = mass * (float(ratios.gyration_sagittal) * length) ** 2
    iy = mass * (float(ratios.gyration_transverse) * length) ** 2
    iz = mass * (float(ratios.gyration_longitudinal) * length) ** 2
    inertia = np.diag([ix, iy, iz]).astype(float)

    return SegmentProperties(
        name=name,
        body_part_id=body_part_id,
        length_m=length,
        proximal_marker=None,
        distal_marker=None,
        mass_kg=mass,
        com_xyz_m=com_xyz,
        inertia_tensor=inertia,
        source_method=method_name,
        source_subject_height_m=float(height_m),
        source_subject_mass_kg=float(mass_kg),
    )


def build_subject_from_ratio_table(
    *,
    subject_id: str,
    height_m: float,
    mass_kg: float,
    sex: str,
    age_years: float | None,
    method_name: str,
    segment_classes: Mapping[str, SegmentRatios],
    segment_name_map: Mapping[str, str],
    normalize_mass: bool = True,
) -> SubjectAnthropometrics:
    """Materialise a :class:`SubjectAnthropometrics` from a ratio table.

    Parameters
    ----------
    subject_id, height_m, mass_kg, sex, age_years
        Subject metadata; validated against the same invariants
        enforced by :class:`SubjectAnthropometrics`.
    method_name
        Provenance string written to ``source_method`` on every
        emitted :class:`SegmentProperties` and on the parent
        :class:`SubjectAnthropometrics`.
    segment_classes
        Mapping ``class_id -> SegmentRatios`` listing the
        published ratios for each segment class.
    segment_name_map
        Mapping ``anatomical_name -> class_id``. The output
        produces one :class:`SegmentProperties` per key here.
    normalize_mass
        If True (default), rescale every segment mass so the
        per-subject sum equals ``mass_kg`` exactly. Published
        regression tables only sum to 1.0 when applied to the
        exact segmentation the original authors used; emitting
        a different segmentation (e.g. de Leva's 21 named
        segments including virtual shoulder/hip joints) requires
        a renormalisation step to preserve mass closure. The
        per-segment ratio is preserved in
        ``segment_classes`` for downstream introspection.

    Raises
    ------
    ValueError
        If any subject input is invalid, the ratio table is
        empty, or a name in ``segment_name_map`` references a
        class missing from ``segment_classes``.
    """
    _validate_subject_inputs(subject_id, height_m, mass_kg, sex, age_years)
    if not segment_classes:
        raise ValueError("segment_classes must be non-empty")
    if not segment_name_map:
        raise ValueError("segment_name_map must be non-empty")

    # First pass: validate name-map references and compute the
    # raw mass ratio sum so we can decide whether to normalise.
    raw_ratio_sum = 0.0
    for anatomical_name, class_id in segment_name_map.items():
        if class_id not in segment_classes:
            raise ValueError(
                f"segment_name_map references unknown class "
                f"{class_id!r} for {anatomical_name!r}"
            )
        raw_ratio_sum += float(segment_classes[class_id].mass_ratio)

    if raw_ratio_sum <= 0:
        raise ValueError(
            "sum of mass ratios across segment_name_map must be "
            f"positive, got {raw_ratio_sum!r}"
        )

    mass_scale = (1.0 / raw_ratio_sum) if normalize_mass else 1.0

    segments: list[tuple[str, SegmentProperties]] = []
    for anatomical_name, class_id in segment_name_map.items():
        ratios = segment_classes[class_id]
        if normalize_mass and mass_scale != 1.0:
            scaled_ratios = SegmentRatios(
                mass_ratio=ratios.mass_ratio * mass_scale,
                length_ratio=ratios.length_ratio,
                com_proximal_ratio=ratios.com_proximal_ratio,
                gyration_sagittal=ratios.gyration_sagittal,
                gyration_transverse=ratios.gyration_transverse,
                gyration_longitudinal=ratios.gyration_longitudinal,
            )
        else:
            scaled_ratios = ratios
        props = _segment_properties_from_ratios(
            name=anatomical_name,
            body_part_id=class_id,
            ratios=scaled_ratios,
            height_m=height_m,
            mass_kg=mass_kg,
            method_name=method_name,
        )
        segments.append((anatomical_name, props))

    return SubjectAnthropometrics(
        subject_id=subject_id,
        height_m=float(height_m),
        mass_kg=float(mass_kg),
        segments=tuple(segments),
        source_method=method_name,
        age_years=age_years,
        sex=sex,
    )
