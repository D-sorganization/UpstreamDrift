"""Canonical :class:`SubjectAnthropometrics` dataclass.

A :class:`SubjectAnthropometrics` instance bundles every
:class:`SegmentProperties` that describes one human subject,
along with the subject-level scalars required by downstream
estimators (height, mass, optional age and sex).

Like :class:`~.segment_properties.SegmentProperties`, the
dataclass is frozen and validates all invariants on
construction (Design by Contract).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._types import Sex
from .segment_properties import SegmentProperties


def _require_non_empty_str(value: object, label: str) -> None:
    """Raise ``ValueError`` if *value* is not a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")


def _require_positive(value: float, label: str) -> None:
    """Raise ``ValueError`` if *value* is not strictly positive."""
    import math  # local import keeps top-level symbol surface small

    if not (
        isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0
    ):
        raise ValueError(f"{label} must be a positive finite number, got {value!r}")


@dataclass(frozen=True)
class SubjectAnthropometrics:
    """All anthropometric data for a single subject, SI units only.

    Invariants (all enforced in :meth:`__post_init__`)
    --------------------------------------------------
    * ``subject_id`` and ``source_method`` are non-empty strings.
    * ``height_m`` and ``mass_kg`` are strictly positive finite floats.
    * ``segments`` is a non-empty tuple of ``(name, props)`` pairs
      where every ``props`` is a :class:`SegmentProperties` and
      every ``name`` is a unique non-empty string.
    * If provided, ``age_years`` is a non-negative finite float.
    * ``sex`` is one of ``"M"``, ``"F"``, ``"unspecified"``.
    """

    subject_id: str
    height_m: float
    mass_kg: float
    segments: tuple[tuple[str, SegmentProperties], ...]
    source_method: str
    age_years: float | None = None
    sex: str = Sex.UNSPECIFIED.value

    def __post_init__(self) -> None:
        _require_non_empty_str(self.subject_id, "subject_id")
        _require_non_empty_str(self.source_method, "source_method")
        _require_positive(self.height_m, "height_m")
        _require_positive(self.mass_kg, "mass_kg")

        if not isinstance(self.segments, tuple):
            raise ValueError(
                f"segments must be a tuple, got {type(self.segments).__name__}"
            )
        if not self.segments:
            raise ValueError("segments must be non-empty")

        seen: set[str] = set()
        for entry in self.segments:
            if not (isinstance(entry, tuple) and len(entry) == 2):
                raise ValueError(
                    f"segments entries must be (name, SegmentProperties) "
                    f"pairs, got {entry!r}"
                )
            seg_name, seg_props = entry
            _require_non_empty_str(seg_name, "segment name")
            if not isinstance(seg_props, SegmentProperties):
                raise ValueError(
                    f"segment {seg_name!r} must map to a SegmentProperties "
                    f"instance, got {type(seg_props).__name__}"
                )
            if seg_name in seen:
                raise ValueError(f"duplicate segment name: {seg_name!r}")
            seen.add(seg_name)

        if self.age_years is not None:
            import math

            if not (
                isinstance(self.age_years, (int, float))
                and math.isfinite(float(self.age_years))
                and self.age_years >= 0
            ):
                raise ValueError(
                    "age_years must be a non-negative finite number, "
                    f"got {self.age_years!r}"
                )

        valid_sex = {member.value for member in Sex}
        if self.sex not in valid_sex:
            raise ValueError(
                f"sex must be one of {sorted(valid_sex)}, got {self.sex!r}"
            )
