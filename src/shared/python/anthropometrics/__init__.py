"""Canonical anthropometrics data model and Protocols.

Foundation package for the anthropometrics EPIC. Provides:

* :class:`SegmentProperties` — frozen dataclass describing the
  inertial/dimensional properties of a single body segment.
* :class:`SubjectAnthropometrics` — frozen dataclass bundling
  all segments for one subject.
* :class:`Estimator`, :class:`Reader`, :class:`Writer`,
  :class:`EngineAdapter` — runtime-checkable Protocols for
  downstream subsystems.

All public objects are validated against physical-realisability
invariants at construction time (Design by Contract).
"""

from __future__ import annotations

from ._subject_anthropometrics import SubjectAnthropometrics
from ._types import Sex
from .contracts import EngineAdapter, Estimator, Reader, Writer
from .readers import C3DSubjectMetadata, read_c3d_subject_metadata
from .segment_properties import SegmentProperties

__all__ = [
    "C3DSubjectMetadata",
    "EngineAdapter",
    "Estimator",
    "Reader",
    "SegmentProperties",
    "Sex",
    "SubjectAnthropometrics",
    "Writer",
    "read_c3d_subject_metadata",
]
