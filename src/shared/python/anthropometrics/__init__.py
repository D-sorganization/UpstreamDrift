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
from .persistence import (
    SCHEMA_VERSION,
    default_subjects_dir,
    load_subject,
    save_subject,
)
from .readers import C3DSubjectMetadata, read_c3d_subject_metadata
from .segment_properties import SegmentProperties

# Optional Qt UI surface — only re-exported when PyQt6 is installed.
# Resolved as a UNION: importing the package never fails even on
# PyQt6-less systems; consumers wishing to instantiate the panel are
# expected to install the ``gui-tools`` / ``gui-test`` extras.
try:  # pragma: no cover - presence depends on the install environment
    from .ui.segment_properties_panel import SegmentPropertiesPanel as _Panel
except Exception:  # pragma: no cover - PyQt6 missing or unloadable
    SegmentPropertiesPanel = None  # type: ignore[assignment]
else:
    SegmentPropertiesPanel = _Panel

__all__ = [
    "C3DSubjectMetadata",
    "EngineAdapter",
    "Estimator",
    "Reader",
    "SCHEMA_VERSION",
    "SegmentProperties",
    "SegmentPropertiesPanel",
    "Sex",
    "SubjectAnthropometrics",
    "Writer",
    "default_subjects_dir",
    "load_subject",
    "read_c3d_subject_metadata",
    "save_subject",
]
