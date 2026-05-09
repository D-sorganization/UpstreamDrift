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

from . import engine_adapters as engine_adapters
from ._subject_anthropometrics import SubjectAnthropometrics
from ._types import Sex
from .contracts import EngineAdapter, Estimator, Reader, Writer
from .engine_adapters import ADAPTER_REGISTRY
from .persistence import (
    SCHEMA_VERSION,
    default_subjects_dir,
    load_subject,
    save_subject,
)
from .readers import C3DSubjectMetadata, read_c3d_subject_metadata, read_mjcf_body
from .segment_properties import SegmentProperties
from .writers import write_mjcf_body

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

# UNION resolution: combine top-level public names with the
# engine_adapters subpackage's own __all__ so e.g.
# ``from anthropometrics import DrakeAdapter`` works.
_LOCAL_ALL = [
    "ADAPTER_REGISTRY",
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
    "engine_adapters",
    "load_subject",
    "read_c3d_subject_metadata",
    "read_mjcf_body",
    "save_subject",
    "write_mjcf_body",
]
__all__ = sorted(set(_LOCAL_ALL) | set(engine_adapters.__all__))  # noqa: PLE0605

# Re-export every concrete adapter at the top level for ergonomic
# ``from anthropometrics import DrakeAdapter`` imports.
for _name in engine_adapters.__all__:
    globals()[_name] = getattr(engine_adapters, _name)
del _name
