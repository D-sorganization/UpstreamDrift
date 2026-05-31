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
from ._subject_anthropometrics import SubjectAnthropometrics  # noqa: F401
from ._types import Sex  # noqa: F401
from .addbiomechanics_priors import (  # noqa: F401
    PRIOR_SCHEMA_VERSION,
    CalibrationInertiaPriorSet,
    InertiaPriorParameter,
    build_inertia_priors_from_subject,
    load_addbiomechanics_inertia_priors,
    save_inertia_priors,
)
from .contracts import EngineAdapter, Estimator, Reader, Writer  # noqa: F401
from .engine_adapters import ADAPTER_REGISTRY  # noqa: F401
from .persistence import (
    SCHEMA_VERSION,  # noqa: F401
    default_subjects_dir,  # noqa: F401
    load_subject,  # noqa: F401
    save_subject,  # noqa: F401
)
from .pipeline import run_pipeline  # noqa: F401
from .readers import C3DSubjectMetadata, read_c3d_subject_metadata, read_mjcf_body  # noqa: F401
from .segment_properties import SegmentProperties  # noqa: F401
from .writers import write_mjcf_body  # noqa: F401

# Optional Qt UI surface — only re-exported when PyQt6 is installed.
# Resolved as a UNION: importing the package never fails even on
# PyQt6-less systems; consumers wishing to instantiate the panel are
# expected to install the ``gui-tools`` / ``gui-test`` extras.
try:  # pragma: no cover - presence depends on the install environment
    from .ui.segment_properties_panel import SegmentPropertiesPanel as _Panel
except Exception:  # pragma: no cover - PyQt6 missing or unloadable  # noqa: BLE001
    SegmentPropertiesPanel = None  # type: ignore[assignment]
else:
    SegmentPropertiesPanel = _Panel

try:  # pragma: no cover - presence depends on the install environment
    from .ui.calibration_dialog import SubjectCalibrationDialog as _Dialog
except Exception:  # pragma: no cover - PyQt6 missing or unloadable  # noqa: BLE001
    SubjectCalibrationDialog = None  # type: ignore[assignment]
else:
    SubjectCalibrationDialog = _Dialog

# UNION resolution: combine top-level public names with the
# engine_adapters subpackage's own __all__ so e.g.
# ``from anthropometrics import DrakeAdapter`` works.
_LOCAL_ALL = [
    "ADAPTER_REGISTRY",
    "CalibrationInertiaPriorSet",
    "C3DSubjectMetadata",
    "EngineAdapter",
    "Estimator",
    "InertiaPriorParameter",
    "PRIOR_SCHEMA_VERSION",
    "Reader",
    "SCHEMA_VERSION",
    "SegmentProperties",
    "SegmentPropertiesPanel",
    "Sex",
    "SubjectAnthropometrics",
    "SubjectCalibrationDialog",
    "Writer",
    "build_inertia_priors_from_subject",
    "default_subjects_dir",
    "engine_adapters",
    "load_addbiomechanics_inertia_priors",
    "load_subject",
    "read_c3d_subject_metadata",
    "read_mjcf_body",
    "run_pipeline",
    "save_inertia_priors",
    "save_subject",
    "write_mjcf_body",
]
__all__ = sorted(set(_LOCAL_ALL) | set(engine_adapters.__all__))  # noqa: PLE0605

# Re-export every concrete adapter at the top level for ergonomic
# ``from anthropometrics import DrakeAdapter`` imports.
for _name in engine_adapters.__all__:
    globals()[_name] = getattr(engine_adapters, _name)
del _name
