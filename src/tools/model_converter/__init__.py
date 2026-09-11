"""Cross-engine multibody model converter from canonical specification.

Exports validated URDF, MJCF, and Simscape models from `golfer_canonical.yaml`.
"""

from __future__ import annotations

from tools.model_converter.matlab_exporter import export_matlab_parameters
from tools.model_converter.mjcf_exporter import export_mjcf
from tools.model_converter.schema_validator import (
    CanonicalModel,
    Geometry,
    Inertia,
    Joint,
    JointDof,
    RootBody,
    Segment,
    Transform,
    ValidationError,
    validate_canonical_model,
)
from tools.model_converter.urdf_exporter import export_urdf

__all__ = [
    "CanonicalModel",
    "Geometry",
    "Inertia",
    "Joint",
    "JointDof",
    "RootBody",
    "Segment",
    "Transform",
    "ValidationError",
    "export_matlab_parameters",
    "export_mjcf",
    "export_urdf",
    "validate_canonical_model",
]
