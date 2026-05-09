"""Readers for ingesting subject metadata from external file formats.

Each module exposes a ``read_*`` function returning either a
format-specific metadata dataclass or a canonical anthropometric
record (e.g. :class:`SegmentProperties`). Estimators downstream
(:class:`anthropometrics.contracts.Estimator`) consume these
dataclasses to build a fully-populated
:class:`anthropometrics.SubjectAnthropometrics`.
"""

from __future__ import annotations

from .c3d_subject_info import C3DSubjectMetadata, read_c3d_subject_metadata
from .mjcf_body import read_mjcf_body
from .osim_body import read_osim_body
from .urdf_inertial import read_urdf_inertial

__all__ = [
    "C3DSubjectMetadata",
    "read_c3d_subject_metadata",
    "read_mjcf_body",
    "read_osim_body",
    "read_urdf_inertial",
]
