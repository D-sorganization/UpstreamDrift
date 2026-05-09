"""Readers for ingesting subject metadata from external file formats.

Each module exposes a ``read_*`` function returning a frozen,
format-specific metadata dataclass. Estimators downstream
(:class:`anthropometrics.contracts.Estimator`) consume these
dataclasses to build a fully-populated
:class:`anthropometrics.SubjectAnthropometrics`.
"""

from __future__ import annotations

from .c3d_subject_info import C3DSubjectMetadata, read_c3d_subject_metadata
from .urdf_inertial import read_urdf_inertial

__all__ = ["C3DSubjectMetadata", "read_c3d_subject_metadata", "read_urdf_inertial"]
