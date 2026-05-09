"""Writers for emitting subject and segment data to external file formats.

Each module exposes a ``write_*`` function that converts a frozen
canonical dataclass (typically
:class:`anthropometrics.SegmentProperties` or
:class:`anthropometrics.SubjectAnthropometrics`) into a
format-specific representation. Writers are paired with the
corresponding reader in :mod:`anthropometrics.readers` so that
``read(write(x)) == x`` for any valid input.
"""

from __future__ import annotations

from .mjcf_body import write_mjcf_body
from .osim_body import write_osim_body
from .urdf_inertial import write_urdf_inertial

__all__ = ["write_mjcf_body", "write_osim_body", "write_urdf_inertial"]
