"""Writers for emitting subject and segment data to external file formats.

Each module exposes a ``write_*`` function. Writers are paired with
the corresponding reader in :mod:`anthropometrics.readers` so that
``read(write(x)) == x`` for any valid input.
"""

from __future__ import annotations

from .urdf_inertial import write_urdf_inertial

__all__ = ["write_urdf_inertial"]
