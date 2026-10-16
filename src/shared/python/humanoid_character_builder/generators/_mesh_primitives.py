"""Compatibility re-export of the primitive mesh generator.

There is exactly one ``PrimitiveMeshGenerator`` implementation, and it lives in
``_primitive_generator`` (issue #12706), which is what the public
``mesh_generator`` facade and the MakeHuman/SMPL-X fallbacks import. This module
used to hold a second copy of the class. Its only behavioural difference -- an
explicit ``solver_status="failure"`` when trimesh is missing -- was ported.

Import via ``mesh_generator`` (the public module).
"""

from __future__ import annotations

from ._primitive_generator import PrimitiveMeshGenerator

__all__ = ["PrimitiveMeshGenerator"]
