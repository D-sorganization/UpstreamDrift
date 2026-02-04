"""Deformable Object Simulation.

This module provides simulation of deformable objects:
- Soft bodies (FEM-based)
- Cables and ropes
- Cloth and fabric
"""

from __future__ import annotations

from src.research.deformable.objects import (
    Cable,
    Cloth,
    DeformableObject,
    MaterialProperties,
    SoftBody,
)

__all__ = [
    "DeformableObject",
    "SoftBody",
    "Cable",
    "Cloth",
    "MaterialProperties",
]
