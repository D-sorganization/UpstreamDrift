"""
Heavy Integration Contracts — Trimesh
======================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: Trimesh can create meshes, compute inertia properties, perform
boolean operations, and export — as used by the humanoid character builder.
"""

from __future__ import annotations


import pytest

pytestmark = pytest.mark.live_simulation
