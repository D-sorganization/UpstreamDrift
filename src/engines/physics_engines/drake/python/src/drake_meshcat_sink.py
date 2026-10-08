"""Drake MeshCat visualizer sink adapting pydrake.geometry.Meshcat to MeshcatSink (FTO-5, #11290).

Design notes:
  - Explicit imports only from pydrake packages per CLAUDE.md guidelines.
  - When pydrake is unavailable (e.g. headless Windows nodes without Drake wheel),
    the module gracefully handles the absence so other packages can import without failure.
  - A cone uses MeshcatCone if exposed by pydrake.geometry; otherwise it falls back
    to Cylinder(radius_bottom_m, length_m) as documented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import pydrake

    HAS_PYDRAKE = bool(pydrake)
except Exception:
    HAS_PYDRAKE = False


class DrakeMeshcatSink:
    """Adapts Drake's pydrake.geometry.Meshcat to the shared MeshcatSink protocol."""

    def __init__(self, meshcat: Any) -> None:
        self._meshcat = meshcat

    def set_cylinder(
        self,
        path: str,
        length_m: float,
        radius_top_m: float,
        radius_bottom_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        if not HAS_PYDRAKE:
            raise RuntimeError("pydrake is not installed in this environment")

        from pydrake.geometry import Cylinder, Rgba  # type: ignore[import-not-found]

        # Use effective radius; if top is 0 (cone), check if MeshcatCone exists
        clean_path = path.strip("/")
        drake_rgba = Rgba(rgba[0], rgba[1], rgba[2], rgba[3])

        shape: Any
        if radius_top_m <= 1e-6:
            try:
                from pydrake.geometry import (
                    MeshcatCone,  # type: ignore[import-not-found]
                )

                shape = MeshcatCone(length_m, radius_bottom_m, radius_bottom_m)
            except ImportError:
                # Fallback: Cylinder approximation for Drake versions lacking MeshcatCone
                shape = Cylinder(radius_bottom_m, length_m)
        else:
            shape = Cylinder(radius_bottom_m, length_m)

        self._meshcat.SetObject(clean_path, shape, drake_rgba)

    def set_transform(self, path: str, matrix4x4: npt.NDArray[np.float64]) -> None:
        if not HAS_PYDRAKE:
            raise RuntimeError("pydrake is not installed in this environment")

        from pydrake.math import (  # type: ignore[import-not-found]
            RigidTransform,
            RotationMatrix,
        )

        clean_path = path.strip("/")
        r = RotationMatrix(matrix4x4[:3, :3])
        p = matrix4x4[:3, 3]
        x_val = RigidTransform(r, p)  # type: ignore[arg-type]
        self._meshcat.SetTransform(clean_path, x_val)

    def delete(self, path: str) -> None:
        clean_path = path.strip("/")
        self._meshcat.Delete(clean_path)
