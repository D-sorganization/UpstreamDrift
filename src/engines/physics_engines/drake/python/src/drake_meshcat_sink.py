"""Drake MeshCat visualizer sink adapting pydrake.geometry.Meshcat to MeshcatSink (FTO-5, #11290).

Design notes:
  - Explicit imports only from pydrake packages per CLAUDE.md guidelines.
  - When pydrake is unavailable (e.g. headless Windows nodes without Drake wheel),
    the module gracefully handles the absence so other packages can import without failure.
  - A cone uses MeshcatCone if exposed by pydrake.geometry; otherwise it falls back
    to Cylinder(radius_bottom_m, length_m) as documented.
  - The shared renderer poses a shape centred on the origin along local +y with
    the cone tip at +y (the three.js convention). Drake's Cylinder runs along
    local +z and MeshcatCone has its apex at the origin opening along +z, so
    set_transform appends a fixed per-shape correction (#11729).
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


def _z_to_y_correction() -> npt.NDArray[np.float64]:
    """Centred +z shape (Drake Cylinder) to the renderer's centred +y frame."""
    t = np.eye(4)
    t[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    return t


def _cone_correction(height_m: float) -> npt.NDArray[np.float64]:
    """MeshcatCone (apex at origin, base at +z height) to a centred cone, tip at +y."""
    t = np.eye(4)
    t[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    t[1, 3] = 0.5 * height_m
    return t


class DrakeMeshcatSink:
    """Adapts Drake's pydrake.geometry.Meshcat to the shared MeshcatSink protocol."""

    def __init__(self, meshcat: Any) -> None:
        self._meshcat = meshcat
        self._corrections: dict[str, npt.NDArray[np.float64]] = {}

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
        correction = _z_to_y_correction()
        if radius_top_m <= 1e-6:
            try:
                from pydrake.geometry import (
                    MeshcatCone,  # type: ignore[import-not-found]
                )

                shape = MeshcatCone(length_m, radius_bottom_m, radius_bottom_m)
                correction = _cone_correction(length_m)
            except ImportError:
                # Fallback: Cylinder approximation for Drake versions lacking MeshcatCone
                shape = Cylinder(radius_bottom_m, length_m)
        else:
            shape = Cylinder(radius_bottom_m, length_m)

        self._corrections[clean_path] = correction
        self._meshcat.SetObject(clean_path, shape, drake_rgba)

    def set_transform(self, path: str, matrix4x4: npt.NDArray[np.float64]) -> None:
        if not HAS_PYDRAKE:
            raise RuntimeError("pydrake is not installed in this environment")

        from pydrake.math import (  # type: ignore[import-not-found]
            RigidTransform,
            RotationMatrix,
        )

        clean_path = path.strip("/")
        pose = np.asarray(matrix4x4, dtype=np.float64)
        correction = self._corrections.get(clean_path)
        if correction is not None:
            pose = pose @ correction
        r = RotationMatrix(pose[:3, :3])
        p = pose[:3, 3]
        x_val = RigidTransform(r, p)  # type: ignore[arg-type]
        self._meshcat.SetTransform(clean_path, x_val)

    def delete(self, path: str) -> None:
        clean_path = path.strip("/")
        prefix = clean_path + "/"
        for key in [
            k for k in self._corrections if k == clean_path or k.startswith(prefix)
        ]:
            del self._corrections[key]
        self._meshcat.Delete(clean_path)
