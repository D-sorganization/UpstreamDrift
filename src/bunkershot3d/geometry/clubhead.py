"""
Clubhead geometry generator for BunkerShot3D.
"""

from pathlib import Path
import numpy as np


class ClubheadGenerator:
    """Generates a simple parametric 3D mesh for a golf wedge."""

    def __init__(
        self,
        loft_deg: float = 60.0,
        bounce_deg: float = 10.0,
        width: float = 0.05,
        height: float = 0.04,
    ) -> None:
        """Initialize the generator with wedge parameters."""
        self.loft_rad = np.radians(loft_deg)
        self.bounce_rad = np.radians(bounce_deg)
        self.width = width
        self.height = height

    def generate_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate vertices and faces for a simple wedge.
        Returns:
            vertices (np.ndarray): Nx3 array of vertex coordinates.
            faces (np.ndarray): Mx3 array of face indices.
        """
        # Extremely simplified geometry: a wedge-shaped triangular prism
        # 6 vertices total:
        # Front face (inclined by loft)
        # Back face (vertical)
        # Bottom face (inclined by bounce)

        # Dimensions
        w2 = self.width / 2
        h = self.height

        # Basic profile in x-z plane (y is along the width)
        # Origin at leading edge (0,0,0)
        # x is target direction (backwards is +x)
        # z is up

        # Leading edge
        v0 = np.array([0, -w2, 0])
        v1 = np.array([0, w2, 0])

        # Top edge
        dx_top = h * np.tan(self.loft_rad)
        v2 = np.array([dx_top, -w2, h])
        v3 = np.array([dx_top, w2, h])

        # Trailing edge
        # Assume a sole width
        sole_width = 0.02
        dz_back = -sole_width * np.sin(self.bounce_rad)
        dx_back = sole_width * np.cos(self.bounce_rad)
        v4 = np.array([dx_back, -w2, dz_back])
        v5 = np.array([dx_back, w2, dz_back])

        vertices = np.vstack([v0, v1, v2, v3, v4, v5])

        # Faces (triangles), correctly winding (outward normal)
        faces = np.array(
            [
                # Face (loft)
                [0, 1, 3],
                [0, 3, 2],
                # Sole (bounce)
                [0, 4, 5],
                [0, 5, 1],
                # Back
                [4, 2, 3],
                [4, 3, 5],
                # Sides
                [0, 2, 4],
                [1, 5, 3],
            ]
        )

        return vertices, faces

    def export_stl(self, filepath: Path | str) -> None:
        """Export the generated mesh to an ASCII STL file."""
        vertices, faces = self.generate_mesh()

        with open(filepath, "w") as f:
            f.write("solid wedge\n")
            for face in faces:
                v0, v1, v2 = vertices[face]
                # Normal vector
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                if norm > 0:
                    n = n / norm

                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid wedge\n")
