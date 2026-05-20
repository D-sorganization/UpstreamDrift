from __future__ import annotations

import numpy as np

from src.shared.python.physics.terrain import Terrain


class TerrainGeometryGenerator:
    """Generate terrain geometry for physics engines.

    Creates meshes and heightfield data from terrain configuration
    for use in various physics engines.
    """

    def __init__(self, terrain: Terrain) -> None:
        """Initialize generator.

        Args:
            terrain: Terrain configuration
        """
        if terrain is None:
            raise ValueError("terrain must be provided")
        self.terrain = terrain

    def generate_mesh(self) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """Generate triangle mesh from terrain.

        Returns:
            Tuple of (vertices, triangles)
            - vertices: (N, 3) array of vertex positions
            - triangles: List of (i, j, k) vertex index tuples
        """
        elev = self.terrain.elevation
        n_rows, n_cols = elev.data.shape

        # Generate vertices
        vertices = []
        for j in range(n_rows):
            for i in range(n_cols):
                x = elev.origin_x + i * elev.resolution
                y = elev.origin_y + j * elev.resolution
                z = elev.data[j, i]
                vertices.append([x, y, z])

        vertices_array = np.array(vertices)

        # Generate triangles (2 per grid cell)
        triangles: list[tuple[int, int, int]] = []
        for j in range(n_rows - 1):
            for i in range(n_cols - 1):
                # Vertex indices for this cell
                v00 = j * n_cols + i
                v10 = j * n_cols + (i + 1)
                v01 = (j + 1) * n_cols + i
                v11 = (j + 1) * n_cols + (i + 1)

                # Two triangles per cell
                triangles.append((v00, v10, v11))
                triangles.append((v00, v11, v01))

        return vertices_array, triangles

    def generate_mujoco_hfield(self) -> tuple[np.ndarray, tuple[float, float]]:
        """Generate MuJoCo heightfield data.

        Returns:
            Tuple of (data, size)
            - data: 2D array of normalized heights [0, 1]
            - size: (width, length) in meters
        """
        elev = self.terrain.elevation

        # Normalize heights to [0, 1] range
        data = elev.data.copy()
        h_min = data.min()
        h_max = data.max()
        h_range = h_max - h_min if h_max > h_min else 1.0

        normalized = (data - h_min) / h_range

        size = (elev.width, elev.length)

        return normalized, size

    def generate_mujoco_xml(self, name: str = "terrain") -> str:
        """Generate MuJoCo XML snippet for terrain.

        Args:
            name: Name for the terrain geom

        Returns:
            XML string for inclusion in MuJoCo model
        """
        if name is None:
            raise ValueError("name must be provided")
        elev = self.terrain.elevation
        n_rows, n_cols = elev.data.shape

        # Calculate height range
        h_min = float(elev.data.min())
        h_max = float(elev.data.max())
        h_range = h_max - h_min if h_max > h_min else 0.1

        # Get average friction
        material = self.terrain.get_material(elev.width / 2, elev.length / 2)
        friction = material.friction_coefficient

        # Create XML
        # Note: Heightfield data would be written separately to a binary file
        # and referenced here. This generates the XML structure.
        xml_parts = [
            "<asset>",
            f'  <hfield name="{name}_hfield" nrow="{n_rows}" ncol="{n_cols}" size="{elev.width / 2} {elev.length / 2} {h_range} 0.1"/>',
            "</asset>",
            "<worldbody>",
            f'  <geom name="{name}" type="hfield" hfield="{name}_hfield" pos="{elev.width / 2} {elev.length / 2} {h_min}" friction="{friction} 0.005 0.0001"/>',
            "</worldbody>",
        ]

        return "\n".join(xml_parts)

    def generate_urdf_collision(self, name: str = "terrain") -> str:
        """Generate URDF collision geometry for terrain.

        For simplicity, generates a box approximation. Full mesh
        support would require external mesh file.

        Args:
            name: Name for the collision geometry

        Returns:
            URDF XML snippet
        """
        if name is None:
            raise ValueError("name must be provided")
        elev = self.terrain.elevation
        h_max = float(elev.data.max())
        h_min = float(elev.data.min())

        xml = f"""<link name="{name}">
  <collision>
    <origin xyz="{elev.width / 2} {elev.length / 2} {(h_max + h_min) / 2}" rpy="0 0 0"/>
    <geometry>
      <box size="{elev.width} {elev.length} {h_max - h_min + 0.1}"/>
    </geometry>
  </collision>
</link>"""

        return xml
