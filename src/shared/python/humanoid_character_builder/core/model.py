"""
Data structures for the humanoid model.

This module defines the classes used to represent the humanoid model,
including links, joints, and the model itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from humanoid_character_builder.contracts import precondition
from humanoid_character_builder.mesh.inertia_calculator import InertiaResult
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R


@dataclass
class GeneratedLink:
    """Generated URDF link data."""

    name: str
    mass: float
    inertia: InertiaResult
    visual_geometry: dict[str, Any]
    collision_geometry: dict[str, Any] | None
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]


@dataclass
class GeneratedJoint:
    """Generated URDF joint data."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    limits: dict[str, float] | None
    dynamics: dict[str, float]


@dataclass
class SupportPolygon:
    """Represents the support polygon of the model."""

    vertices: list[tuple[float, float]]  # (x, y) coordinates of vertices

    def contains(self, point: tuple[float, float]) -> bool:
        """Check if a point is inside the support polygon."""
        if not (point is not None):
            raise ValueError("point must be provided")
        if not (point is not None):
            raise ValueError("point must be provided")
        if len(self.vertices) < 3:
            return False

        # Using ray casting algorithm or checking sign of cross products
        # Since it's a convex polygon (hull), we can check if point is on the same side of all edges

        px, py = point
        n = len(self.vertices)

        # Check winding order first or just assume consistency?
        # A safer way for convex polygon is to check cross product signs

        prev_cross: float = 0.0
        for i in range(n):
            p1x, p1y = self.vertices[i]
            p2x, p2y = self.vertices[(i + 1) % n]

            edge_x, edge_y = p2x - p1x, p2y - p1y
            diff_x, diff_y = px - p1x, py - p1y

            cross = edge_x * diff_y - edge_y * diff_x

            if cross != 0:
                if prev_cross == 0:
                    prev_cross = cross
                elif (cross > 0) != (prev_cross > 0):
                    return False

        return True

    def distance_to_edge(self, point: tuple[float, float]) -> float:
        """Compute minimum distance from point to the polygon edge."""
        if not (point is not None):
            raise ValueError("point must be provided")
        if not (point is not None):
            raise ValueError("point must be provided")
        if not self.contains(point):
            return -1.0  # Or positive distance to polygon? Convention usually margin > 0 is stable.
            # If outside, negative margin.

        px, py = point
        n = len(self.vertices)
        min_dist = float("inf")

        # Distance from point to line segment P1-P2
        for i in range(n):
            p1 = np.array(self.vertices[i])
            p2 = np.array(self.vertices[(i + 1) % n])
            p = np.array([px, py])

            # Project p onto line containing p1-p2
            l2 = np.sum((p1 - p2) ** 2)
            if l2 == 0:
                dist = np.linalg.norm(p - p1)
            else:
                t = max(0, min(1, np.dot(p - p1, p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                dist = np.linalg.norm(p - projection)

            if dist < min_dist:
                min_dist = float(dist)

        return min_dist


class HumanoidModel:
    """Representation of the complete humanoid model."""

    @precondition(
        lambda links, joints: len(links) > 0,
        "Model must have at least one link",
    )
    @precondition(
        lambda links, joints, root_link_name: root_link_name in links,
        "Root link must exist in links",
    )
    def __init__(
        self,
        links: dict[str, GeneratedLink],
        joints: list[GeneratedJoint],
        root_link_name: str = "pelvis",
    ):
        if not (links is not None):
            raise ValueError("links must be provided")
        if not (links is not None):
            raise ValueError("links must be provided")
        self.links = links
        self.joints = joints
        self.root_link_name = root_link_name

        # Build tree structure
        self.children_map: dict[str, list[GeneratedJoint]] = {name: [] for name in links}
        self.joint_map: dict[str, GeneratedJoint] = {j.name: j for j in joints}

        for joint in joints:
            if joint.parent in self.children_map:
                self.children_map[joint.parent].append(joint)

    def get_global_transforms(self) -> dict[str, np.ndarray]:
        """
        Compute global transforms for all links (assuming zero joint angles).

        Returns:
            Dictionary mapping link name to 4x4 transformation matrix.
        """
        transforms = {}

        # Stack: (link_name, parent_transform)
        # Root transform is identity (or aligned with world frame)
        # Usually pelvis is at some height?
        # The URDF generator sets pelvis origin? No, the generator sets origins relative to parent.
        # The root link doesn't have a parent joint in the list (or it's connected to world).
        # In URDF, the first link is the root.

        stack = [(self.root_link_name, np.eye(4))]

        while stack:
            link_name, parent_transform = stack.pop()
            transforms[link_name] = parent_transform

            for joint in self.children_map.get(link_name, []):
                # T_parent_child = T_joint_origin
                # T_global_child = T_global_parent * T_joint_origin

                xyz = joint.origin_xyz
                rpy = joint.origin_rpy

                T_joint = np.eye(4)
                T_joint[:3, 3] = xyz
                T_joint[:3, :3] = R.from_euler("xyz", rpy).as_matrix()

                child_transform = parent_transform @ T_joint
                stack.append((joint.child, child_transform))

        return transforms

    def compute_center_of_mass(self) -> tuple[float, float, float]:
        """
        Compute the global center of mass of the model.

        Returns:
            (x, y, z) global coordinates of COM.
        """
        transforms = self.get_global_transforms()

        total_mass = 0.0
        weighted_pos = np.zeros(3)

        for link_name, link in self.links.items():
            if link_name not in transforms:
                continue

            T_global = transforms[link_name]

            # link.origin_xyz is the COM position in the link frame
            com_local = np.array(link.origin_xyz)

            # Transform to global
            # p_global = R * p_local + t
            R_global = T_global[:3, :3]
            t_global = T_global[:3, 3]

            com_global = R_global @ com_local + t_global

            weighted_pos += com_global * link.mass
            total_mass += link.mass

        if total_mass == 0:
            return (0.0, 0.0, 0.0)

        com = weighted_pos / total_mass
        return (float(com[0]), float(com[1]), float(com[2]))

    def compute_support_polygon(self) -> SupportPolygon:
        """
        Compute the support polygon projected on the ground (XY plane).
        Assumes feet are the support.
        """
        transforms = self.get_global_transforms()
        feet_links = self._identify_feet_links(transforms)
        points = self._collect_footprint_points(feet_links, transforms)
        return self._create_support_polygon_from_points(points)

    def _identify_feet_links(self, transforms: dict[str, np.ndarray]) -> list[str]:
        """Identify links that form the support base (feet)."""
        # Heuristic: links containing "foot" in name
        if not (transforms is not None):
            raise ValueError("transforms must be provided")
        if not (transforms is not None):
            raise ValueError("transforms must be provided")
        feet_links = [name for name in self.links if "foot" in name]

        if not feet_links:
            # Fallback: find lowest links
            sorted_links = sorted(
                self.links.keys(),
                key=lambda name: (transforms[name][2, 3] if name in transforms else float("inf")),
            )
            feet_links = sorted_links[:2]  # Take lowest 2

        return feet_links

    def _collect_footprint_points(
        self,
        feet_links: list[str],
        transforms: dict[str, np.ndarray],
    ) -> list[tuple[float, float]]:
        """Collect 2D footprint points from feet links."""
        if not (feet_links is not None):
            raise ValueError("feet_links must be provided")
        if not (feet_links is not None):
            raise ValueError("feet_links must be provided")
        points: list[tuple[float, float]] = []

        for link_name in feet_links:
            if link_name not in transforms:
                continue

            T_global = transforms[link_name]
            link = self.links[link_name]
            footprint = self._compute_link_footprint(link)

            # Transform points to global XY
            for pt in footprint:
                pt_global = T_global[:3, :3] @ np.array(pt) + T_global[:3, 3]
                points.append((float(pt_global[0]), float(pt_global[1])))

        return points

    def _compute_link_footprint(self, link: GeneratedLink) -> list[list[float]]:
        """Compute local footprint points for a link based on its geometry."""
        if not (link is not None):
            raise ValueError("link must be provided")
        if not (link is not None):
            raise ValueError("link must be provided")
        geom = link.collision_geometry or link.visual_geometry

        if geom and geom.get("type") == "box":
            return self._compute_box_footprint(geom["size"])
        if geom and geom.get("type") in ("cylinder", "capsule"):
            return self._compute_cylinder_footprint(geom["radius"], geom["length"])
        # Just use COM
        return [list(link.origin_xyz)]

    def _compute_box_footprint(self, size: tuple[float, float, float]) -> list[list[float]]:
        """Compute footprint points for box geometry (8 corners)."""
        if not (size is not None):
            raise ValueError("size must be provided")
        if not (size is not None):
            raise ValueError("size must be provided")
        sx, sy, sz = size
        footprint = []
        for dx in [-sx / 2, sx / 2]:
            for dy in [-sy / 2, sy / 2]:
                for dz in [-sz / 2, sz / 2]:
                    footprint.append([dx, dy, dz])
        return footprint

    def _compute_cylinder_footprint(self, radius: float, length: float) -> list[list[float]]:
        """Compute footprint points for cylinder/capsule geometry."""
        if not (radius is not None):
            raise ValueError("radius must be provided")
        if not (radius is not None):
            raise ValueError("radius must be provided")
        footprint = []
        for theta in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            footprint.append([radius * np.cos(theta), radius * np.sin(theta), -length / 2])
            footprint.append([radius * np.cos(theta), radius * np.sin(theta), length / 2])
        return footprint

    def _create_support_polygon_from_points(
        self, points: list[tuple[float, float]]
    ) -> SupportPolygon:
        """Create support polygon from 2D points using convex hull."""
        if not (points is not None):
            raise ValueError("points must be provided")
        if not (points is not None):
            raise ValueError("points must be provided")
        if len(points) < 3:
            return SupportPolygon(points)

        points_array = np.array(points)
        try:
            hull = ConvexHull(points_array)
            vertices = points_array[hull.vertices]
            return SupportPolygon([tuple(p) for p in vertices])
        except (KeyError, ValueError, TypeError):
            return SupportPolygon(points)
