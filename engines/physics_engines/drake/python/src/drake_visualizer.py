"""Drake Meshcat Visualization Helper."""

import typing

from pydrake.all import (
    Context,
    Cylinder,
    Meshcat,
    MultibodyPlant,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Sphere,
)
import numpy as np

FRAME_AXIS_LENGTH_M: typing.Final[float] = (
    0.2  # [m] Axle length for frame visualization
)
FRAME_AXIS_RADIUS_M: typing.Final[float] = 0.005  # [m] Axle radius
COM_SPHERE_RADIUS_M: typing.Final[float] = 0.015  # [m] COM marker radius


class DrakeVisualizer:
    """Helper class to manage advanced visualizations in Meshcat."""

    def __init__(self, meshcat: Meshcat, plant: MultibodyPlant) -> None:  # type: ignore[no-any-unimported]
        self.meshcat = meshcat
        self.plant = plant
        self.prefix = "visual_overlays"

        # Track active visualizations
        self.visible_frames: set[str] = set()
        self.visible_coms: set[str] = set()
        self.visible_ellipsoids: set[str] = set()

    def toggle_frame(self, body_name: str, visible: bool) -> None:  # noqa: FBT001
        """Toggle coordinate frame visualization for a body."""
        from numpy import pi

        path = f"{self.prefix}/frames/{body_name}"
        if visible:
            # Draw X, Y, Z axes
            length = FRAME_AXIS_LENGTH_M
            radius = FRAME_AXIS_RADIUS_M

            # X Axis (Red)
            self.meshcat.SetObject(
                f"{path}/x", Cylinder(radius, length), Rgba(1, 0, 0, 1)
            )
            X_x = RigidTransform(
                RotationMatrix.MakeYRotation(pi / 2), [length / 2, 0, 0]
            )
            self.meshcat.SetTransform(f"{path}/x", X_x)

            # Y Axis (Green)
            self.meshcat.SetObject(
                f"{path}/y", Cylinder(radius, length), Rgba(0, 1, 0, 1)
            )
            X_y = RigidTransform(
                RotationMatrix.MakeXRotation(-pi / 2), [0, length / 2, 0]
            )
            self.meshcat.SetTransform(f"{path}/y", X_y)

            # Z Axis (Blue)
            self.meshcat.SetObject(
                f"{path}/z", Cylinder(radius, length), Rgba(0, 0, 1, 1)
            )
            X_z = RigidTransform(RotationMatrix(), [0, 0, length / 2])
            self.meshcat.SetTransform(f"{path}/z", X_z)

            self.visible_frames.add(body_name)
        else:
            self.meshcat.Delete(path)
            self.visible_frames.discard(body_name)

    def update_frame_transforms(self, context: Context) -> None:  # type: ignore[no-any-unimported]
        """Update transforms of visible frames."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        for body_name in self.visible_frames:
            body = self.plant.GetBodyByName(body_name)
            X_WB = self.plant.EvalBodyPoseInWorld(plant_context, body)
            path = f"{self.prefix}/frames/{body_name}"
            self.meshcat.SetTransform(path, X_WB)

    def toggle_com(self, body_name: str, visible: bool) -> None:  # noqa: FBT001
        """Toggle Center of Mass visualization for a body."""
        path = f"{self.prefix}/coms/{body_name}"
        if visible:
            # Sphere for COM
            self.meshcat.SetObject(
                path, Sphere(COM_SPHERE_RADIUS_M), Rgba(1, 1, 0, 1)
            )  # Yellow
            self.visible_coms.add(body_name)
        else:
            self.meshcat.Delete(path)
            self.visible_coms.discard(body_name)

    def update_com_transforms(self, context: Context) -> None:  # type: ignore[no-any-unimported]
        """Update transforms of visible COMs."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        for body_name in self.visible_coms:
            body = self.plant.GetBodyByName(body_name)
            X_WB = self.plant.EvalBodyPoseInWorld(plant_context, body)

            # Get COM offset in body frame
            # SpatialInertia -> get_com() returns center of mass in body frame
            M_B = body.CalcSpatialInertiaInBodyFrame(plant_context)
            p_BCom = M_B.get_com()

            # Transform to World
            p_WCom = X_WB @ p_BCom

            self.meshcat.SetTransform(
                path=f"{self.prefix}/coms/{body_name}",
                X_ParentChild=RigidTransform(p_WCom),
            )

    def draw_ellipsoid(
        self,
        name: str,
        rotation_matrix: np.ndarray,
        radii: np.ndarray,
        position: np.ndarray,
        color: tuple[float, float, float, float]
    ) -> None:
        """Draw ellipsoid in Meshcat.

        Args:
            name: Unique identifier.
            rotation_matrix: 3x3 rotation (axes).
            radii: Length of semi-axes.
            position: Center position.
            color: (r, g, b, alpha)
        """
        path = f"{self.prefix}/ellipsoids/{name}"

        # Drake Meshcat supports generic SetObject with scaling in transform
        # Create a unit sphere
        self.meshcat.SetObject(path, Sphere(1.0), Rgba(*color))

        # Transform T = [Rot * Scale | Pos]
        # X_WE = RigidTransform(RotationMatrix(rotation_matrix), position)

        # However, RigidTransform does not support non-uniform scaling (shear).
        # Meshcat's SetTransform usually takes a 4x4 matrix or RigidTransform.
        # If Drake's python binding for SetTransform supports numpy array, we can use that.
        # Otherwise, we might be limited to uniform scaling if using RigidTransform.
        # Check PyDrake Meshcat SetTransform signature: (path, X_ParentChild) or (path, matrix)

        T = np.eye(4)
        T[:3, :3] = rotation_matrix @ np.diag(radii)
        T[:3, 3] = position

        # Pass numpy array directly if supported
        self.meshcat.SetTransform(path, T)
        self.visible_ellipsoids.add(name)

    def clear_ellipsoids(self) -> None:
        self.meshcat.Delete(f"{self.prefix}/ellipsoids")
        self.visible_ellipsoids.clear()

    def clear_all(self) -> None:
        """Clear all overlays."""
        self.meshcat.Delete(self.prefix)
        self.visible_frames.clear()
        self.visible_coms.clear()
        self.visible_ellipsoids.clear()
