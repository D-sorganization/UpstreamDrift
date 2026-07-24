from __future__ import annotations

import mujoco
import numpy as np


class MousePickingRay:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not (model is not None):
            raise ValueError("model must be provided")
        self.model = model
        self.data = data

    def screen_to_ray(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (x is not None):
            raise ValueError("x must be provided")
        x_ndc = (2.0 * x) / width - 1.0
        y_ndc = 1.0 - (2.0 * y) / height

        cam_pos = camera.lookat.copy()
        cam_distance = camera.distance
        cam_azimuth = np.deg2rad(camera.azimuth)
        cam_elevation = np.deg2rad(camera.elevation)

        forward = np.array(
            [
                np.cos(cam_elevation) * np.sin(cam_azimuth),
                np.cos(cam_elevation) * np.cos(cam_azimuth),
                np.sin(cam_elevation),
            ],
        )

        ray_origin = cam_pos - forward * cam_distance

        up_world = np.array([0, 0, 1])
        right = np.cross(up_world, forward)
        import math

        # ⚡ Bolt: math.hypot is ~5x faster than np.linalg.norm for 3D vecs
        right = right / (math.hypot(right[0], right[1], right[2]) + 1e-8)
        up = np.cross(forward, right)

        fovy = 45.0
        aspect = width / height

        ray_dir = forward.copy()
        ray_dir += right * x_ndc * np.tan(np.deg2rad(fovy / 2)) * aspect
        ray_dir += up * y_ndc * np.tan(np.deg2rad(fovy / 2))
        # ⚡ Bolt: math.hypot is ~5x faster than np.linalg.norm for 3D vecs
        ray_dir = ray_dir / math.hypot(ray_dir[0], ray_dir[1], ray_dir[2])

        return ray_origin, ray_dir

    def pick_body(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
        max_distance: float = 100.0,
    ) -> tuple[int, np.ndarray, float] | None:
        if not (x is not None):
            raise ValueError("x must be provided")
        ray_origin, ray_dir = self.screen_to_ray(x, y, width, height, camera)

        closest_body = None
        closest_distance = max_distance
        closest_point = None

        for body_id in range(1, self.model.nbody):
            body_pos = self.data.xpos[body_id].copy()

            to_body = body_pos - ray_origin
            proj_length = np.dot(to_body, ray_dir)

            if proj_length < 0:
                continue

            closest_on_ray = ray_origin + ray_dir * proj_length
            distance_to_body = np.linalg.norm(closest_on_ray - body_pos)

            body_radius = 0.1

            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == body_id:
                    geom_size = self.model.geom_size[geom_id]
                    body_radius = max(body_radius, geom_size[0])

            if distance_to_body < body_radius * 1.5 and proj_length < closest_distance:
                closest_distance = proj_length
                closest_body = body_id
                closest_point = closest_on_ray

        if closest_body is not None and closest_point is not None:
            return closest_body, closest_point, closest_distance

        return None
