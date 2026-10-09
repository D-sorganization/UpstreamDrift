"""Rigid delivery, finite-head impact, and native flight for matched shots.

The club frame uses native +X target, +Y left, +Z up. Contact horizontal is
positive toward the toe and vertical is positive toward the crown. This is a
conditional physical model whose head properties are explicit assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.physics.ball_simulator import BallFlightSimulator
from src.shared.python.physics.rigid_contact import (
    RigidContactInput,
    solve_rigid_contact,
)

from .club_geometry import club_delivery
from .matched_config import MatchedConfig
from .physics import ShotOutcome, outcome_from_launch


class UnsupportedShot(ValueError):
    """A sampled delivery lies outside the stated model domain."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class MatchedDeliveryOutcome:
    """A first-contact shot and its actual central delivered face geometry."""

    outcome: ShotOutcome
    delivered_face_deg: float
    delivered_loft_deg: float
    contact_face_deg: float
    contact_loft_deg: float


class MatchedShotPhysics:
    """Map explicit club delivery and bounded face contact into ball flight."""

    def __init__(self) -> None:
        self._flight = BallFlightSimulator()

    def simulate(
        self,
        *,
        face_deg: float,
        path_deg: float,
        config: MatchedConfig,
        sample_trajectory: bool = False,
        nominal_face_deg: float = 0.0,
    ) -> ShotOutcome:
        """Implement the nominal-face calibration protocol without rotating carry."""
        # Calibration varies nominal yaw directly. Shaft twist is independent.
        if not math.isfinite(nominal_face_deg) or abs(nominal_face_deg) > 30:
            raise UnsupportedShot("delivery_out_of_range")
        return self.simulate_delivery(
            face_deg,
            path_deg,
            config,
            sample_trajectory=sample_trajectory,
        ).outcome

    def simulate_delivery(
        self,
        nominal_face_deg: float,
        path_deg: float,
        config: MatchedConfig,
        *,
        shaft_rotation_deg: float = 0.0,
        contact_horizontal_m: float = 0.0,
        contact_vertical_m: float = 0.0,
        sample_trajectory: bool = False,
    ) -> MatchedDeliveryOutcome:
        """Simulate one delivery; return finite first-contact carry or raise.

        Geometric or nonairborne samples raise ``UnsupportedShot`` for runner
        population accounting. Numerical solver and flight failures propagate.
        """
        if not isinstance(config, MatchedConfig):
            raise TypeError("config must be MatchedConfig")
        values = (
            nominal_face_deg,
            path_deg,
            shaft_rotation_deg,
            contact_horizontal_m,
            contact_vertical_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise UnsupportedShot("delivery_nonfinite")
        if any(abs(value) > 30 for value in values[:3]):
            raise UnsupportedShot("delivery_out_of_range")
        h, v = contact_horizontal_m, contact_vertical_m
        if abs(h) > config.face_half_width_m or abs(v) > config.face_half_height_m:
            raise UnsupportedShot("off_face")

        delivered = club_delivery(
            config.loft_deg,
            config.lie_deg,
            config.shaft_lean_deg,
            nominal_face_deg,
            shaft_rotation_deg,
        )
        normal = np.asarray(delivered.normal)
        heel = np.asarray(delivered.heel_axis)
        up = np.asarray(delivered.up_axis)
        frame = np.column_stack((normal, heel, up))
        inertia = frame @ np.diag(config.inertia_diagonal_kg_m2) @ frame.T

        bulge = h * h / config.bulge_radius_m if config.bulge_radius_m else 0.0
        roll = v * v / config.roll_radius_m if config.roll_radius_m else 0.0
        lever = (config.cg_depth_m - 0.5 * (bulge + roll)) * normal - h * heel + v * up
        surface_normal = normal.copy()
        if config.bulge_radius_m:
            surface_normal -= h / config.bulge_radius_m * heel
        if config.roll_radius_m:
            surface_normal += v / config.roll_radius_m * up
        surface_normal /= np.linalg.norm(surface_normal)

        path = math.radians(path_deg)
        attack = math.radians(config.attack_angle_deg)
        velocity = config.club_speed_mps * np.array(
            [
                math.cos(attack) * math.cos(path),
                -math.cos(attack) * math.sin(path),
                math.sin(attack),
            ]
        )
        if float(np.dot(velocity, surface_normal)) <= 0:
            raise UnsupportedShot("delivery_nonapproaching")
        contact = solve_rigid_contact(
            RigidContactInput(
                club_velocity_mps=velocity,
                club_angular_velocity_rad_s=np.zeros(3),
                ball_velocity_mps=np.zeros(3),
                ball_angular_velocity_rad_s=np.zeros(3),
                normal=surface_normal,
                club_lever_m=lever,
                club_inertia_world_kg_m2=inertia,
                club_mass_kg=config.clubhead_mass_kg,
                restitution=config.cor,
                friction_coefficient=config.friction_coefficient,
            )
        )
        ball_velocity = np.asarray(contact.ball_velocity_mps, dtype=float)
        if not np.all(np.isfinite(ball_velocity)):
            raise RuntimeError("contact produced nonfinite ball velocity")
        if ball_velocity[2] <= 0:
            raise UnsupportedShot("nonairborne_launch")
        outcome = outcome_from_launch(
            ball_velocity,
            np.asarray(contact.ball_angular_velocity_rad_s, dtype=float),
            flight=self._flight,
            max_time_s=config.max_time_s,
            dt_s=config.dt_s,
            sample_trajectory=sample_trajectory,
        )
        return MatchedDeliveryOutcome(
            outcome=outcome,
            delivered_face_deg=delivered.face_deg,
            delivered_loft_deg=delivered.loft_deg,
            contact_face_deg=math.degrees(
                math.atan2(-surface_normal[1], surface_normal[0])
            ),
            contact_loft_deg=math.degrees(
                math.atan2(surface_normal[2], math.hypot(*surface_normal[:2]))
            ),
        )
