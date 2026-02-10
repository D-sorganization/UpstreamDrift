"""
Pendulum Putter Model Builder.

Creates a Perfy-style pendulum putting robot based on Dave Pelz's design.
The model features a rigid stand with pendulum arms holding an
interchangeable putter club.

Design principles (Pragmatic Programmer):
- Small, focused functions
- Flexible configuration via dataclasses
- Reversible design (club interchangeability)
- Easy to maintain and extend

Physics:
- Single DOF revolute joint at shoulder (pendulum pivot)
- Pure pendulum motion in X-Z plane (Y-axis rotation)
- Low damping for realistic swing behavior
- Configurable arm length affects natural frequency: omega = sqrt(g*m*d/I)

Usage:
    builder = PendulumPutterModelBuilder(arm_length_m=0.4)
    result = builder.build()
    builder.save("pendulum_putter.urdf")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_generation.builders.base_builder import BaseURDFBuilder, BuildResult
from model_generation.core.constants import (
    GRAVITY_M_S2,
    INTERMEDIATE_LINK_MASS,
)
from model_generation.core.types import (
    Geometry,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)

# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class ClubConfig:
    """Configuration for interchangeable golf club.

    Allows customization of club dimensions and mass properties
    for different club types (putter, wedge, etc.).

    Attributes:
        grip_length_m: Length of grip section in meters
        grip_radius_m: Radius of grip in meters
        grip_mass_kg: Mass of grip in kg
        shaft_length_m: Length of shaft in meters
        shaft_radius_m: Radius of shaft in meters
        shaft_mass_kg: Mass of shaft in kg
        head_mass_kg: Mass of club head in kg
        head_dimensions_m: (length, width, height) of head in meters
    """

    grip_length_m: float = 0.20
    grip_radius_m: float = 0.012
    grip_mass_kg: float = 0.040
    shaft_length_m: float = 0.80
    shaft_radius_m: float = 0.005
    shaft_mass_kg: float = 0.060
    head_mass_kg: float = 0.340
    head_dimensions_m: tuple[float, float, float] = (0.10, 0.06, 0.025)

    def total_mass(self) -> float:
        """Calculate total club mass."""
        return self.grip_mass_kg + self.shaft_mass_kg + self.head_mass_kg

    def total_length(self) -> float:
        """Calculate total club length."""
        return self.grip_length_m + self.shaft_length_m


@dataclass
class StandConfig:
    """Configuration for the robot stand/base.

    Attributes:
        base_mass_kg: Mass of the base plate
        base_dimensions_m: (x, y, z) dimensions of base
        post_height_m: Height of vertical post
        post_radius_m: Radius of vertical post
        post_mass_kg: Mass of vertical post
    """

    base_mass_kg: float = 8.0
    base_dimensions_m: tuple[float, float, float] = (0.30, 0.30, 0.05)
    post_height_m: float = 0.80
    post_radius_m: float = 0.025
    post_mass_kg: float = 1.5


@dataclass
class PendulumConfig:
    """Configuration for pendulum arm.

    Attributes:
        arm_length_m: Length of pendulum arm
        arm_radius_m: Radius of arm cylinder
        arm_mass_kg: Mass of arm
        mount_mass_kg: Mass of shoulder mount
        damping: Joint damping coefficient
        friction: Joint friction coefficient
        swing_limit_rad: Maximum swing angle from vertical
    """

    arm_length_m: float = 0.40
    arm_radius_m: float = 0.015
    arm_mass_kg: float = 0.3
    mount_mass_kg: float = 0.5
    damping: float = 0.05
    friction: float = 0.01
    swing_limit_rad: float = math.pi / 4  # 45 degrees


# =============================================================================
# Link Creation Functions (Small, focused, testable)
# =============================================================================


def create_world_link() -> Link:
    """Create the world reference frame link."""
    return Link(
        name="world",
        inertia=Inertia(
            ixx=1e-9,
            iyy=1e-9,
            izz=1e-9,
            mass=INTERMEDIATE_LINK_MASS,
        ),
    )


def create_base_link(config: StandConfig) -> Link:
    """Create the heavy base plate link."""
    x, y, z = config.base_dimensions_m
    return Link(
        name="base_link",
        inertia=Inertia.from_box(config.base_mass_kg, x, y, z),
        visual_geometry=Geometry.box(x, y, z),
        visual_origin=Origin.from_position(0, 0, z / 2),
        visual_material=Material("base_material", (0.3, 0.3, 0.35, 1.0)),
        collision_geometry=Geometry.box(x, y, z),
        collision_origin=Origin.from_position(0, 0, z / 2),
    )


def create_vertical_post_link(config: StandConfig) -> Link:
    """Create the vertical support post link."""
    return Link(
        name="vertical_post",
        inertia=Inertia.from_cylinder(
            config.post_mass_kg,
            config.post_radius_m,
            config.post_height_m,
            axis="z",
        ),
        visual_geometry=Geometry.cylinder(config.post_radius_m, config.post_height_m),
        visual_origin=Origin.from_position(0, 0, config.post_height_m / 2),
        visual_material=Material.metal(),
        collision_geometry=Geometry.cylinder(
            config.post_radius_m, config.post_height_m
        ),
        collision_origin=Origin.from_position(0, 0, config.post_height_m / 2),
    )


def create_shoulder_mount_link(config: PendulumConfig) -> Link:
    """Create the shoulder mount (pendulum pivot housing) link."""
    mount_radius = 0.04
    mount_length = 0.08
    return Link(
        name="shoulder_mount",
        inertia=Inertia.from_cylinder(
            config.mount_mass_kg, mount_radius, mount_length, axis="y"
        ),
        visual_geometry=Geometry.cylinder(mount_radius, mount_length),
        visual_origin=Origin(xyz=(0, 0, 0), rpy=(math.pi / 2, 0, 0)),
        visual_material=Material("mount_material", (0.5, 0.5, 0.55, 1.0)),
        collision_geometry=Geometry.cylinder(mount_radius, mount_length),
        collision_origin=Origin(xyz=(0, 0, 0), rpy=(math.pi / 2, 0, 0)),
    )


def create_pendulum_arm_link(config: PendulumConfig) -> Link:
    """Create the pendulum arm link."""
    return Link(
        name="pendulum_arm",
        inertia=Inertia.from_cylinder(
            config.arm_mass_kg,
            config.arm_radius_m,
            config.arm_length_m,
            axis="z",
        ),
        visual_geometry=Geometry.cylinder(config.arm_radius_m, config.arm_length_m),
        visual_origin=Origin.from_position(0, 0, -config.arm_length_m / 2),
        visual_material=Material("arm_material", (0.4, 0.4, 0.45, 1.0)),
        collision_geometry=Geometry.cylinder(config.arm_radius_m, config.arm_length_m),
        collision_origin=Origin.from_position(0, 0, -config.arm_length_m / 2),
    )


def create_club_mount_link() -> Link:
    """Create the club attachment point link."""
    mount_size = 0.03
    return Link(
        name="club_mount",
        inertia=Inertia.from_box(0.1, mount_size, mount_size, mount_size),
        visual_geometry=Geometry.box(mount_size, mount_size, mount_size),
        visual_origin=Origin.from_position(0, 0, 0),
        visual_material=Material("mount_material", (0.2, 0.2, 0.25, 1.0)),
        collision_geometry=Geometry.box(mount_size, mount_size, mount_size),
        collision_origin=Origin.from_position(0, 0, 0),
    )


# =============================================================================
# Club Link Creation Functions
# =============================================================================


def create_club_grip_link(config: ClubConfig) -> Link:
    """Create the club grip link."""
    return Link(
        name="club_grip",
        inertia=Inertia.from_cylinder(
            config.grip_mass_kg,
            config.grip_radius_m,
            config.grip_length_m,
            axis="z",
        ),
        visual_geometry=Geometry.cylinder(config.grip_radius_m, config.grip_length_m),
        visual_origin=Origin.from_position(0, 0, -config.grip_length_m / 2),
        visual_material=Material("grip_material", (0.1, 0.1, 0.1, 1.0)),
        collision_geometry=Geometry.cylinder(
            config.grip_radius_m, config.grip_length_m
        ),
        collision_origin=Origin.from_position(0, 0, -config.grip_length_m / 2),
    )


def create_club_shaft_link(config: ClubConfig) -> Link:
    """Create the club shaft link."""
    return Link(
        name="club_shaft",
        inertia=Inertia.from_cylinder(
            config.shaft_mass_kg,
            config.shaft_radius_m,
            config.shaft_length_m,
            axis="z",
        ),
        visual_geometry=Geometry.cylinder(config.shaft_radius_m, config.shaft_length_m),
        visual_origin=Origin.from_position(0, 0, -config.shaft_length_m / 2),
        visual_material=Material("shaft_material", (0.2, 0.2, 0.22, 1.0)),
        collision_geometry=Geometry.cylinder(
            config.shaft_radius_m, config.shaft_length_m
        ),
        collision_origin=Origin.from_position(0, 0, -config.shaft_length_m / 2),
    )


def create_club_head_link(config: ClubConfig) -> Link:
    """Create the club head link."""
    lx, ly, lz = config.head_dimensions_m
    return Link(
        name="club_head",
        inertia=Inertia.from_box(config.head_mass_kg, lx, ly, lz),
        visual_geometry=Geometry.box(lx, ly, lz),
        visual_origin=Origin.from_position(lx / 2, 0, 0),
        visual_material=Material("head_material", (0.7, 0.7, 0.75, 1.0)),
        collision_geometry=Geometry.box(lx, ly, lz),
        collision_origin=Origin.from_position(lx / 2, 0, 0),
    )


# =============================================================================
# Joint Creation Functions
# =============================================================================


def create_world_to_base_joint() -> Joint:
    """Create fixed joint from world to base (allows positioning)."""
    return Joint(
        name="world_to_base",
        joint_type=JointType.FIXED,
        parent="world",
        child="base_link",
        origin=Origin.from_position(0, 0, 0),
    )


def create_base_to_post_joint(stand_config: StandConfig) -> Joint:
    """Create fixed joint from base to vertical post."""
    base_height = stand_config.base_dimensions_m[2]
    return Joint(
        name="base_to_post",
        joint_type=JointType.FIXED,
        parent="base_link",
        child="vertical_post",
        origin=Origin.from_position(0, 0, base_height),
    )


def create_post_to_shoulder_joint(stand_config: StandConfig) -> Joint:
    """Create fixed joint from post top to shoulder mount."""
    return Joint(
        name="post_to_shoulder",
        joint_type=JointType.FIXED,
        parent="vertical_post",
        child="shoulder_mount",
        origin=Origin.from_position(0, 0, stand_config.post_height_m),
    )


def create_pendulum_joint(config: PendulumConfig) -> Joint:
    """Create the main pendulum revolute joint.

    This is the key joint - single DOF rotation about Y-axis
    for pendulum motion in the X-Z plane (sagittal plane swing).
    """
    return Joint(
        name="pendulum_joint",
        joint_type=JointType.REVOLUTE,
        parent="shoulder_mount",
        child="pendulum_arm",
        origin=Origin.from_position(0, 0, 0),
        axis=(0, 1, 0),  # Y-axis rotation for X-Z plane swing
        limits=JointLimits(
            lower=-config.swing_limit_rad,
            upper=config.swing_limit_rad,
            effort=100.0,
            velocity=5.0,
        ),
        dynamics=JointDynamics(
            damping=config.damping,
            friction=config.friction,
        ),
    )


def create_arm_to_club_mount_joint(pendulum_config: PendulumConfig) -> Joint:
    """Create fixed joint from arm end to club mount."""
    return Joint(
        name="arm_to_mount",
        joint_type=JointType.FIXED,
        parent="pendulum_arm",
        child="club_mount",
        origin=Origin.from_position(0, 0, -pendulum_config.arm_length_m),
    )


def create_mount_to_grip_joint() -> Joint:
    """Create fixed joint from club mount to grip."""
    return Joint(
        name="mount_to_grip",
        joint_type=JointType.FIXED,
        parent="club_mount",
        child="club_grip",
        origin=Origin.from_position(0, 0, 0),
    )


def create_grip_to_shaft_joint(club_config: ClubConfig) -> Joint:
    """Create fixed joint from grip to shaft."""
    return Joint(
        name="grip_to_shaft",
        joint_type=JointType.FIXED,
        parent="club_grip",
        child="club_shaft",
        origin=Origin.from_position(0, 0, -club_config.grip_length_m),
    )


def create_shaft_to_head_joint(club_config: ClubConfig) -> Joint:
    """Create fixed joint from shaft to head."""
    return Joint(
        name="shaft_to_head",
        joint_type=JointType.FIXED,
        parent="club_shaft",
        child="club_head",
        origin=Origin.from_position(0, 0, -club_config.shaft_length_m),
    )


# =============================================================================
# Main Builder Class
# =============================================================================


class PendulumPutterModelBuilder(BaseURDFBuilder):
    """
    Builder for Perfy-style pendulum putting robot.

    Creates a URDF model of a pendulum-based putting device inspired by
    Dave Pelz's Perfy robot. The model consists of:
    - Heavy base for stability
    - Vertical post to shoulder height
    - Shoulder mount (pendulum pivot)
    - Pendulum arm with single DOF rotation
    - Club mount for interchangeable clubs
    - Optional attached putter club

    Physics:
    - Single DOF: Y-axis rotation at shoulder joint
    - Pendulum motion in X-Z plane (sagittal plane)
    - Low damping for natural swing behavior
    - Natural frequency: omega = sqrt(m*g*d / I_pivot)

    Example:
        builder = PendulumPutterModelBuilder(arm_length_m=0.4)
        result = builder.build()
        if result.success:
            builder.save("pendulum_putter.urdf")
    """

    def __init__(
        self,
        arm_length_m: float = 0.40,
        shoulder_height_m: float = 0.85,
        damping: float = 0.05,
        include_club: bool = True,
        club_config: ClubConfig | None = None,
        stand_config: StandConfig | None = None,
    ):
        """
        Initialize the pendulum putter builder.

        Args:
            arm_length_m: Length of pendulum arm in meters
            shoulder_height_m: Height from ground to shoulder pivot
            damping: Damping coefficient for pendulum joint
            include_club: Whether to include default club
            club_config: Custom club configuration
            stand_config: Custom stand configuration

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(robot_name="pendulum_putter")

        # Validate parameters
        self._validate_parameters(arm_length_m, shoulder_height_m, damping)

        # Store configuration
        self._arm_length_m = arm_length_m
        self._shoulder_height_m = shoulder_height_m
        self._damping = damping
        self._include_club = include_club

        # Create config objects
        self._stand_config = stand_config or self._create_stand_config(
            shoulder_height_m
        )
        self._pendulum_config = self._create_pendulum_config(arm_length_m, damping)
        self._club_config = club_config or ClubConfig()

    def _validate_parameters(
        self, arm_length: float, shoulder_height: float, damping: float
    ) -> None:
        """Validate construction parameters."""
        if arm_length <= 0:
            raise ValueError(f"arm_length_m must be positive (got {arm_length})")
        if shoulder_height <= 0:
            raise ValueError(
                f"shoulder_height_m must be positive (got {shoulder_height})"
            )
        if damping < 0:
            raise ValueError(f"damping must be non-negative (got {damping})")

    def _create_stand_config(self, shoulder_height: float) -> StandConfig:
        """Create stand configuration for given shoulder height."""
        base_height = 0.05
        post_height = shoulder_height - base_height
        return StandConfig(post_height_m=max(0.1, post_height))

    def _create_pendulum_config(
        self, arm_length: float, damping: float
    ) -> PendulumConfig:
        """Create pendulum configuration."""
        return PendulumConfig(
            arm_length_m=arm_length,
            damping=damping,
        )

    def _add_stand_links(self) -> None:
        """Add all stand-related links."""
        self._links.append(create_world_link())
        self._links.append(create_base_link(self._stand_config))
        self._links.append(create_vertical_post_link(self._stand_config))
        self._links.append(create_shoulder_mount_link(self._pendulum_config))

    def _add_stand_joints(self) -> None:
        """Add all stand-related joints."""
        self._joints.append(create_world_to_base_joint())
        self._joints.append(create_base_to_post_joint(self._stand_config))
        self._joints.append(create_post_to_shoulder_joint(self._stand_config))

    def _add_pendulum_links(self) -> None:
        """Add pendulum arm and mount links."""
        self._links.append(create_pendulum_arm_link(self._pendulum_config))
        self._links.append(create_club_mount_link())

    def _add_pendulum_joints(self) -> None:
        """Add pendulum joint and arm-to-mount joint."""
        self._joints.append(create_pendulum_joint(self._pendulum_config))
        self._joints.append(create_arm_to_club_mount_joint(self._pendulum_config))

    def _add_club_links(self) -> None:
        """Add club links if club is included."""
        if not self._include_club:
            return
        self._links.append(create_club_grip_link(self._club_config))
        self._links.append(create_club_shaft_link(self._club_config))
        self._links.append(create_club_head_link(self._club_config))

    def _add_club_joints(self) -> None:
        """Add club joints if club is included."""
        if not self._include_club:
            return
        self._joints.append(create_mount_to_grip_joint())
        self._joints.append(create_grip_to_shaft_joint(self._club_config))
        self._joints.append(create_shaft_to_head_joint(self._club_config))

    def clear(self) -> None:
        """Clear all links and joints."""
        self._links.clear()
        self._joints.clear()
        self._materials.clear()

    def build(self, **kwargs: Any) -> BuildResult:
        """
        Build the complete pendulum putter URDF model.

        Returns:
            BuildResult containing success status, URDF XML, and metadata
        """
        # Clear any existing state
        self.clear()

        # Build the model in order
        self._add_stand_links()
        self._add_stand_joints()
        self._add_pendulum_links()
        self._add_pendulum_joints()
        self._add_club_links()
        self._add_club_joints()

        # Validate
        validation = self.validate()
        if not validation.is_valid:
            return BuildResult(
                success=False,
                validation=validation,
                error_message="; ".join(validation.get_error_messages()),
                links=self._links.copy(),
                joints=self._joints.copy(),
            )

        # Generate URDF
        urdf_xml = self.to_urdf(pretty_print=kwargs.get("pretty_print", True))

        return BuildResult(
            success=True,
            urdf_xml=urdf_xml,
            links=self._links.copy(),
            joints=self._joints.copy(),
            validation=validation,
            metadata=self._build_metadata(),
        )

    def _build_metadata(self) -> dict[str, Any]:
        """Build metadata dictionary for the model."""
        return {
            "robot_name": self._robot_name,
            "arm_length_m": self._arm_length_m,
            "shoulder_height_m": self._shoulder_height_m,
            "damping": self._damping,
            "include_club": self._include_club,
            "natural_frequency_approx_hz": self._estimate_natural_frequency(),
        }

    def _estimate_natural_frequency(self) -> float:
        """Estimate natural frequency of the pendulum system.

        For a compound pendulum: omega = sqrt(m*g*d / I_pivot)
        where d = distance from pivot to COM, I_pivot = moment of inertia about pivot
        """
        # Simplified estimation treating as simple pendulum
        # omega = sqrt(g / L_effective)
        effective_length = self._arm_length_m
        if self._include_club:
            effective_length += self._club_config.total_length() * 0.6  # COM estimate

        omega = math.sqrt(GRAVITY_M_S2 / effective_length)
        return omega / (2 * math.pi)  # Convert to Hz

    def save(self, path: str | Path, pretty_print: bool = True) -> Path:
        """
        Build and save the model to file.

        Args:
            path: Output file path
            pretty_print: Whether to format XML with indentation

        Returns:
            Path to saved file
        """
        # Build if not already built
        result = self.build(pretty_print=pretty_print)
        if not result.success:
            raise RuntimeError(f"Build failed: {result.error_message}")

        # Save to file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if result.urdf_xml is None:
            raise RuntimeError("Build produced no URDF XML output")
        path.write_text(result.urdf_xml)

        return path


# =============================================================================
# Utility Functions
# =============================================================================


def create_default_putter() -> ClubConfig:
    """Create configuration for a standard putter."""
    return ClubConfig(
        grip_length_m=0.20,
        grip_radius_m=0.012,
        grip_mass_kg=0.040,
        shaft_length_m=0.80,
        shaft_radius_m=0.005,
        shaft_mass_kg=0.060,
        head_mass_kg=0.340,
        head_dimensions_m=(0.10, 0.06, 0.025),
    )


def create_mallet_putter() -> ClubConfig:
    """Create configuration for a mallet-style putter."""
    return ClubConfig(
        grip_length_m=0.20,
        grip_radius_m=0.014,
        grip_mass_kg=0.050,
        shaft_length_m=0.85,
        shaft_radius_m=0.005,
        shaft_mass_kg=0.065,
        head_mass_kg=0.380,
        head_dimensions_m=(0.12, 0.08, 0.030),
    )


def create_blade_putter() -> ClubConfig:
    """Create configuration for a blade-style putter."""
    return ClubConfig(
        grip_length_m=0.20,
        grip_radius_m=0.011,
        grip_mass_kg=0.035,
        shaft_length_m=0.82,
        shaft_radius_m=0.004,
        shaft_mass_kg=0.055,
        head_mass_kg=0.320,
        head_dimensions_m=(0.08, 0.04, 0.020),
    )
