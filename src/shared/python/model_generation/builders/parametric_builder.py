"""
Parametric URDF builder for parameter-driven model generation.

This builder generates complete models from high-level parameters
like height, weight, and body proportions.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from model_generation.builders.base_builder import BaseURDFBuilder, BuildResult
from model_generation.core.constants import (
    DEFAULT_HEIGHT_M,
    DEFAULT_JOINT_DAMPING,
    DEFAULT_MASS_KG,
)
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)
from model_generation.inertia.calculator import InertiaCalculator, InertiaMode

logger = logging.getLogger(__name__)


@dataclass
class ParametricConfig:
    """Configuration for parametric builder."""

    # Inertia calculation mode
    inertia_mode: InertiaMode = InertiaMode.PRIMITIVE

    # Density for mesh-based inertia (kg/m^3)
    density: float = 1050.0

    # Joint defaults
    default_joint_damping: float = DEFAULT_JOINT_DAMPING
    default_joint_friction: float = 0.0

    # Geometry options
    use_capsules: bool = True
    generate_collision: bool = True

    # Composite joint handling
    expand_composite_joints: bool = True

    # Material
    default_material: Material = field(default_factory=Material.skin)


class ParametricBuilder(BaseURDFBuilder):
    """
    Parametric URDF builder for generating models from parameters.

    This builder takes high-level parameters (height, mass, proportions)
    and generates a complete URDF model with appropriate segment sizes,
    masses, and inertias.

    Example:
        builder = ParametricBuilder("humanoid")
        builder.set_parameters(height_m=1.80, mass_kg=80.0)
        builder.add_humanoid_segments()
        result = builder.build()
    """

    def __init__(
        self,
        robot_name: str = "humanoid",
        config: ParametricConfig | None = None,
    ):
        """
        Initialize parametric builder.

        Args:
            robot_name: Name for the robot element
            config: Builder configuration
        """
        assert robot_name is not None, "robot_name must be provided"
        super().__init__(robot_name)
        self._config = config or ParametricConfig()
        self._inertia_calc = InertiaCalculator(default_mode=self._config.inertia_mode)

        # Parameters
        self._height_m: float = DEFAULT_HEIGHT_M
        self._mass_kg: float = DEFAULT_MASS_KG
        self._gender_factor: float = 0.5  # 0=female, 1=male
        self._proportions: dict[str, float] = {}

        # Segment templates (can be customized)
        self._segment_templates: dict[str, dict[str, Any]] = {}

    @property
    def height_m(self) -> float:
        """Get height in meters."""
        return self._height_m

    @property
    def mass_kg(self) -> float:
        """Get mass in kg."""
        return self._mass_kg

    def set_parameters(
        self,
        height_m: float | None = None,
        mass_kg: float | None = None,
        gender_factor: float | None = None,
        **proportions: float,
    ) -> ParametricBuilder:
        """
        Set body parameters.

        Args:
            height_m: Total height in meters
            mass_kg: Total mass in kg
            gender_factor: 0=female, 1=male, 0.5=neutral
            **proportions: Proportion factors (e.g., shoulder_width=1.1)

        Returns:
            Self for method chaining
        """
        if height_m is not None:
            if height_m <= 0:
                raise ValueError(f"height_m must be positive, got {height_m}")
            self._height_m = height_m
        if mass_kg is not None:
            if mass_kg <= 0:
                raise ValueError(f"mass_kg must be positive, got {mass_kg}")
            self._mass_kg = mass_kg
        if gender_factor is not None:
            self._gender_factor = max(0.0, min(1.0, gender_factor))
        self._proportions.update(proportions)

        return self

    def add_segment(
        self,
        name: str,
        parent: str | None,
        mass_ratio: float,
        length_ratio: float,
        geometry_type: GeometryType = GeometryType.CAPSULE,
        width_ratio: float = 0.15,
        joint_type: JointType = JointType.REVOLUTE,
        joint_axis: tuple[float, float, float] = (1, 0, 0),
        joint_limits: tuple[float, float] | None = None,
        origin_offset: tuple[float, float, float] = (0, 0, 0),
        material: Material | None = None,
    ) -> ParametricBuilder:
        """
        Add a parametric segment.

        Args:
            name: Segment name
            parent: Parent segment name (None for root)
            mass_ratio: Fraction of total body mass
            length_ratio: Fraction of total height
            geometry_type: Shape type
            width_ratio: Width as fraction of length
            joint_type: Type of joint connecting to parent
            joint_axis: Joint rotation/translation axis
            joint_limits: (lower, upper) limits in radians
            origin_offset: (x, y, z) offset from parent
            material: Visual material

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name is empty, segment name already exists,
                or ratios are non-positive.
        """
        if not name or not name.strip():
            raise ValueError("Segment name must be a non-empty string")
        if mass_ratio <= 0:
            raise ValueError(f"mass_ratio must be positive, got {mass_ratio}")
        if length_ratio <= 0:
            raise ValueError(f"length_ratio must be positive, got {length_ratio}")
        if width_ratio <= 0:
            raise ValueError(f"width_ratio must be positive, got {width_ratio}")

        # Check for duplicate segment names
        existing_names = {link.name for link in self._links}
        if name in existing_names:
            raise ValueError(f"Segment '{name}' already exists")

        # Compute dimensions
        length = self._height_m * length_ratio
        width = length * width_ratio
        mass = self._mass_kg * mass_ratio

        geometry, inertia = self._create_segment_geometry(
            geometry_type,
            mass,
            length,
            width,
        )

        # Create link
        link = Link(
            name=name,
            inertia=inertia,
            visual_geometry=geometry,
            visual_material=material or self._config.default_material,
            collision_geometry=geometry if self._config.generate_collision else None,
        )
        self._links.append(link)

        # Create joint if has parent
        if parent is not None:
            self._create_segment_joint(
                name,
                parent,
                joint_type,
                joint_axis,
                joint_limits,
                origin_offset,
            )

        return self

    @staticmethod
    def _create_segment_geometry(
        geometry_type: GeometryType,
        mass: float,
        length: float,
        width: float,
    ) -> tuple[Geometry, Inertia]:
        """Create geometry and inertia for a segment based on shape type."""
        assert geometry_type is not None, "geometry_type must be provided"
        if geometry_type == GeometryType.CAPSULE:
            radius = width / 2
            cyl_length = max(0.01, length - 2 * radius)
            return Geometry.capsule(radius, cyl_length), Inertia.from_capsule(
                mass,
                radius,
                cyl_length,
            )
        if geometry_type == GeometryType.CYLINDER:
            radius = width / 2
            return Geometry.cylinder(radius, length), Inertia.from_cylinder(
                mass,
                radius,
                length,
            )
        if geometry_type == GeometryType.SPHERE:
            radius = length / 2
            return Geometry.sphere(radius), Inertia.from_sphere(mass, radius)
        # BOX
        return Geometry.box(width, width, length), Inertia.from_box(
            mass,
            width,
            width,
            length,
        )

    def _create_segment_joint(
        self,
        name: str,
        parent: str,
        joint_type: JointType,
        joint_axis: tuple[float, float, float],
        joint_limits: tuple[float, float] | None,
        origin_offset: tuple[float, float, float],
    ) -> None:
        """Create a joint connecting a segment to its parent."""
        assert name is not None, "name must be provided"
        joint_name = f"{parent}_to_{name}"
        limits = None
        if joint_limits and joint_type == JointType.REVOLUTE:
            limits = JointLimits(
                lower=joint_limits[0],
                upper=joint_limits[1],
            )

        joint = Joint(
            name=joint_name,
            joint_type=joint_type,
            parent=parent,
            child=name,
            origin=Origin(xyz=origin_offset),
            axis=joint_axis,
            limits=limits,
            dynamics=JointDynamics(
                damping=self._config.default_joint_damping,
                friction=self._config.default_joint_friction,
            ),
        )
        self._joints.append(joint)

    def add_humanoid_segments(self) -> ParametricBuilder:
        """
        Add standard humanoid body segments.

        This adds a complete humanoid skeleton with:
        - Pelvis (root)
        - Spine (lumbar, thorax)
        - Head and neck
        - Arms (shoulder, upper arm, forearm, hand) x2
        - Legs (thigh, shin, foot) x2

        Returns:
            Self for method chaining
        """
        get_mass, get_length = self._get_anthropometry_helpers()

        # Add body segments in logical groups
        segment_heights = self._add_torso_segments(get_mass, get_length)
        self._add_head_neck_segments(get_mass, get_length, segment_heights)
        self._add_arm_segments(get_mass, get_length, segment_heights)
        self._add_leg_segments(get_mass, get_length, segment_heights)

        return self

    def _get_anthropometry_helpers(
        self,
    ) -> tuple[Callable[[str, float], float], Callable[[str, float], float]]:
        """Get helper functions for anthropometric data lookup."""
        try:
            from model_generation.humanoid.anthropometry import (
                get_segment_length_ratio,
                get_segment_mass_ratio,
            )

            use_anthropometry = True
        except ImportError:
            use_anthropometry = False

        def get_mass(name: str, default: float) -> float:
            if use_anthropometry:
                try:
                    return get_segment_mass_ratio(name, self._gender_factor)
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug("Anthropometry mass lookup failed for %s: %s", name, e)
            return default

        def get_length(name: str, default: float) -> float:
            if use_anthropometry:
                try:
                    return get_segment_length_ratio(name, self._gender_factor)
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(
                        "Anthropometry length lookup failed for %s: %s", name, e
                    )
            return default

        return get_mass, get_length

    def _add_torso_segments(
        self,
        get_mass: Callable[[str, float], float],
        get_length: Callable[[str, float], float],
    ) -> dict[str, float]:
        """Add torso segments (pelvis, lumbar, thorax)."""
        # Pelvis (root)
        assert get_mass is not None, "get_mass must be provided"
        self.add_segment(
            name="pelvis",
            parent=None,
            mass_ratio=get_mass("pelvis", 0.112),
            length_ratio=get_length("pelvis", 0.078),
            geometry_type=GeometryType.BOX,
            width_ratio=1.5,
            material=Material.skin(),
        )

        pelvis_height = self._height_m * get_length("pelvis", 0.078)

        # Lumbar spine
        self.add_segment(
            name="lumbar",
            parent="pelvis",
            mass_ratio=get_mass("lumbar", 0.139),
            length_ratio=get_length("lumbar", 0.108),
            geometry_type=GeometryType.BOX,
            width_ratio=0.7,
            joint_type=JointType.UNIVERSAL,
            joint_limits=(-math.radians(30), math.radians(30)),
            origin_offset=(0, 0, pelvis_height / 2),
        )

        lumbar_height = self._height_m * get_length("lumbar", 0.108)

        # Thorax
        self.add_segment(
            name="thorax",
            parent="lumbar",
            mass_ratio=get_mass("thorax", 0.216),
            length_ratio=get_length("thorax", 0.170),
            geometry_type=GeometryType.BOX,
            width_ratio=0.8,
            joint_type=JointType.UNIVERSAL,
            joint_limits=(-math.radians(20), math.radians(20)),
            origin_offset=(0, 0, lumbar_height),
        )

        thorax_height = self._height_m * get_length("thorax", 0.170)

        return {
            "pelvis": pelvis_height,
            "lumbar": lumbar_height,
            "thorax": thorax_height,
        }

    def _add_head_neck_segments(
        self,
        get_mass: Callable[[str, float], float],
        get_length: Callable[[str, float], float],
        segment_heights: dict[str, float],
    ) -> None:
        """Add head and neck segments."""
        assert get_mass is not None, "get_mass must be provided"
        thorax_height = segment_heights["thorax"]

        # Neck
        self.add_segment(
            name="neck",
            parent="thorax",
            mass_ratio=get_mass("neck", 0.024),
            length_ratio=get_length("neck", 0.052),
            geometry_type=GeometryType.CYLINDER,
            width_ratio=0.5,
            joint_type=JointType.UNIVERSAL,
            joint_limits=(-math.radians(45), math.radians(45)),
            origin_offset=(0, 0, thorax_height),
        )

        neck_height = self._height_m * get_length("neck", 0.052)

        # Head
        self.add_segment(
            name="head",
            parent="neck",
            mass_ratio=get_mass("head", 0.069),
            length_ratio=get_length("head", 0.139),
            geometry_type=GeometryType.SPHERE,
            joint_type=JointType.GIMBAL,
            joint_limits=(-math.radians(60), math.radians(60)),
            origin_offset=(0, 0, neck_height),
        )

    def _add_arm_segments(
        self,
        get_mass: Callable[[str, float], float],
        get_length: Callable[[str, float], float],
        segment_heights: dict[str, float],
    ) -> None:
        """Add arm segments (shoulder, upper arm, forearm, hand) for both sides."""
        assert get_mass is not None, "get_mass must be provided"
        thorax_height = segment_heights["thorax"]
        shoulder_width = (
            self._height_m * 0.23 * self._proportions.get("shoulder_width_factor", 1.0)
        )

        for side, y_sign in [("left", 1), ("right", -1)]:
            self._add_single_arm(
                side, y_sign, get_mass, get_length, thorax_height, shoulder_width
            )

    def _add_single_arm(
        self,
        side: str,
        y_sign: int,
        get_mass: Callable[[str, float], float],
        get_length: Callable[[str, float], float],
        thorax_height: float,
        shoulder_width: float,
    ) -> None:
        """Add segments for a single arm."""
        # Shoulder
        assert side is not None, "side must be provided"
        self.add_segment(
            name=f"{side}_shoulder",
            parent="thorax",
            mass_ratio=0.005,
            length_ratio=0.04,
            geometry_type=GeometryType.SPHERE,
            joint_type=JointType.FIXED,
            origin_offset=(0, y_sign * shoulder_width / 2, thorax_height * 0.9),
        )

        # Upper arm
        self.add_segment(
            name=f"{side}_upper_arm",
            parent=f"{side}_shoulder",
            mass_ratio=get_mass("upper_arm", 0.027),
            length_ratio=get_length("upper_arm", 0.186),
            geometry_type=GeometryType.CAPSULE,
            width_ratio=0.18,
            joint_type=JointType.GIMBAL,
            joint_limits=(-math.pi, math.pi),
            origin_offset=(0, y_sign * 0.02, -0.02),
        )

        upper_arm_length = self._height_m * get_length("upper_arm", 0.186)

        # Forearm
        self.add_segment(
            name=f"{side}_forearm",
            parent=f"{side}_upper_arm",
            mass_ratio=get_mass("forearm", 0.016),
            length_ratio=get_length("forearm", 0.146),
            geometry_type=GeometryType.CAPSULE,
            width_ratio=0.14,
            joint_type=JointType.REVOLUTE,
            joint_axis=(1, 0, 0),
            joint_limits=(0, math.radians(150)),
            origin_offset=(0, 0, -upper_arm_length),
        )

        forearm_length = self._height_m * get_length("forearm", 0.146)

        # Hand
        self.add_segment(
            name=f"{side}_hand",
            parent=f"{side}_forearm",
            mass_ratio=get_mass("hand", 0.006),
            length_ratio=get_length("hand", 0.108),
            geometry_type=GeometryType.BOX,
            width_ratio=0.5,
            joint_type=JointType.UNIVERSAL,
            joint_limits=(-math.radians(80), math.radians(80)),
            origin_offset=(0, 0, -forearm_length),
        )

    def _add_leg_segments(
        self,
        get_mass: Callable[[str, float], float],
        get_length: Callable[[str, float], float],
        segment_heights: dict[str, float],
    ) -> None:
        """Add leg segments (thigh, shin, foot) for both sides."""
        assert get_mass is not None, "get_mass must be provided"
        pelvis_height = segment_heights["pelvis"]
        hip_width = (
            self._height_m * 0.1 * self._proportions.get("hip_width_factor", 1.0)
        )

        for side, y_sign in [("left", 1), ("right", -1)]:
            self._add_single_leg(
                side, y_sign, get_mass, get_length, pelvis_height, hip_width
            )

    def _add_single_leg(
        self,
        side: str,
        y_sign: int,
        get_mass: Callable[[str, float], float],
        get_length: Callable[[str, float], float],
        pelvis_height: float,
        hip_width: float,
    ) -> None:
        """Add segments for a single leg."""
        # Thigh
        assert side is not None, "side must be provided"
        self.add_segment(
            name=f"{side}_thigh",
            parent="pelvis",
            mass_ratio=get_mass("thigh", 0.142),
            length_ratio=get_length("thigh", 0.245),
            geometry_type=GeometryType.CAPSULE,
            width_ratio=0.22,
            joint_type=JointType.GIMBAL,
            joint_limits=(-math.radians(120), math.radians(30)),
            origin_offset=(0, y_sign * hip_width, -pelvis_height / 2),
        )

        thigh_length = self._height_m * get_length("thigh", 0.245)

        # Shin
        self.add_segment(
            name=f"{side}_shin",
            parent=f"{side}_thigh",
            mass_ratio=get_mass("shin", 0.043),
            length_ratio=get_length("shin", 0.246),
            geometry_type=GeometryType.CAPSULE,
            width_ratio=0.14,
            joint_type=JointType.REVOLUTE,
            joint_axis=(1, 0, 0),
            joint_limits=(-math.radians(150), 0),
            origin_offset=(0, 0, -thigh_length),
        )

        shin_length = self._height_m * get_length("shin", 0.246)

        # Foot
        self.add_segment(
            name=f"{side}_foot",
            parent=f"{side}_shin",
            mass_ratio=get_mass("foot", 0.014),
            length_ratio=get_length("foot", 0.152),
            geometry_type=GeometryType.BOX,
            width_ratio=0.35,
            joint_type=JointType.UNIVERSAL,
            joint_limits=(-math.radians(45), math.radians(45)),
            origin_offset=(0, 0, -shin_length),
        )

    def clear(self) -> None:
        """Clear all segments."""
        self._links.clear()
        self._joints.clear()
        self._segment_templates.clear()

    def build(self, **kwargs: Any) -> BuildResult:
        """
        Build the URDF model.

        Returns:
            BuildResult with generated URDF
        """
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
            metadata={
                "height_m": self._height_m,
                "mass_kg": self._mass_kg,
                "gender_factor": self._gender_factor,
                "proportions": self._proportions.copy(),
            },
        )
