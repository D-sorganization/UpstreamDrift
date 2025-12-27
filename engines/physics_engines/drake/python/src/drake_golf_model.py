# drake_golf_model.py
"""Drake Golf Model URDF Generator and Diagram Builder."""

import xml.etree.ElementTree as ET  # noqa: N817, RUF100
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any  # noqa: ICN003
from xml.dom import minidom

import numpy as np
import numpy.typing as npt
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    Diagram,
    DiagramBuilder,
    HalfSpace,
    JointIndex,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    RigidBody,
    RigidTransform,
    RollPitchYaw,
    SceneGraph,
    SpatialInertia,
    Sphere,
    UnitInertia,
)

from shared.python.constants import (
    GOLF_BALL_DIAMETER_M,
)

__all__ = [
    "GolfModelParams",
    "GolfURDFGenerator",
    "SegmentParams",
    "build_golf_swing_diagram",
    "make_cylinder_inertia",
]

# -----------------------------
# Parameter containers
# -----------------------------


@dataclass
class SegmentParams:
    """Parameters for a single body segment."""

    length: float
    mass: float
    radius: float = 0.03


@dataclass
class GolfModelParams:
    """Parameters for the entire golf swing model."""

    # Anthropometric parameters
    # [m] Average adult male torso height; source: anthropometric data (Winter 2009)
    pelvis_to_shoulders: float = 0.35
    # [kg] Combined thoracic/lumbar spine mass estimate
    spine_mass: float = 15.0

    scapula_rod: SegmentParams = field(
        default_factory=lambda: SegmentParams(length=0.12, mass=1.0)
    )
    upper_arm: SegmentParams = field(
        default_factory=lambda: SegmentParams(length=0.30, mass=2.0)
    )
    forearm: SegmentParams = field(
        default_factory=lambda: SegmentParams(length=0.27, mass=1.5)
    )
    hand: SegmentParams = field(
        default_factory=lambda: SegmentParams(length=0.10, mass=0.5)
    )

    club: SegmentParams = field(
        default_factory=lambda: SegmentParams(length=1.05, mass=0.40)
    )

    # Distance between hand attachment points along club [m]
    hand_spacing_m: float = 0.0762

    # Joint axes
    hip_axis: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    spine_twist_axis: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )

    spine_universal_axis_1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    spine_universal_axis_2: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )

    scap_axis_1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    scap_axis_2: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )

    wrist_axis_1: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    wrist_axis_2: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )

    # Shoulder gimbal: yaw -> pitch -> roll
    shoulder_axes: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ] = field(
        default_factory=lambda: (
            np.array([0.0, 0.0, 1.0]),  # yaw
            np.array([0.0, 1.0, 0.0]),  # pitch
            np.array([1.0, 0.0, 0.0]),  # roll
        )
    )

    elbow_axis: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )

    # Contact / ground
    ground_friction_mu_static: float = 0.8
    ground_friction_mu_dynamic: float = 0.6
    # Default clubhead radius based on golf ball diameter approx
    clubhead_radius: float = GOLF_BALL_DIAMETER_M / 2.0 + 0.005  # Slight margin


# -----------------------------
# URDF Generation
# -----------------------------


# -----------------------------
# Helper Functions
# -----------------------------


def make_cylinder_inertia(mass: float, radius: float, length: float) -> SpatialInertia:
    """Create spatial inertia for a solid cylinder with its axis along Z.

    Args:
        mass: The mass of the cylinder [kg].
        radius: The radius of the cylinder [m].
        length: The length of the cylinder [m].

    Returns:
        Spatial inertia of the cylinder about its center of mass.

    Raises:
        ValueError: If mass is non-positive.
    """
    if mass <= 0:
        msg = "Mass must be positive."
        raise ValueError(msg)

    # UnitInertia.SolidCylinder creates a UnitInertia.
    # The cylinder is aligned with Z axis by default in pydrake's SolidCylinder.
    # Axis argument [0, 0, 1] confirms alignment.
    unit_inertia = UnitInertia.SolidCylinder(
        radius,
        length,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),  # type: ignore[arg-type]
    )
    # Construct SpatialInertia using mass, center of mass at [0, 0, 0],
    # and the unit inertia.
    return SpatialInertia(
        mass,
        np.zeros(3, dtype=np.float64),  # type: ignore[arg-type]
        unit_inertia,
    )


class GolfURDFGenerator:
    """Generates URDF for the golf swing model from parameters."""

    def __init__(self, params: GolfModelParams) -> None:
        """Initialize the generator with parameters."""
        self.params = params
        self.root = ET.Element("robot", name="golf_swing_model")
        self.materials: set[str] = set()
        self.dummy_mass = 0.01

        # Add basic material
        self._add_material("gray", "0.5 0.5 0.5 1.0")

    def _add_material(self, name: str, rgba: str) -> None:
        """Add a material definition to the URDF."""
        if name not in self.materials:
            mat = ET.SubElement(self.root, "material", name=name)
            ET.SubElement(mat, "color", rgba=rgba)
            self.materials.add(name)

    def _np_to_str(self, arr: npt.ArrayLike) -> str:
        """Convert numpy array to space-separated string."""
        return " ".join(f"{x:.6g}" for x in np.array(arr).flatten())

    def _transform_to_origin_xml(self, X: RigidTransform) -> ET.Element:  # noqa: N803
        """Convert RigidTransform to XML origin element."""
        origin = ET.Element("origin")
        p = X.translation()
        rpy = RollPitchYaw(X.rotation()).vector()
        origin.set("xyz", self._np_to_str(p))
        origin.set("rpy", self._np_to_str(rpy))
        return origin

    def _create_inertia_xml(
        self, mass: float, unit_inertia: UnitInertia, com: npt.ArrayLike
    ) -> ET.Element:
        """Create inertial XML element."""
        inertial = ET.Element("inertial")
        ET.SubElement(inertial, "mass", value=f"{mass:.6g}")

        origin = ET.Element("origin")
        origin.set("xyz", self._np_to_str(com))
        origin.set("rpy", "0 0 0")
        inertial.append(origin)

        rot_inertia = unit_inertia * mass
        moments = rot_inertia.get_moments()  # Ixx, Iyy, Izz
        products = rot_inertia.get_products()  # Ixy, Ixz, Iyz

        inertia_elem = ET.SubElement(inertial, "inertia")
        inertia_elem.set("ixx", f"{moments[0]:.6g}")
        inertia_elem.set("iyy", f"{moments[1]:.6g}")
        inertia_elem.set("izz", f"{moments[2]:.6g}")
        inertia_elem.set("ixy", f"{products[0]:.6g}")
        inertia_elem.set("ixz", f"{products[1]:.6g}")
        inertia_elem.set("iyz", f"{products[2]:.6g}")

        return inertial

    def add_link(  # noqa: PLR0913
        self,
        name: str,
        mass: float,
        unit_inertia: UnitInertia,
        visual_shape_tag: str | None = None,
        visual_params: dict[str, Any] | None = None,
        com_offset: npt.ArrayLike | None = None,
    ) -> ET.Element:
        """Add a link to the model."""
        if com_offset is None:
            com_offset = np.zeros(3)
        link = ET.SubElement(self.root, "link", name=name)

        # Inertial
        inertial = self._create_inertia_xml(mass, unit_inertia, com_offset)
        link.append(inertial)

        # Visual/Collision
        if visual_shape_tag:
            assert visual_params is not None  # noqa: S101
            for tag in ["visual", "collision"]:
                vis = ET.SubElement(link, tag)
                ET.SubElement(
                    vis, "origin", xyz=self._np_to_str(com_offset), rpy="0 0 0"
                )
                geom = ET.SubElement(vis, "geometry")

                if visual_shape_tag == "box":
                    ET.SubElement(
                        geom, "box", size=self._np_to_str(visual_params["size"])
                    )
                elif visual_shape_tag == "cylinder":
                    ET.SubElement(
                        geom,
                        "cylinder",
                        length=str(visual_params["length"]),
                        radius=str(visual_params["radius"]),
                    )
                elif visual_shape_tag == "sphere":
                    ET.SubElement(geom, "sphere", radius=str(visual_params["radius"]))

                if tag == "visual":
                    ET.SubElement(vis, "material", name="gray")

        return link

    def add_joint(  # noqa: PLR0913
        self,
        name: str,
        joint_type: str,
        parent: str,
        child: str,
        origin_transform: RigidTransform,
        axis: npt.ArrayLike | None = None,
    ) -> None:
        """Add a joint to the model."""
        joint = ET.SubElement(self.root, "joint", name=name, type=joint_type)
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)

        joint.append(self._transform_to_origin_xml(origin_transform))

        if axis is not None:
            ET.SubElement(joint, "axis", xyz=self._np_to_str(axis))

    def generate(self) -> str:  # noqa: PLR0915
        """Generate the URDF string."""
        p = self.params

        # 1. Pelvis
        pelvis_dims = np.array([0.3, 0.2, 0.2])
        I_pelvis = UnitInertia.SolidBox(pelvis_dims[0], pelvis_dims[1], pelvis_dims[2])
        self.add_link("pelvis", p.spine_mass, I_pelvis, "box", {"size": pelvis_dims})

        # 2. Spine Base (Hip Joint)
        sb_dims = np.array([0.1, 0.1, 0.1])
        I_sb = UnitInertia.SolidBox(sb_dims[0], sb_dims[1], sb_dims[2])
        self.add_link("spine_base", 1.0, I_sb, "box", {"size": sb_dims})
        self.add_joint(
            "hip_yaw", "revolute", "pelvis", "spine_base", RigidTransform(), p.hip_axis
        )

        # 3. Lower Spine
        ls_dims = np.array([0.2, 0.2, p.pelvis_to_shoulders * 0.5])
        I_ls = UnitInertia.SolidBox(ls_dims[0], ls_dims[1], ls_dims[2])
        self.add_link("spine_dummy", self.dummy_mass, UnitInertia.SolidSphere(0.01))
        self.add_link("lower_spine", p.spine_mass * 0.5, I_ls, "box", {"size": ls_dims})

        self.add_joint(
            "spine_universal_1",
            "revolute",
            "spine_base",
            "spine_dummy",
            RigidTransform(),
            p.spine_universal_axis_1,
        )
        self.add_joint(
            "spine_universal_2",
            "revolute",
            "spine_dummy",
            "lower_spine",
            RigidTransform(),
            p.spine_universal_axis_2,
        )

        # 4. Upper Spine
        us_dims = np.array([0.2, 0.2, p.pelvis_to_shoulders * 0.5])
        I_us = UnitInertia.SolidBox(us_dims[0], us_dims[1], us_dims[2])
        us_offset = np.array([0.0, 0.0, p.pelvis_to_shoulders * 0.25])
        self.add_link(
            "upper_spine",
            p.spine_mass * 0.5,
            I_us,
            "box",
            {"size": us_dims},
            com_offset=us_offset,
        )

        self.add_joint(
            "spine_twist",
            "revolute",
            "lower_spine",
            "upper_spine",
            RigidTransform(
                p=np.array([0.0, 0.0, p.pelvis_to_shoulders * 0.25], dtype=np.float64)  # type: ignore[arg-type]
            ),
            p.spine_twist_axis,
        )

        # Torso Hub
        hub_dims = np.array([0.3, 0.3, 0.2])
        I_hub = UnitInertia.SolidBox(hub_dims[0], hub_dims[1], hub_dims[2])
        self.add_link("upper_torso_hub", 5.0, I_hub, "box", {"size": hub_dims})

        # Weld at top of upper spine (+0.25*pts relative to Body,
        # so +0.5*pts relative to Link)
        self.add_joint(
            "torso_weld",
            "fixed",
            "upper_spine",
            "upper_torso_hub",
            RigidTransform(
                p=np.array([0.0, 0.0, p.pelvis_to_shoulders * 0.5], dtype=np.float64)  # type: ignore[arg-type]
            ),
        )

        # 5. Arms
        for side in ["left", "right"]:
            sign = 1.0 if side == "right" else -1.0

            # Scapula
            scap_offset = np.array([0.0, sign * 0.18, 0.10], dtype=np.float64)  # type: ignore[arg-type]
            scap_len = p.scapula_rod.length

            self.add_link(
                f"{side}_scapula_dummy", self.dummy_mass, UnitInertia.SolidSphere(0.01)
            )

            scap_body_offset = np.array([0.0, 0.0, scap_len / 2.0], dtype=np.float64)  # type: ignore[arg-type]
            I_scap = UnitInertia.SolidCylinder(
                p.scapula_rod.radius,
                scap_len,
                np.array([0.0, 0.0, 1.0], dtype=np.float64),  # type: ignore[arg-type]
            )
            self.add_link(
                f"{side}_scapula_rod",
                p.scapula_rod.mass,
                I_scap,
                "cylinder",
                {"radius": p.scapula_rod.radius, "length": scap_len},
                com_offset=scap_body_offset,
            )

            self.add_joint(
                f"{side}_scapula_universal_1",
                "revolute",
                "upper_torso_hub",
                f"{side}_scapula_dummy",
                RigidTransform(p=scap_offset),  # type: ignore[arg-type]
                p.scap_axis_1,
            )
            self.add_joint(
                f"{side}_scapula_universal_2",
                "revolute",
                f"{side}_scapula_dummy",
                f"{side}_scapula_rod",
                RigidTransform(),
                p.scap_axis_2,
            )

            # Shoulder
            self.add_link(
                f"{side}_shoulder_yaw_link",
                0.1,
                UnitInertia.SolidSphere(0.05),
                "sphere",
                {"radius": 0.05},
            )
            self.add_joint(
                f"{side}_shoulder_yaw",
                "revolute",
                f"{side}_scapula_rod",
                f"{side}_shoulder_yaw_link",
                RigidTransform(
                    p=np.array([0.0, 0.0, scap_len], dtype=np.float64)  # type: ignore[arg-type]
                ),
                p.shoulder_axes[0],
            )

            self.add_link(
                f"{side}_shoulder_pitch_link",
                0.1,
                UnitInertia.SolidSphere(0.05),
                "sphere",
                {"radius": 0.05},
            )
            self.add_joint(
                f"{side}_shoulder_pitch",
                "revolute",
                f"{side}_shoulder_yaw_link",
                f"{side}_shoulder_pitch_link",
                RigidTransform(),
                p.shoulder_axes[1],
            )

            self.add_link(
                f"{side}_shoulder_roll_link",
                0.1,
                UnitInertia.SolidSphere(0.05),
                "sphere",
                {"radius": 0.05},
            )
            self.add_joint(
                f"{side}_shoulder_roll",
                "revolute",
                f"{side}_shoulder_pitch_link",
                f"{side}_shoulder_roll_link",
                RigidTransform(),
                p.shoulder_axes[2],
            )

            # Upper Arm
            ua_len = p.upper_arm.length
            I_ua = UnitInertia.SolidCylinder(
                p.upper_arm.radius,
                ua_len,
                np.array([0.0, 0.0, 1.0], dtype=np.float64),  # type: ignore[arg-type]
            )

            self.add_link(
                f"{side}_upper_arm",
                p.upper_arm.mass,
                I_ua,
                "cylinder",
                {"radius": p.upper_arm.radius, "length": ua_len},
            )

            self.add_joint(
                f"{side}_upper_arm_weld",
                "fixed",
                f"{side}_shoulder_roll_link",
                f"{side}_upper_arm",
                RigidTransform(
                    p=np.array([0.0, 0.0, -ua_len / 2.0], dtype=np.float64)  # type: ignore[arg-type]
                ),
            )

            # Elbow
            # At -L/2 relative to Body (Bottom).
            self.add_joint(
                f"{side}_elbow",
                "revolute",
                f"{side}_upper_arm",
                f"{side}_forearm",
                RigidTransform(
                    p=np.array([0.0, 0.0, -ua_len / 2.0], dtype=np.float64)  # type: ignore[arg-type]
                ),
                p.elbow_axis,
            )

            # Forearm
            fa_len = p.forearm.length
            I_fa = UnitInertia.SolidCylinder(
                p.forearm.radius,
                fa_len,
                np.array([0.0, 0.0, 1.0], dtype=np.float64),  # type: ignore[arg-type]
            )
            fa_offset = np.array([0.0, 0.0, fa_len / 2.0], dtype=np.float64)  # type: ignore[arg-type]

            self.add_link(
                f"{side}_forearm",
                p.forearm.mass,
                I_fa,
                "cylinder",
                {"radius": p.forearm.radius, "length": fa_len},
                com_offset=fa_offset,
            )

            # Wrist
            self.add_link(
                f"{side}_wrist_dummy", self.dummy_mass, UnitInertia.SolidSphere(0.01)
            )

            hand_len = p.hand.length
            I_hand = UnitInertia.SolidCylinder(
                p.hand.radius,
                hand_len,
                np.array([0.0, 0.0, 1.0], dtype=np.float64),  # type: ignore[arg-type]
            )
            hand_offset = np.array([0.0, 0.0, hand_len / 2.0], dtype=np.float64)  # type: ignore[arg-type]
            self.add_link(
                f"{side}_hand",
                p.hand.mass,
                I_hand,
                "cylinder",
                {"radius": p.hand.radius, "length": hand_len},
                com_offset=hand_offset,
            )

            self.add_joint(
                f"{side}_wrist_universal_1",
                "revolute",
                f"{side}_forearm",
                f"{side}_wrist_dummy",
                RigidTransform(
                    p=np.array([0.0, 0.0, fa_len], dtype=np.float64)  # type: ignore[arg-type]
                ),
                p.wrist_axis_1,
            )
            self.add_joint(
                f"{side}_wrist_universal_2",
                "revolute",
                f"{side}_wrist_dummy",
                f"{side}_hand",
                RigidTransform(),
                p.wrist_axis_2,
            )

        # 6. Club (Attached to Left Hand)
        c_len = p.club.length
        I_club = UnitInertia.SolidCylinder(
            p.club.radius,
            c_len,
            np.array([0.0, 0.0, 1.0], dtype=np.float64),  # type: ignore[arg-type]
        )
        # Grip at butt (start of cylinder in link frame), COM at L/2
        club_com_offset = np.array([0.0, 0.0, c_len / 2.0], dtype=np.float64)  # type: ignore[arg-type]

        self.add_link(
            "club",
            p.club.mass,
            I_club,
            "cylinder",
            {"radius": p.club.radius, "length": c_len},
            com_offset=club_com_offset,
        )

        # Attach to left hand (tip)
        self.add_joint(
            "grip_lead",
            "fixed",
            "left_hand",
            "club",
            RigidTransform(
                p=np.array([0.0, 0.0, p.hand.length], dtype=np.float64)  # type: ignore[arg-type]
            ),
        )

        xml_str = ET.tostring(self.root, encoding="utf-8")
        return minidom.parseString(xml_str).toprettyxml(indent="  ")  # noqa: S318


# -----------------------------
# Main builder
# -----------------------------


def add_ground_and_club_contact(
    plant: MultibodyPlant,
    club: RigidBody,
    params: GolfModelParams,
) -> None:
    """Add ground and club contact geometry to the plant."""
    world_body = plant.world_body()
    X_WG = RigidTransform()

    friction = CoulombFriction(
        params.ground_friction_mu_static, params.ground_friction_mu_dynamic
    )
    plant.RegisterCollisionGeometry(
        world_body,
        X_WG,
        HalfSpace(),
        "ground_collision",
        friction,  # type: ignore[arg-type]
    )
    # Add Visual for ground with color
    # Use a large Box instead of HalfSpace for better compatibility with Meshcat
    ground_box = Box(100.0, 100.0, 1.0)
    X_WG_visual = RigidTransform(p=np.array([0, 0, -0.5]))
    plant.RegisterVisualGeometry(
        world_body,
        X_WG_visual,
        ground_box,
        "ground_visual",
        np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float64),  # type: ignore[arg-type]
    )

    # Clubhead collision sphere
    X_C_H = RigidTransform(
        p=np.array([0.0, 0.0, params.club.length / 2.0], dtype=np.float64)  # type: ignore[arg-type]
    )
    plant.RegisterCollisionGeometry(
        club,
        X_C_H,
        Sphere(params.clubhead_radius),
        "clubhead_collision",
        friction,  # type: ignore[arg-type]
    )
    plant.RegisterVisualGeometry(
        club,
        X_C_H,
        Sphere(params.clubhead_radius),
        "clubhead_visual",
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64),  # type: ignore[arg-type]
    )


def add_joint_actuators(plant: MultibodyPlant) -> None:
    """Add actuators to all single-dof joints."""
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        if joint.num_velocities() != 1:
            continue
        plant.AddJointActuator(f"{joint.name()}_act", joint)


def build_golf_swing_diagram(
    params: GolfModelParams | None = None,
    urdf_path: str | None = None,
    meshcat: Meshcat | None = None,
) -> tuple[Diagram, MultibodyPlant, SceneGraph]:
    """Build the full Drake diagram for the golf swing."""
    if params is None:
        params = GolfModelParams()

    # Generate URDF
    generator = GolfURDFGenerator(params)
    urdf_content = generator.generate()

    if urdf_path:
        Path(urdf_path).write_text(urdf_content, encoding="utf-8")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)

    parser = Parser(plant)
    # Load from string to avoid file system dependency/side effects
    model_instance = parser.AddModelsFromString(urdf_content, "urdf")[0]

    # Add Right Hand Constraint
    right_hand = plant.GetBodyByName("right_hand", model_instance)
    club = plant.GetBodyByName("club", model_instance)

    # Calculate points in Body Frames
    # Right Hand Tip: [0, 0, hand_len/2] (Body is at L/2, Tip at L)
    p_right_hand = np.array([0.0, 0.0, params.hand.length / 2.0])

    # Club Trail Point relative to Club Body.
    # Grip Lead (Link Frame) is at 0 in Link Frame.
    # Club Body Center is at +L/2 in Link Frame.
    # Grip Trail is at +spacing in Link Frame.
    # Grip Trail in Body Frame = (+spacing) - (+L/2) = spacing - L/2.
    p_club_trail = np.array(
        [0.0, 0.0, params.hand_spacing_m - params.club.length / 2.0]
    )

    plant.AddBallConstraint(
        body_A=right_hand, p_AP=p_right_hand, body_B=club, p_BQ=p_club_trail
    )

    # Ground
    add_ground_and_club_contact(plant, club, params)

    # Actuators
    add_joint_actuators(plant)

    plant.Finalize()

    # Visualization
    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    return diagram, plant, scene_graph
