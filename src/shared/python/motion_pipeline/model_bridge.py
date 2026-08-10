"""SkeletonRig → URDF bridge for engine-backed solvers.

Part of epic #8390 (B2/#8397, reused by C1/#8401). Renders a canonical
``SkeletonRig`` as a URDF string so engine-backed backends (Drake
``MultibodyPlant`` via its parser, Pinocchio ``buildModelFromXML``) can
consume pipeline rigs without engine-specific loaders.

Conventions:
- One revolute URDF joint per (rig joint, axis) pair, matching the DOF
  ordering of ``JointTrajectory`` frames (rig-dict order, then axis order).
- Multi-axis rig joints expand into chains through small intermediate links
  co-located at the joint origin.
- The root link is named ``{rig.id}_root``; callers that want a fixed base
  weld it to the world (Drake) or rely on the default fixed base (Pinocchio).
- Links carry small nonzero mass/inertia so engines that reject zero-inertia
  moving bodies (Drake) accept the model.
- A ``SimpleTransmission`` per DOF gives Drake one actuator per joint.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from .contracts import SkeletonRig

_AXIS_XYZ = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}

# Small but Drake-safe inertials. Values are deliberately generic: matching
# solvers regularize in joint space, so segment inertia acts as conditioning,
# not as a biomechanical claim.
_LINK_MASS = 1.0
_LINK_INERTIA = 1e-2
_DUMMY_MASS = 1e-2
_DUMMY_INERTIA = 1e-4
_DEFAULT_LIMIT = 3.141592653589793
_DEFAULT_EFFORT = 200.0
_DEFAULT_VELOCITY = 50.0


def _inertial(mass: float, inertia: float) -> str:
    return (
        "<inertial>"
        f'<mass value="{mass}"/>'
        f'<inertia ixx="{inertia}" ixy="0" ixz="0" '
        f'iyy="{inertia}" iyz="0" izz="{inertia}"/>'
        "</inertial>"
    )


def rig_to_urdf(rig: SkeletonRig, *, model_name: str | None = None) -> str:
    """Render ``rig`` as a URDF string.

    Args:
        rig: Canonical skeleton rig (validated by its own contracts).
        model_name: Optional URDF robot name; defaults to ``rig.id``.

    Returns:
        URDF XML with one revolute joint (+ transmission) per rig DOF.

    Raises:
        ValueError: If the rig has no DOFs (nothing to articulate).
    """
    if rig.num_dofs < 1:
        raise ValueError("rig must have at least one DOF to build a URDF")

    name = escape(model_name or rig.id)
    root_link = f"{rig.id}_root"
    parts: list[str] = [f'<robot name="{name}">']
    parts.append(
        f'<link name="{escape(root_link)}">{_inertial(_LINK_MASS, _LINK_INERTIA)}</link>'
    )

    for joint_name, joint_def in rig.joints.items():
        parent_link = (
            f"{joint_def.parent}_link" if joint_def.parent is not None else root_link
        )
        n_axes = len(joint_def.axes)
        for axis_idx, axis in enumerate(joint_def.axes):
            is_last = axis_idx == n_axes - 1
            child_link = (
                f"{joint_name}_link" if is_last else f"{joint_name}_dof{axis_idx}_link"
            )
            mass, inertia = (
                (_LINK_MASS, _LINK_INERTIA)
                if is_last
                else (_DUMMY_MASS, _DUMMY_INERTIA)
            )
            # The T-pose offset applies once, on the first DOF of the joint.
            if axis_idx == 0:
                ox, oy, oz = joint_def.tpose_offset
            else:
                ox = oy = oz = 0.0
            if axis_idx < len(joint_def.limits):
                lower = joint_def.limits[axis_idx].lower
                upper = joint_def.limits[axis_idx].upper
            else:
                lower, upper = -_DEFAULT_LIMIT, _DEFAULT_LIMIT
            urdf_joint = f"{joint_name}_dof{axis_idx}"

            parts.append(
                f'<link name="{escape(child_link)}">{_inertial(mass, inertia)}</link>'
            )
            parts.append(
                f'<joint name="{escape(urdf_joint)}" type="revolute">'
                f'<parent link="{escape(parent_link)}"/>'
                f'<child link="{escape(child_link)}"/>'
                f'<origin xyz="{ox} {oy} {oz}" rpy="0 0 0"/>'
                f'<axis xyz="{_AXIS_XYZ[str(axis)]}"/>'
                f'<limit lower="{lower}" upper="{upper}" '
                f'effort="{_DEFAULT_EFFORT}" velocity="{_DEFAULT_VELOCITY}"/>'
                "</joint>"
            )
            parts.append(
                '<transmission type="SimpleTransmission">'
                f'<actuator name="{escape(urdf_joint)}_act"/>'
                f'<joint name="{escape(urdf_joint)}"/>'
                "<mechanicalReduction>1</mechanicalReduction>"
                "</transmission>"
            )
            parent_link = child_link

    parts.append("</robot>")
    return "".join(parts)


def rig_root_link_name(rig: SkeletonRig) -> str:
    """Name of the URDF root link produced by :func:`rig_to_urdf`."""
    return f"{rig.id}_root"


def rig_joint_link_name(joint_name: str) -> str:
    """URDF link name carrying the body of rig joint ``joint_name``."""
    return f"{joint_name}_link"


def rig_to_pinocchio_model(rig: SkeletonRig):
    """Build a ``pin.Model`` from ``rig`` via the URDF bridge.

    The Pinocchio model has a fixed base at the rig root, one revolute DOF
    per rig DOF in rig order, and a frame named
    ``rig_joint_link_name(joint)`` for every rig joint.

    Raises:
        RuntimeError: When the pinocchio bindings are not installed.
    """
    try:
        from importlib.util import find_spec

        available = find_spec("pinocchio") is not None
    except (ValueError, ModuleNotFoundError):
        available = False
    if not available:
        raise RuntimeError(
            "pinocchio is not installed. Install the pinocchio extra: "
            "pip install 'upstream-drift[pinocchio]'"
        )
    import pinocchio as pin

    return pin.buildModelFromXML(rig_to_urdf(rig))
