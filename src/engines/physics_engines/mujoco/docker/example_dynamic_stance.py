from __future__ import annotations

import logging
import os

import dm_control
import imageio
import numpy as np
from dm_control import mjcf, suite

logger = logging.getLogger(__name__)

# Path to CMU Humanoid XMl in the container
# We need to find it first. Usually in dm_control/suite/humanoid_CMU.xml
# But strictly speaking we should use the suite to get the model structure if possible?
# Re-loading via mjcf is cleaner for editing.
# Let's assume standard install path or try to locate it.
# Fallback: We can modify the `suite.load` behavior by patching? No.
# WE WILL USE REFLECTION to find the xml path from the loaded env if possible,
# or just assume the standard path in site-packages.

# Target Pose: Address Position
TARGET_POSE = {
    "lowerbackrx": 0.35,
    "upperbackrx": 0.15,
    "rtibiarx": 0.1,
    "ltibiarx": 0.1,
    "rfemurrx": -0.2,
    "lfemurrx": -0.2,
    "rfootrx": -0.05,
    "lfootrx": -0.05,
    # Arms closer together (holding club)
    "rhumerusrx": -0.4,
    "lhumerusrx": -0.4,  # More forward
    "rhumerusrz": -0.4,
    "lhumerusrz": 0.4,  # Rotate in towards body
    "rhumerusry": -0.2,
    "lhumerusry": 0.2,  # Twist
    "rradiusrx": 0.5,
    "lradiusrx": 0.5,  # Bent elbows slightly
}


def get_cmu_xml_path() -> str:
    """Locate the CMU Humanoid XML file."""
    # Heuristic to find the XML
    suite_dir = os.path.dirname(dm_control.suite.__file__)
    return os.path.join(suite_dir, "humanoid_CMU.xml")


def pd_control(physics, target_pose, actuators, kp=10.0, kd=1.0) -> np.ndarray:
    """Compute PD control action."""
    action = np.zeros(physics.model.nu)
    for joint_name, target_angle in target_pose.items():
        try:
            current_q = physics.named.data.qpos[joint_name]
            current_v = physics.named.data.qvel[joint_name]
            error = target_angle - current_q
            torque = (kp * error) - (kd * current_v)
            if joint_name in actuators:
                action[actuators[joint_name]] = torque
        except (RuntimeError, ValueError, AttributeError):
            pass
    return action


def customize_model(physics) -> None:
    """Apply colors and geometric adjustments."""
    # Grey Shirt
    GREY_SHIRT = [0.6, 0.6, 0.6, 1.0]
    # Darker Brown Pants
    BROWN_PANTS = [0.4, 0.2, 0.0, 1.0]
    BLACK_SHOES = [0.1, 0.1, 0.1, 1.0]
    SKIN_TONE = [0.8, 0.6, 0.4, 1.0]
    WHITE_EYES = [1.0, 1.0, 1.0, 1.0]
    # Silver Club
    SILVER_CLUB = [0.8, 0.8, 0.8, 1.0]

    geometry_names = physics.model.id2name
    ngeom = physics.model.ngeom

    for i in range(ngeom):
        name = geometry_names(i, "geom")
        if not name:
            continue
        name = name.lower()

        # Eyes
        if "eye" in name:
            physics.model.geom_rgba[i] = WHITE_EYES

        # Club
        elif "golf_club" in name:
            physics.model.geom_rgba[i] = SILVER_CLUB

        # Skin (Head/Hands)
        elif "head" in name or "hand" in name:
            physics.model.geom_rgba[i] = SKIN_TONE

        # Shirt
        # Explicitly added upperback
        elif any(
            part in name
            for part in [
                "chest",
                "rib",
                "sternum",
                "clavicle",
                "shoulder",
                "arm",
                "humerus",
                "radius",
                "wrist",
                "scapula",
                "torso",
                "abdomen",
                "waist",
                "spine",
                "thorax",
                "back",
                "lumbar",
                "upperback",
            ]
        ):
            physics.model.geom_rgba[i] = GREY_SHIRT

        # Pants (Legs + Butt)
        # Added 'hipjoint' which seemed to be the butt in the logs
        elif any(
            part in name
            for part in [
                "thigh",
                "shin",
                "leg",
                "femur",
                "tibia",
                "fibula",
                "calf",
                "crus",
                "patella",
                "butt",
                "pelvis",
                "sacrum",
                "glute",
                "ilium",
                "ischium",
                "hipjoint",
                "root_geom",
            ]
        ):
            physics.model.geom_rgba[i] = BROWN_PANTS

        # Shoes
        elif any(part in name for part in ["foot", "feet", "toe", "ankle"]):
            physics.model.geom_rgba[i] = BLACK_SHOES


def _load_and_patch_xml(xml_path) -> mjcf.Physics:
    """Load the CMU Humanoid XML, patch it, and return compiled physics.

    Returns the compiled physics object, or None if patching fails.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found at {xml_path}")

    logger.info("Found XML: %s", xml_path)

    with open(xml_path) as f:
        xml_string = f.read()

    logger.info("--- XML HEADER DEBUG ---")
    logger.info("%s", "\n".join(xml_string.split("\n")[:20]))
    logger.info("------------------------")

    # Build assets dictionary for included files
    suite_dir = os.path.dirname(xml_path)
    common_dir = os.path.join(suite_dir, "common")
    assets = {}
    for filename in ["skybox.xml", "visual.xml", "materials.xml"]:
        path = os.path.join(common_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                assets[f"./common/{filename}"] = f.read()

    # FIX XML ERROR: "repeated default class name"
    xml_string = xml_string.replace('class="main"', 'class="main_custom"')

    # Load directly from string with assets
    root = mjcf.from_xml_string(xml_string, assets=assets)

    # Add Golf Club
    _attach_golf_club(root)

    # Add Custom "Face-On" Camera
    _add_face_on_camera(root)

    # Compile
    physics = mjcf.Physics.from_mjcf_model(root)
    logger.info("Successfully compiled modified model with Golf Club and Camera!")
    return physics


def _attach_golf_club(root) -> None:
    """Attach a golf club geometry to the right hand body."""
    rhand = root.find("body", "rhand")
    if rhand:
        logger.info("Attaching golf club to rhand...")
        # Club Shaft (Silver)
        rhand.add(
            "geom",
            name="golf_club_shaft",
            type="cylinder",
            size=[0.02, 0.5],
            pos=[0, 0, -0.6],
            quat=[1, 0, 0, 0],
            rgba=[0.8, 0.8, 0.8, 1],
            mass=0.5,
        )
        # Club Head (Silver/Dark)
        rhand.add(
            "geom",
            name="golf_club_head",
            type="box",
            size=[0.06, 0.12, 0.04],
            pos=[0, 0.05, -1.1],
            rgba=[0.7, 0.7, 0.7, 1],
        )


def _add_face_on_camera(root) -> None:
    """Add a face-on camera to the worldbody."""
    worldbody = root.find("worldbody", "world")
    if worldbody:
        logger.info("Adding 'face_on' camera...")
        worldbody.add(
            "camera",
            name="face_on",
            pos=[2.5, 0, 1.4],
            mode="targetbody",
            target="root",
        )


def _setup_physics(xml_path) -> mjcf.Physics:
    """Set up physics, falling back to suite.load if patching fails."""
    try:
        physics = _load_and_patch_xml(xml_path)
    except ImportError as e:
        logger.error("Warning: MJCF patching failed: %s", e)
        logger.info("Falling back to standard suite.load (No Club/Camera).")
        env = suite.load(domain_name="humanoid_CMU", task_name="stand")
        physics = env.physics
    return physics


def _find_face_on_camera(physics) -> int:
    """Find the face_on camera id, defaulting to 0."""
    logger.info("\nAvailable Cameras:")
    ncam = physics.model.ncam
    camera_id = 0
    for i in range(ncam):
        name = physics.model.id2name(i, "camera")
        logger.info("Camera %s: %s", i, name)
        if name == "face_on":
            camera_id = i
    return camera_id


def _set_initial_pose(physics) -> None:
    """Reset physics and set the initial address pose."""
    with physics.reset_context():
        # Z-height 0.96 adjusted empirically for CMU model to ensure feet
        # contact ground.
        physics.data.qpos[2] = 0.96

        # Set initial joint angles DIRECTLY to Target Pose
        # This prevents the "T-pose" start.
        for joint, angle in TARGET_POSE.items():
            try:
                if joint in physics.named.data.qpos:
                    physics.named.data.qpos[joint] = angle
            except (RuntimeError, ValueError, AttributeError):
                pass


def _run_simulation_loop(physics, actuators, camera_id) -> None:
    """Run the simulation loop, recording frames and saving video."""
    logger.info("Simulating...")
    frames = []
    fps = 30
    duration_sec = 8  # Running longer
    steps = duration_sec * fps

    for i in range(steps):
        # Control
        action = pd_control(
            physics, TARGET_POSE, actuators, kp=40.0, kd=4.0
        )  # Tweaked gains for holding pose

        # Apply control
        physics.set_control(action)

        # Step physics
        physics.step()

        # Render
        pixels = physics.render(height=480, width=640, camera_id=camera_id)
        frames.append(pixels)

        if i % 30 == 0:
            logger.info("Frame %s/%s", i, steps)

    filename = "humanoid_dynamic_stance.mp4"
    imageio.mimsave(filename, frames, fps=fps)
    logger.info("Saved to %s", filename)


def main() -> None:
    """Run the dynamic stance example."""
    logger.info("Locating CMU Humanoid MJCF...")
    xml_path = get_cmu_xml_path()

    # Setup physics (with XML patching or fallback)
    physics = _setup_physics(xml_path)

    # Map actuators
    actuators = {}
    for i in range(physics.model.nu):
        name = physics.model.id2name(i, "actuator")
        if name:
            actuators[name] = i

    # Find camera
    camera_id = _find_face_on_camera(physics)

    # Customize (Colors + Size)
    customize_model(physics)

    # Reset and SET INITIAL POSE
    _set_initial_pose(physics)

    # Run simulation loop and save video
    _run_simulation_loop(physics, actuators, camera_id)


if __name__ == "__main__":
    main()
