import os

import dm_control
import imageio
import numpy as np
from dm_control import mjcf, suite

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
        except Exception:
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


def main() -> None:
    """Run the dynamic stance example."""
    print("Locating CMU Humanoid MJCF...")
    xml_path = get_cmu_xml_path()

    physics = None
    env = None

    try:
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found at {xml_path}")

        print(f"Found XML: {xml_path}")

        # READ AND PATCH XML
        with open(xml_path) as f:
            xml_string = f.read()

        print("--- XML HEADER DEBUG ---")
        print("\n".join(xml_string.split("\n")[:20]))
        print("------------------------")

        # FIX ASSET PATHS
        # The XML uses <include file="./common/skybox.xml"/>
        # We need to resolve these relative to the XML's directory.
        suite_dir = os.path.dirname(xml_path)
        common_dir = os.path.join(suite_dir, "common")

        # Create an assets dictionary mapping filenames to content/paths?
        # dm_control expects assets dict: { 'filename': binary_content } usually.
        # But we can also pass the 'dir' to a loader? No, from_xml_string
        # takes 'assets'.

        # Easier fix: replace './common/' in the XML string with absolute path?
        # No, mujoco doesn't always like absolute paths in includes if
        # security dictating.
        # But let's try replacing './common/' with the absolute path string.
        # xml_string = xml_string.replace('./common/', f'{common_dir}/')

        # BETTER: Use mjcf.from_path(xml_path) but patch the ElementTree
        # *after* loading?
        # But the error "repeated default class name" happened *during* load
        # (validation).

        # Let's try supplying the assets dictionary.
        assets = {}
        for filename in ["skybox.xml", "visual.xml", "materials.xml"]:
            path = os.path.join(common_dir, filename)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    # Key must match the include file attribute exactly?
                    # <include file="./common/skybox.xml"/> -> key "./common/skybox.xml"
                    assets[f"./common/{filename}"] = f.read()

        # FIX XML ERROR: "repeated default class name"
        xml_string = xml_string.replace('class="main"', 'class="main_custom"')

        # Load directly from string with assets
        root = mjcf.from_xml_string(xml_string, assets=assets)

        # If we got here, great. If not, the previous code failed here.
        # The PREVIOUS trace failed at `mjcf.from_path(xml_path)`.

        # Add Golf Club
        rhand = root.find("body", "rhand")
        if rhand:
            print("Attaching golf club to rhand...")
            # Club Shaft (Silver)
            # Reduced length 0.9 -> 0.5 (Total length ~1m)
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
            )  # Adjusted pos for shorter shaft

        # Add Custom "Face-On" Camera
        worldbody = root.find("worldbody", "world")
        # Assuming Humanoid faces +X or +Y?
        # "back" camera is usually behind. "side" is side.
        # Let's try placing one at X=3, Y=0, Z=1.5 looking at target.
        if worldbody:
            print("Adding 'face_on' camera...")
            # mode="targetbody" target="root"
            worldbody.add(
                "camera",
                name="face_on",
                pos=[2.5, 0, 1.4],
                mode="targetbody",
                target="root",
            )

        # Compile
        physics = mjcf.Physics.from_mjcf_model(root)
        print("Successfully compiled modified model with Golf Club and Camera!")

    except Exception as e:
        print(f"⚠️ Warning: MJCF patching failed: {e}")
        print("Falling back to standard suite.load (No Club/Camera).")
        env = suite.load(domain_name="humanoid_CMU", task_name="stand")
        physics = env.physics

    # Map actuators
    actuators = {}
    for i in range(physics.model.nu):
        name = physics.model.id2name(i, "actuator")
        if name:
            actuators[name] = i

    # List Cameras
    print("\nAvailable Cameras:")
    ncam = physics.model.ncam
    camera_id = 0
    for i in range(ncam):
        name = physics.model.id2name(i, "camera")
        print(f"Camera {i}: {name}")
        if name == "face_on":
            camera_id = i

    # Customize (Colors + Size)
    customize_model(physics)

    # Reset and SET INITIAL POSE
    with physics.reset_context():
        # Set Root Position (Height)
        # Root is typically first qpos [0-6]
        # Z-height is index 2.
        # T-pose default is high. Let's start him lower so feet touch ground.
        # Trial and error: 1.3 was too high. Try 0.95 (CMU model is often
        # smaller/different scale).
        # Z-height 0.96 adjusted empirically for CMU model to ensure feet
        # contact ground.
        physics.data.qpos[2] = 0.96

        # Set initial joint angles DIRECTLY to Target Pose
        # This prevents the "T-pose" start.
        for joint, angle in TARGET_POSE.items():
            try:
                # Direct named access is preferred and safer than .get() which
                # returns value
                if joint in physics.named.data.qpos:
                    physics.named.data.qpos[joint] = angle
            except Exception:
                pass

    print("Simulating...")
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
        # Uses the camera_id found above (defaults to 0, sets to 'face_on'
        # index if found)
        pixels = physics.render(height=480, width=640, camera_id=camera_id)
        frames.append(pixels)

        if i % 30 == 0:
            print(f"Frame {i}/{steps}")

    filename = "humanoid_dynamic_stance.mp4"
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    main()
