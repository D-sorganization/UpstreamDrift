import imageio
from dm_control import suite

# Define key poses for a golf swing (simplified kinematic targets)
# The CMU humanoid has 56 joints. We will target major joints for the swing.
# Joint names in CMU model are typically like 'R_Shoulder', 'L_Elbow', etc.
# We will use a dictionary of target angles.
# Note: These angles are rough approximations for visual effect.

# Correct CMU Joint Names derived from logs
# Hips/Legs: lfemur*, ltibia*, lfoot*
# Arms: lhumerus*, lradius*, lwrist*, lhand*
# Spine: lowerback*, upperback*, thorax*

POSES = {
    "Address": {
        # Bend forward
        "lowerbackrx": 0.3,
        "upperbackrx": 0.1,
        # Arms down (neutral)
        "rhumerusrx": -0.2,
        "lhumerusrx": -0.2,
        # knees bent
        "rtibiarx": 0.3,
        "ltibiarx": 0.3,
        "rfemurrx": -0.4,
        "lfemurrx": -0.4,
    },
    "Backswing": {
        # Rotate torso right (z-axis negative?)
        "lowerbackrz": -0.4,
        "upperbackrz": -0.4,
        "thoraxrz": -0.4,
        # Arms up and right
        "rhumerusrx": -1.2,
        "rhumerusrz": 0.5,
        "rradiusrx": 1.5,  # Right elbow bent
        "lhumerusrx": -1.0,
        "lhumerusrz": 0.8,  # Left arm straight-ish across chest
        # Weight shift
        "rfemurrx": -0.1,
        "lfemurrx": -0.5,
    },
    "Downswing": {
        # Unwind torso
        "lowerbackrz": 0.0,
        "upperbackrz": 0.0,
        "thoraxrz": 0.0,
        # Arms dropping
        "rhumerusrx": -0.5,
        "lhumerusrx": -0.5,
    },
    "Impact": {
        # Torso left
        "lowerbackrz": 0.2,
        "upperbackrz": 0.2,
        # Arms straight down
        "rhumerusrx": -0.1,
        "lhumerusrx": -0.1,
        "rradiusrx": 0.1,
    },
    "FollowThrough": {
        # Full turn left
        "lowerbackrz": 0.8,
        "upperbackrz": 0.8,
        # Arms high left
        "rhumerusrx": -1.5,
        "rhumerusrz": -0.5,
        "lhumerusrx": -1.5,
        "lhumerusrz": -1.0,
        "lradiusrx": 1.5,  # Left elbow bent
        # Finish on left leg
        "rfemurrx": -0.2,
        "rtibiarx": 0.8,  # Right foot up/knee bent pivot
        "lfemurrx": 0.0,
        "ltibiarx": 0.1,
    },
}

SWING_SEQUENCE = ["Address", "Backswing", "Downswing", "Impact", "FollowThrough"]
DURATION = [50, 60, 20, 10, 60]  # ticks per phase


def get_cmu_joint_names(env) -> list[str]:
    """Retrieve joint names for the CMU model."""
    # This is a best-effort inspection.
    # If explicit names aren't in named.data, we might need a hardcoded
    # mapping or try to print them.
    try:
        names = env.physics.named.data.qpos.axes.row.names
        return list(names)
    except Exception:
        return []


def interpolate_pose(start_pose, end_pose, alpha) -> dict[str, float]:
    """Linearly interpolate between two poses."""
    result = start_pose.copy()
    for joint, end_val in end_pose.items():
        start_val = start_pose.get(joint, 0.0)
        result[joint] = start_val + (end_val - start_val) * alpha
    return dict(result)


def main() -> None:
    """Run the golf swing example."""
    print("Loading humanoid_CMU:stand task...")
    try:
        env = suite.load(domain_name="humanoid_CMU", task_name="stand")
    except Exception as e:
        print(f"Error loading humanoid_CMU: {e}")
        return

    # Print available joints to help debugging/tuning
    print("Available joints (qpos lines):")
    joint_names = get_cmu_joint_names(env)
    for name in joint_names:
        print(name)

    # Initialize simulation
    env.reset()
    colorize_humanoid(env)

    frames = []

    # Current state of joints (starts at 0)
    current_pose: dict[str, float] = {}

    print("Simulating golf swing...")

    total_steps = 0
    for i, phase_name in enumerate(SWING_SEQUENCE):
        target_pose = POSES[phase_name]
        steps = DURATION[i]

        start_pose = current_pose.copy()

        for step in range(steps):
            # Use steps-1 to ensure we reach alpha=1.0 at the end of the phase
            denom = max(1, steps - 1)
            alpha = step / denom
            interpolated = interpolate_pose(start_pose, target_pose, alpha)
            current_pose = interpolated

            # Apply pose to physics
            # Note: qpos includes position (xyz) and orientation (quat) of root.
            # We need to be careful not to overwrite the root if we don't want to.
            # However, for kinematic animation, we can just write to known
            # named joints.
            # Apply pose directly to qpos
            for joint_name, angle in interpolated.items():
                try:
                    if joint_name in env.physics.named.data.qpos:
                        env.physics.named.data.qpos[joint_name] = angle
                except Exception:
                    pass

            # Kinematics update (computes geoms/sites positions from qpos)
            # We do NOT use step() because we are manually forcing the pose.
            env.physics.forward()

            # Render
            pixels = env.physics.render(height=480, width=640, camera_id=0)

            # We can print the phase to console.

            frames.append(pixels)
            total_steps += 1

        print(f"Completed phase: {phase_name}")

    print(f"Captured {len(frames)} frames.")

    filename = "humanoid_golf_swing.mp4"
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved to {filename}")


def colorize_humanoid(env) -> None:
    """Apply custom colors to the humanoid key body parts."""
    print("Applying custom outfit colors...")
    # Colors (R, G, B, A)
    BLUE_SHIRT = [0.0, 0.0, 1.0, 1.0]
    BROWN_PANTS = [0.6, 0.3, 0.0, 1.0]
    BLACK_SHOES = [0.1, 0.1, 0.1, 1.0]

    # Try to find geoms that match body parts
    # Note: env.physics.model.geom_rgba has shape (ngeom, 4)
    # We iter over names to find indices.

    try:
        geometry_names = env.physics.model.id2name
        ngeom = env.physics.model.ngeom

        for i in range(ngeom):
            name = geometry_names(i, "geom")
            if not name:
                continue
            name = name.lower()

            # Heuristic matching
            # Shirt: Upper body including neck, thorax, back
            if any(
                part in name
                for part in [
                    "torso",
                    "chest",
                    "rib",
                    "sternum",
                    "clavicle",
                    "shoulder",
                    "arm",
                    "abdomen",
                    "waist",
                    "spine",
                    "thorax",
                    "neck",
                    "back",
                    "humerus",
                    "radius",
                    "wrist",
                ]
            ):
                env.physics.model.geom_rgba[i] = BLUE_SHIRT

            # Pants: Lower body starting from hips/pelvis down to ankles
            elif any(
                part in name
                for part in [
                    "thigh",
                    "shin",
                    "leg",
                    "pelvis",
                    "butt",
                    "hip",
                    "femur",
                    "tibia",
                ]
            ):
                env.physics.model.geom_rgba[i] = BROWN_PANTS

            # Shoes: Feet
            elif any(part in name for part in ["foot", "feet", "toe", "ankle"]):
                env.physics.model.geom_rgba[i] = BLACK_SHOES

    except Exception as e:
        print(f"Error coloring model: {e}")


if __name__ == "__main__":
    main()
