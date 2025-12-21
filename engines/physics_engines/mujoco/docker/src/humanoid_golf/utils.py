import os

import dm_control.suite
import numpy as np
from dm_control import mjcf

SHIRT_PARTS = [
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

PANTS_PARTS = [
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


def get_cmu_xml_path() -> str:
    """Locate the CMU Humanoid XML file within dm_control."""
    suite_dir = os.path.dirname(dm_control.suite.__file__)
    xml_path = os.path.join(suite_dir, "humanoid_CMU.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"Could not find 'humanoid_CMU.xml' at expected location: {xml_path}"
        )
    return xml_path


def get_actuator_indices(physics) -> dict[str, int]:
    """Map actuator names to their indices."""
    mapping = {}
    for i in range(physics.model.nu):
        name = physics.model.id2name(i, "actuator")
        if name:
            mapping[name] = i
    return mapping


def load_humanoid_with_props(
    target_height=1.8,
    weight_percent=100.0,
    club_params=None,
    two_handed=False,
    enhance_face=False,
    articulated_fingers=False,
) -> mjcf.Physics:
    """
    Load the CMU humanoid with updated props and features.
    """

    xml_path = get_cmu_xml_path()

    # Read XML
    with open(xml_path) as f:
        xml_string = f.read()

    # FIX ASSET PATHS
    suite_dir = os.path.dirname(xml_path)
    common_dir = os.path.join(suite_dir, "common")

    assets = {}
    for filename in ["skybox.xml", "visual.xml", "materials.xml"]:
        path = os.path.join(common_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                assets[f"./common/{filename}"] = f.read()

    # FIX XML ERROR
    xml_string = xml_string.replace('class="main"', 'class="main_custom"')

    # Load MJCF
    root = mjcf.from_xml_string(xml_string, assets=assets)

    # --- SCALING FACTORS ---
    height_scale = target_height / 1.56
    width_scale = np.sqrt(weight_percent / 100.0) * height_scale

    print(f"Scaling Height by {height_scale:.3f} (Target: {target_height}m)")
    print(f"Scaling Width by {width_scale:.3f} (Weight: {weight_percent}%)")

    # Recursively scale positions and size
    for body in root.find_all("body"):
        pos = getattr(body, "pos", None)
        if pos is not None:
            body.pos = [x * height_scale for x in pos]

    for geom in root.find_all("geom"):
        # Scale size
        size = getattr(geom, "size", None)
        if size is not None:
            new_size = []
            if len(size) == 1:  # Sphere
                new_size = [size[0] * width_scale]
            elif len(size) == 2:  # Capsule/Cylinder
                new_size = [size[0] * width_scale, size[1] * height_scale]
            elif len(size) == 3:  # Box
                new_size = [
                    size[0] * width_scale,
                    size[1] * width_scale,
                    size[2] * height_scale,
                ]
            geom.size = new_size

        # Scale local pos
        pos = getattr(geom, "pos", None)
        if pos is not None:
            geom.pos = [x * height_scale for x in pos]

    # Scale Joint anchors (pos)
    for joint in root.find_all("joint"):
        pos = getattr(joint, "pos", None)
        if pos is not None:
            joint.pos = [x * height_scale for x in pos]

    # Scale Sites
    for site in root.find_all("site"):
        pos = getattr(site, "pos", None)
        if pos is not None:
            site.pos = [x * height_scale for x in pos]

    # --- ENHANCEMENTS ---
    if enhance_face:
        _add_face_features(root, height_scale, width_scale)

    if articulated_fingers:
        _add_articulated_fingers(root, height_scale, width_scale)

    # --- ATTACH CLUB ---
    if club_params is None:
        club_params = {"length": 1.0, "mass": 0.5, "head_size": 1.0}

    _attach_club(root, height_scale, width_scale, club_params, two_handed)

    # --- CAMERAS ---
    if root.worldbody:
        root.worldbody.add(
            "camera",
            name="face_on",
            pos=[2.5 * height_scale, 0, 1.4 * height_scale],
            mode="targetbody",
            target="root",
        )

    return mjcf.Physics.from_mjcf_model(root)


def _add_face_features(root, h_scale, w_scale) -> None:
    """Add facial features like nose and mouth."""
    head = root.find("body", "head")
    if not head:
        return

    # Add Nose
    head.add(
        "geom",
        name="nose",
        type="capsule",
        size=[0.01 * w_scale, 0.02 * h_scale],
        pos=[0.1 * h_scale, 0, -0.05 * h_scale],
        quat=[0.707, 0, 0.707, 0],
        rgba=[0.8, 0.6, 0.4, 1],
    )

    # Add Mouth
    head.add(
        "geom",
        name="mouth",
        type="capsule",
        size=[0.005 * w_scale, 0.03 * w_scale],
        pos=[0.09 * h_scale, 0, -0.12 * h_scale],
        quat=[1, 0, 0, 0],
        rgba=[0.6, 0.3, 0.3, 1],
    )


def _add_articulated_fingers(root, h_scale, w_scale) -> None:
    """Add articulated fingers to the model."""
    for side in ["l", "r"]:
        hand = root.find("body", f"{side}hand")
        if not hand:
            continue

        # Clean up legacy actuators for the removed fingers
        # Attempts to find actuators targeting the old 'lfingers/rfingers' joints
        # NOTE: Must do this BEFORE removing the fingers body/joints
        to_remove = []
        for act in root.find_all("actuator"):
            # Ensure act.joint is valid and has a name attribute
            if act.joint and act.joint.name and f"{side}fingers" in act.joint.name:
                to_remove.append(act)

        for act in to_remove:
            act.remove()

        fingers = hand.find("body", f"{side}fingers")
        if fingers:
            hand.remove(fingers)

        palm = hand.add("body", name=f"{side}palm", pos=[0, 0, 0])

        # Finger joints
        # Range in radians: -90 deg is approx -1.57 rad
        for i in range(1, 5):
            parent = palm.add(
                "body",
                name=f"{side}finger{i}",
                pos=[0.02 * i * w_scale, -0.05 * h_scale, 0],
            )
            parent.add(
                "joint",
                name=f"{side}finger{i}_joint",
                type="hinge",
                axis=[1, 0, 0],
                range=[-1.57, 0],
                stiffness=50,
                damping=1,
            )
            parent.add(
                "geom",
                name=f"{side}finger{i}_geom",
                type="box",
                size=[0.015 * w_scale, 0.03 * w_scale, 0.025 * h_scale],
                rgba=[0.8, 0.6, 0.4, 1],
                pos=[0, 0, -0.025 * h_scale],
            )


def _attach_club(root, h_scale, w_scale, params, two_handed) -> None:
    """Attach the golf club to the model."""
    rhand = root.find("body", "rhand")
    if not rhand:
        return

    length = params.get("length", 1.0) * h_scale
    mass = params.get("mass", 0.5)

    # Shaft
    # Position adjusted so the grip is inside rhand
    rhand.add(
        "geom",
        name="golf_club_shaft",
        type="cylinder",
        size=[0.015 * w_scale, length / 2],
        pos=[0, 0.05 * w_scale, -length / 2 - 0.05 * h_scale],
        quat=[1, 0, 0, 0],
        rgba=[0.8, 0.8, 0.8, 1],
        mass=mass * 0.8,
    )

    # Head
    rhand.add(
        "geom",
        name="golf_club_head",
        type="box",
        size=[0.06 * w_scale, 0.1 * w_scale, 0.03 * h_scale],
        pos=[0, 0.1 * w_scale, -length - 0.05 * h_scale],
        rgba=[0.3, 0.3, 0.3, 1],
        mass=mass * 0.2,
    )

    if two_handed:
        lhand = root.find("body", "lhand")
        if lhand:
            # Site on Club (attached to rhand)
            # Locate it where left hand should be (slightly above right hand?)
            rhand.add(
                "site", name="club_grip_site", pos=[0, 0.05 * w_scale, -0.15 * h_scale]
            )

            # Site on Left Hand (palm center)
            lhand.add("site", name="lhand_grip_site", pos=[0, 0, -0.05 * h_scale])

            # Exclude contact between hands to avoid explosion
            contact = root.find("contact")
            if contact is None:
                contact = root.add("contact")
            contact.add("exclude", body1="rhand", body2="lhand")

            # Equality connect
            equality = root.find("equality")
            if equality is None:
                equality = root.add("equality")
            equality.add("connect", site1="lhand_grip_site", site2="club_grip_site")


def customize_visuals(physics, config=None) -> None:
    """Apply colors and visual tweaks."""
    # Defaults
    colors = {
        "shirt": [0.6, 0.6, 0.6, 1.0],
        "pants": [0.4, 0.2, 0.0, 1.0],
        "shoes": [0.1, 0.1, 0.1, 1.0],
        "skin": [0.8, 0.6, 0.4, 1.0],
        "eyes": [1.0, 1.0, 1.0, 1.0],
        "club": [0.8, 0.8, 0.8, 1.0],
    }

    if config and "colors" in config:
        for k, v in config["colors"].items():
            if k in colors:
                colors[k] = v

    geometry_names = physics.model.id2name
    ngeom = physics.model.ngeom

    for i in range(ngeom):
        name = geometry_names(i, "geom")
        if not name:
            continue
        name = name.lower()

        if "eye" in name:
            physics.model.geom_rgba[i] = colors["eyes"]
        elif "golf_club" in name:
            physics.model.geom_rgba[i] = colors["club"]
        elif "nose" in name or "mouth" in name:
            continue
        elif "head" in name or "hand" in name or "finger" in name or "thumb" in name:
            physics.model.geom_rgba[i] = colors["skin"]
        elif "floor" in name or "ground" in name:
            # Golf Course Green
            physics.model.geom_rgba[i] = [0.1, 0.6, 0.1, 1.0]
        elif any(part in name for part in SHIRT_PARTS):
            physics.model.geom_rgba[i] = colors["shirt"]
        elif any(part in name for part in PANTS_PARTS):
            physics.model.geom_rgba[i] = colors["pants"]
        elif any(part in name for part in ["foot", "feet", "toe", "ankle"]):
            physics.model.geom_rgba[i] = colors["shoes"]

    # Thicker arms
    for i in range(ngeom):
        name = geometry_names(i, "geom")
        if not name:
            continue
        name = name.lower()
        if any(part in name for part in ["arm", "humerus", "radius", "wrist"]):
            physics.model.geom_size[i][0] *= 1.1
