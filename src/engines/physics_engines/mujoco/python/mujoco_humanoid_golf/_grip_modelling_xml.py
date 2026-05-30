"""XML preprocessing helpers for the Grip Modelling hand models."""

from __future__ import annotations

import re
from pathlib import Path

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

def prefix_hand_assets(content: str, hand_prefix: str) -> str:
    """Prefix all mesh and material definitions.

    Avoids collisions by prefixing assets and their references.

    Args:
        content: The XML content of the hand model.
        hand_prefix: The prefix to apply (e.g. 'rh' or 'lh').

    Returns:
        The XML content with prefixed assets and references.
    """
    # 1. Identify and prefix materials
    material_names = re.findall(r'<material\s+[^>]*name="([^"]+)"', content)
    for mat in material_names:
        if mat.startswith(hand_prefix + "_"):
            continue
        new_mat = f"{hand_prefix}_{mat}"
        content = re.sub(
            rf'(<material\s+[^>]*name=)"{mat}"', rf'\1"{new_mat}"', content
        )
        content = re.sub(rf'(\bmaterial=)"{mat}"', rf'\1"{new_mat}"', content)

    # 2. Identify and prefix meshes
    mesh_tags = re.findall(r"<mesh\s+([^>]+)/>", content)
    for tag_attrs in mesh_tags:
        file_match = re.search(r'file="([^"]+)"', tag_attrs)
        name_match = re.search(r'name="([^"]+)"', tag_attrs)

        if name_match:
            mesh_name = name_match.group(1)
        elif file_match:
            file_path = file_match.group(1)
            mesh_name = Path(file_path).stem
        else:
            continue

        if mesh_name.startswith(hand_prefix + "_"):
            continue

        new_mesh_name = f"{hand_prefix}_{mesh_name}"

        # Replace the old mesh tag with named mesh tag
        old_tag = f"<mesh {tag_attrs}/>"
        if name_match:
            new_attrs = re.sub(
                rf'name="{mesh_name}"', f'name="{new_mesh_name}"', tag_attrs
            )
        else:
            new_attrs = f'name="{new_mesh_name}" {tag_attrs}'
        new_tag = f"<mesh {new_attrs}/>"
        content = content.replace(old_tag, new_tag)

        # Replace mesh references in geoms
        content = re.sub(rf'(\bmesh=)"{mesh_name}"', rf'\1"{new_mesh_name}"', content)

    return content


def get_hand_content(
    folder_path: Path,
    filename: str,
    body_name_pattern: str,
    is_both: bool,
) -> str:
    """Read a hand XML file, inject freejoint, and strip mujoco tags.

    Args:
        folder_path: Path to the hand assets directory.
        filename: Name of the hand XML file to load.
        body_name_pattern: Name of the root hand body (e.g. rh_forearm).
        is_both: True if both right and left hands are loaded side-by-side.

    Returns:
        The processed hand XML content as a string.
    """
    if folder_path is None:
        raise ValueError("folder_path must be provided")
    full_path = folder_path / filename
    if not full_path.exists():
        return ""

    try:
        content = full_path.read_text("utf-8")

        # Check if freejoint already exists
        if "freejoint" not in content:
            pattern = f'(<body[^>]*name="{body_name_pattern}"[^>]*>)'
            match = re.search(pattern, content)
            if match:
                logger.info("Injecting freejoint into %s", filename)
                insertion = match.group(1) + "\n      <freejoint/>"
                content = content.replace(match.group(1), insertion)
            else:
                logger.warning(
                    "Could not find body '%s' in %s to inject freejoint",
                    body_name_pattern,
                    filename,
                )

        # Strip <mujoco> tags to allow embedding
        content = re.sub(r"<mujoco[^>]*>", "", content)
        content = content.replace("</mujoco>", "")

        # When merging both hands, prefix default class names to avoid collisions
        if is_both:
            hand_prefix = "right" if "right" in filename.lower() else "left"
            # Find all default class names
            class_names = re.findall(r'<default class="([^"]+)">', content)
            for class_name in set(class_names):
                new_name = f"{hand_prefix}_{class_name}"
                content = content.replace(
                    f'class="{class_name}"', f'class="{new_name}"'
                )

            # Prefix meshes and materials to avoid asset collisions
            hand_prefix_short = "rh" if "right" in filename.lower() else "lh"
            content = prefix_hand_assets(content, hand_prefix_short)

        return content
    except (RuntimeError, ValueError, OSError):
        logger.exception("Failed to process hand file %s", filename)
        return ""


def inline_hand_includes(
    xml_content: str,
    scene_path: Path,
    folder_path: Path,
    is_both: bool,
) -> str:
    """Inline hand XML includes and inject extracted bodies into worldbody.

    Args:
        xml_content: Current XML content of the scene.
        scene_path: Path to the scene file.
        folder_path: Path to the hand assets directory.
        is_both: True if both hands are active side-by-side.

    Returns:
        The processed scene XML content.
    """
    if xml_content is None:
        raise ValueError("xml_content must be provided")
    extracted_bodies: list[str] = []
    extracted_post_bodies: list[str] = []

    def extract_sections(filename: str, body_pattern: str) -> str:
        """Extract worldbody and post-worldbody XML content from a hand model."""
        if filename is None:
            raise ValueError("filename must be provided")
        content = get_hand_content(folder_path, filename, body_pattern, is_both)

        # Extract worldbody
        bodies_match = re.search(
            r"<worldbody[^>]*>(.*?)</worldbody>", content, re.DOTALL
        )
        if bodies_match:
            extracted_bodies.append(bodies_match.group(1))
            content = re.sub(
                r"<worldbody[^>]*>.*?</worldbody>", "", content, flags=re.DOTALL
            )

        # Extract post-worldbody elements to prevent out-of-order XML crashes
        for tag in ["contact", "tendon", "actuator", "equality"]:
            tag_match = re.search(f"<{tag}[^>]*>(.*?)</{tag}>", content, re.DOTALL)
            if tag_match:
                extracted_post_bodies.append(f"<{tag}>\n{tag_match.group(1)}\n</{tag}>")
                content = re.sub(
                    f"<{tag}[^>]*>.*?</{tag}>", "", content, flags=re.DOTALL
                )

        return content

    if is_both:
        right_defs = extract_sections("right_hand.xml", "rh_forearm")
        left_defs = extract_sections("left_hand.xml", "lh_forearm")

        xml_content = re.sub(
            r'<include[^>]*file="right_hand.xml"[^>]*/>', right_defs, xml_content
        )
        xml_content = re.sub(
            r'<include[^>]*file="left_hand.xml"[^>]*/>', left_defs, xml_content
        )
    else:
        if 'file="right_hand.xml"' in xml_content:
            target_body = "rh_forearm"
            if "allegro" in str(folder_path).lower():
                target_body = "right_hand"

            defs = extract_sections("right_hand.xml", target_body)
            xml_content = re.sub(
                r'<include[^>]*file="right_hand.xml"[^>]*/>',
                defs,
                xml_content,
            )
        elif 'file="left_hand.xml"' in xml_content:
            target_body = "lh_forearm"
            if "allegro" in str(folder_path).lower():
                target_body = "left_hand"

            defs = extract_sections("left_hand.xml", target_body)
            xml_content = re.sub(
                r'<include[^>]*file="left_hand.xml"[^>]*/>',
                defs,
                xml_content,
            )

    # Inject extracted bodies into the scene's worldbody
    if extracted_bodies:
        bodies_str = "\n".join(extracted_bodies)
        xml_content = re.sub(
            r"(<worldbody[^>]*>)", r"\1\n" + bodies_str, xml_content, count=1
        )

    # Append post-worldbody elements to the end of the file
    if extracted_post_bodies:
        post_str = "\n".join(extracted_post_bodies) + "\n"
        if "</mujoco>" in xml_content:
            xml_content = xml_content.replace("</mujoco>", post_str + "</mujoco>")
        else:
            xml_content += post_str

    return xml_content


def ensure_offscreen_visual(xml_content: str) -> str:
    """Ensure the XML has offscreen framebuffer settings for rendering.

    Args:
        xml_content: Current XML content.

    Returns:
        The processed XML content.
    """
    offscreen_global = '<global offwidth="1920" offheight="1080"/>'
    if "<visual>" in xml_content:
        if "<global" in xml_content:

            def update_global_tag(m: re.Match) -> str:
                """Replace offscreen render dimensions in a global tag."""
                attrs = m.group(1).replace("/", "").strip()
                attrs = re.sub(r'offwidth="[^"]*"', "", attrs)
                attrs = re.sub(r'offheight="[^"]*"', "", attrs)
                return f'<global {attrs} offwidth="1920" offheight="1080"/>'

            xml_content = re.sub(
                r"<global([^>]*)>", update_global_tag, xml_content, count=1
            )
        else:
            xml_content = xml_content.replace(
                "<visual>",
                f"<visual>\n    {offscreen_global}",
            )
    else:
        xml_content = xml_content.replace(
            "</mujoco>",
            f"<visual>\n  {offscreen_global}\n</visual>\n</mujoco>",
        )
    return xml_content


def inject_cylinder_object(xml_content: str) -> str:
    """Inject a cylinder grip object into the scene if not present.

    Args:
        xml_content: Current XML content.

    Returns:
        The processed XML content.
    """
    if (
        "club_handle" not in xml_content
        and 'name="club_handle"' not in xml_content
        and 'name="object"' not in xml_content
    ):
        cylinder_body = """
    <body name="club_handle" pos="0.3 0 0.1">
      <freejoint/>
      <geom type="cylinder" size="0.015 0.15" rgba="0.8 0.2 0.2 1"
            mass="0.3" condim="4" friction="1 0.5 0.5"/>
    </body>
        """
        last_worldbody_end = xml_content.rfind("</worldbody>")
        if last_worldbody_end != -1:
            xml_content = (
                xml_content[:last_worldbody_end]
                + f"{cylinder_body}\n  "
                + xml_content[last_worldbody_end:]
            )
    return xml_content


def inject_mocap_bodies(xml_content: str, scene_path: Path, is_both: bool) -> str:
    """Inject mocap bodies and weld constraints for hand positioning.

    Args:
        xml_content: Current XML content.
        scene_path: Path to the scene.xml file.
        is_both: True if both right and left hands are loaded side-by-side.

    Returns:
        The processed XML content.
    """
    if xml_content is None:
        raise ValueError("xml_content must be provided")
    mocap_xml = ""
    equality_xml = "<equality>\n"

    rh_pos = "0 -0.18 0.05" if is_both else "0 0 0"
    lh_pos = "0 0.18 0.05" if is_both else "0 0 0"

    # Right Hand Mocap (only add if not already present)
    if (
        is_both or "right" in str(scene_path).lower()
    ) and 'name="rh_mocap"' not in xml_content:
        mocap_xml += f"""
    <body name="rh_mocap" mocap="true" pos="{rh_pos}">
        <geom type="box" size="0.02 0.02 0.02" rgba="0 1 0 0.5" contype="0"
              conaffinity="0"/>
    </body>
        """
        equality_xml += (
            '    <weld body1="rh_mocap" body2="rh_forearm" solref="0.02 1" '
            'solimp="0.9 0.95 0.001"/>\n'
        )

    # Left Hand Mocap (only add if not already present)
    if (
        is_both or "left" in str(scene_path).lower()
    ) and 'name="lh_mocap"' not in xml_content:
        mocap_xml += f"""
    <body name="lh_mocap" mocap="true" pos="{lh_pos}">
        <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 0.5" contype="0"
              conaffinity="0"/>
    </body>
        """
        equality_xml += (
            '    <weld body1="lh_mocap" body2="lh_forearm" solref="0.02 1" '
            'solimp="0.9 0.95 0.001"/>\n'
        )

    equality_xml += "  </equality>"

    # Insert Mocap bodies before the last </worldbody>
    if mocap_xml:
        last_worldbody_end = xml_content.rfind("</worldbody>")
        if last_worldbody_end != -1:
            xml_content = (
                xml_content[:last_worldbody_end]
                + f"{mocap_xml}\n  "
                + xml_content[last_worldbody_end:]
            )

    # Insert Equality section before </mujoco> (or merge if exists)
    if "</equality>" in xml_content:
        equality_content = (
            equality_xml.strip()
            .replace("<equality>", "")
            .replace("</equality>", "")
        )
        xml_content = xml_content.replace(
            "</equality>", f"{equality_content}\n  </equality>"
        )
    else:
        xml_content = xml_content.replace("</mujoco>", f"{equality_xml}\n</mujoco>")

    return xml_content


def prepare_scene_xml(
    scene_path: Path, folder_path: Path, is_both: bool = False
) -> str:
    """Read scene file and inject absolute paths and cylinder object.

    Args:
        scene_path: Path to the main scene file.
        folder_path: Path to the folder containing hand assets.
        is_both: True if both hands are active side-by-side.

    Returns:
        Prepared XML content.
    """
    if scene_path is None:
        raise ValueError("scene_path must be provided")
    xml_content = scene_path.read_text("utf-8")

    # 1. Inline hand XML includes and extract worldbodies
    xml_content = inline_hand_includes(xml_content, scene_path, folder_path, is_both)

    # 2. Ensure offscreen framebuffer is large enough for renderer
    xml_content = ensure_offscreen_visual(xml_content)

    # 3. Inject Cylinder Object (only if not present)
    xml_content = inject_cylinder_object(xml_content)

    # 4. Inject Mocap Bodies and Welds for Hands
    xml_content = inject_mocap_bodies(xml_content, scene_path, is_both)

    logger.info("Successfully prepared scene XML with movable hands and mocap bodies.")
    return xml_content
