"""MJCF exporter from canonical YAML specification."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class MJCFExporter:
    """Export MJCF from canonical YAML model specification."""

    def __init__(self, yaml_path: Path | str) -> None:
        """Initialize MJCF exporter.

        Args:
            yaml_path: Path to canonical YAML specification
        """
        self.yaml_path = Path(yaml_path)
        with self.yaml_path.open() as f:
            self.spec = yaml.safe_load(f)

        # Build children map for O(1) lookup
        self._children_map: dict[str, list[dict[str, typing.Any]]] = {}
        for segment in self.spec.get("segments", []):
            parent = segment.get("parent")
            if parent:
                if parent not in self._children_map:
                    self._children_map[parent] = []
                self._children_map[parent].append(segment)

    def export(self, output_path: Path | str) -> None:
        """Export MJCF file.

        Args:
            output_path: Path to output MJCF file
        """
        output = Path(output_path)
        mjcf_content = self._generate_mjcf()
        output.write_text(mjcf_content, encoding="utf-8")
        logger.info("Exported MJCF to %s", output)

    def _generate_mjcf(self) -> str:
        """Generate MJCF XML content.

        Returns:
            MJCF XML string
        """
        lines = ['<mujoco model="golfer">']
        lines.append("  <!-- Generated from canonical YAML specification -->")

        # Options
        lines.append(
            '  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>'
        )

        # Visual
        lines.append("  <visual>")
        lines.append('    <global offwidth="1024" offheight="1024"/>')
        lines.append('    <map znear="0.01" zfar="50"/>')
        lines.append("  </visual>")

        # Worldbody
        lines.append("  <worldbody>")
        lines.append(
            '    <geom name="floor" type="plane" size="10 10 0.1" '
            'rgba="0.8 0.8 0.8 1"/>'
        )

        # Root body
        root = self.spec["root"]
        root_pos = root.get("position", [0.0, 0.0, 0.9])
        lines.append(
            f'    <body name="{root["name"]}" '
            f'pos="{root_pos[0]} {root_pos[1]} {root_pos[2]}">'
        )
        lines.extend(self._generate_body_geom(root))
        lines.extend(self._generate_segments_mjcf(root["name"]))
        lines.append("    </body>")

        lines.append("  </worldbody>")
        lines.append("</mujoco>")
        return "\n".join(lines)

    def _generate_segments_mjcf(self, parent_name: str, depth: int = 1) -> list[str]:
        """Generate MJCF for segments recursively.

        Args:
            parent_name: Parent body name
            depth: Indentation depth

        Returns:
            List of MJCF lines
        """
        lines = []
        indent = "  " * (depth + 1)

        for segment in self._children_map.get(parent_name, []):
            seg_name = segment["name"]
            frame_offset = segment.get("frame_offset", [0.0, 0.0, 0.0])
            joint = segment.get("joint", {})

            lines.append(
                f'{indent}<body name="{seg_name}" '
                f'pos="{frame_offset[0]} {frame_offset[1]} {frame_offset[2]}">'
            )

            # Joint
            joint_type = joint.get("type", "hinge")
            if joint_type == "revolute":
                axis = joint.get("axis", [0, 0, 1])
                limits = joint.get("limits", [-3.14, 3.14])
                damping = joint.get("damping", 0.0)
                lines.append(
                    f'{indent}  <joint name="{seg_name}_joint" type="hinge" '
                    f'axis="{axis[0]} {axis[1]} {axis[2]}" '
                    f'range="{limits[0]} {limits[1]}" damping="{damping}"/>'
                )
            elif joint_type == "ball":
                lines.append(f'{indent}  <joint name="{seg_name}_joint" type="ball"/>')
            elif joint_type == "fixed":
                lines.append(f'{indent}  <joint name="{seg_name}_joint" type="fixed"/>')

            # Geometry
            lines.extend(
                [indent + "  " + line for line in self._generate_body_geom(segment)]
            )

            # Recursive children
            lines.extend(self._generate_segments_mjcf(seg_name, depth + 1))

            lines.append(f"{indent}</body>")

        return lines

    def _generate_body_geom(self, body: dict[str, typing.Any]) -> list[str]:
        """Generate geometry for body.

        Args:
            body: Body specification

        Returns:
            List of MJCF lines
        """
        lines = []
        geom = body.get("geometry", {})
        geom_type = geom.get("type", "box")
        rgba = geom.get("visual_rgba", [0.5, 0.5, 0.5, 1.0])
        rgba_str = f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"

        if geom_type == "box":
            size = geom.get("size", [0.1, 0.1, 0.1])
            lines.append(
                f'      <geom name="{body["name"]}_geom" type="box" '
                f'size="{size[0]} {size[1]} {size[2]}" rgba="{rgba_str}" '
                f'mass="{body["mass"]}"/>'
            )
        elif geom_type == "sphere":
            size = geom.get("size", 0.1)
            lines.append(
                f'      <geom name="{body["name"]}_geom" type="sphere" '
                f'size="{size}" rgba="{rgba_str}" mass="{body["mass"]}"/>'
            )
        elif geom_type in ("cylinder", "capsule"):
            size = geom.get("size", [0.1, 0.1])
            geom_tag = "cylinder" if geom_type == "cylinder" else "capsule"
            lines.append(
                f'      <geom name="{body["name"]}_geom" type="{geom_tag}" '
                f'size="{size[0]} {size[1]}" rgba="{rgba_str}" '
                f'mass="{body["mass"]}"/>'
            )

        return lines
