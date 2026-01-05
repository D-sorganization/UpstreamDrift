"""Interactive URDF Generator Tool.

This tool allows users to graphically create URDF models by adding links and joints.
"""

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URDFGenerator(QtWidgets.QMainWindow):
    """Main window for the URDF Generator."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Graphic URDF Generator")
        self.resize(1200, 800)

        # Data model
        self.robot_name = "robot"
        self.links: list[dict] = []
        self.joints: list[dict] = []

        # UI Setup
        self._setup_ui()
        self._create_default_base()

    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left Panel: Controls
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setFixedWidth(400)

        # Robot Name
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Robot Name:"))
        self.name_input = QtWidgets.QLineEdit(self.robot_name)
        self.name_input.textChanged.connect(self._on_name_changed)
        name_layout.addWidget(self.name_input)
        left_layout.addLayout(name_layout)

        # Elements List (Tree)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Element", "Type"])
        left_layout.addWidget(self.tree)

        # Buttons
        btn_layout = QtWidgets.QGridLayout()
        self.add_link_btn = QtWidgets.QPushButton("Add Link")
        self.add_link_btn.clicked.connect(self._add_link_dialog)
        self.add_joint_btn = QtWidgets.QPushButton("Add Joint")
        self.add_joint_btn.clicked.connect(self._add_joint_dialog)
        self.remove_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected)

        btn_layout.addWidget(self.add_link_btn, 0, 0)
        btn_layout.addWidget(self.add_joint_btn, 0, 1)
        btn_layout.addWidget(self.remove_btn, 1, 0, 1, 2)
        left_layout.addLayout(btn_layout)

        # XML Preview
        left_layout.addWidget(QtWidgets.QLabel("URDF XML Preview:"))
        self.xml_preview = QtWidgets.QTextEdit()
        self.xml_preview.setReadOnly(True)
        self.xml_preview.setFont(QtGui.QFont("Courier", 9))
        left_layout.addWidget(self.xml_preview)

        # File Actions
        file_layout = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save URDF")
        self.save_btn.clicked.connect(self._save_urdf)
        self.load_btn = QtWidgets.QPushButton("Load URDF")
        self.load_btn.clicked.connect(self._load_urdf)
        file_layout.addWidget(self.save_btn)
        file_layout.addWidget(self.load_btn)
        left_layout.addLayout(file_layout)

        main_layout.addWidget(left_panel)

        # Right Panel: Visualization (Placeholder for now, or simple OpenGL/Matplotlib)
        # Ideally, we would reuse MuJoCoSimWidget here, but we need to generate MJCF/URDF first.
        # For this version, we will focus on the structure and XML generation.
        # We can add a "Visualize" button that launches the viewer.

        right_panel = QtWidgets.QGroupBox("Visualization")
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self.viz_label = QtWidgets.QLabel("Visualization requires MuJoCo engine.")
        self.viz_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.viz_label)

        self.viz_btn = QtWidgets.QPushButton("Visualize Current Model")
        self.viz_btn.clicked.connect(self._visualize_model)
        right_layout.addWidget(self.viz_btn)

        main_layout.addWidget(right_panel)

        self._update_xml_preview()

    def _create_default_base(self) -> None:
        """Create a default base link."""
        self.links.append(
            {
                "name": "base_link",
                "geometry_type": "box",
                "size": "0.5 0.5 0.1",
                "color": "0.8 0.8 0.8 1",
            }
        )
        self._refresh_tree()
        self._update_xml_preview()

    def _on_name_changed(self, text: str) -> None:
        self.robot_name = text
        self._update_xml_preview()

    def _refresh_tree(self) -> None:
        """Rebuild the tree widget."""
        self.tree.clear()

        # Links
        links_item = QtWidgets.QTreeWidgetItem(self.tree, ["Links"])
        for link in self.links:
            item = QtWidgets.QTreeWidgetItem(links_item, [link["name"], "Link"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, link)

        links_item.setExpanded(True)

        # Joints
        joints_item = QtWidgets.QTreeWidgetItem(self.tree, ["Joints"])
        for joint in self.joints:
            item = QtWidgets.QTreeWidgetItem(joints_item, [joint["name"], "Joint"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, joint)

        joints_item.setExpanded(True)

    def _add_link_dialog(self) -> None:
        """Show dialog to add a link."""
        dialog = LinkDialog(self)
        if dialog.exec():
            link_data = dialog.get_data()
            self.links.append(link_data)
            self._refresh_tree()
            self._update_xml_preview()

    def _add_joint_dialog(self) -> None:
        """Show dialog to add a joint."""
        # Get potential parent/child links
        link_names = [link["name"] for link in self.links]
        if len(link_names) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Not enough links", "You need at least 2 links to create a joint."
            )
            return

        dialog = JointDialog(link_names, self)
        if dialog.exec():
            joint_data = dialog.get_data()
            self.joints.append(joint_data)
            self._refresh_tree()
            self._update_xml_preview()

    def _remove_selected(self) -> None:
        """Remove selected item."""
        item = self.tree.currentItem()
        if not item or item.parent() is None:
            return

        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        parent_item = item.parent()
        if parent_item is not None:
            parent_text = parent_item.text(0)
        else:
            return

        if parent_text == "Links":
            self.links.remove(data)
            # Also remove connected joints? For now, keep simple.
        elif parent_text == "Joints":
            self.joints.remove(data)

        self._refresh_tree()
        self._update_xml_preview()

    def _generate_urdf_xml(self) -> str:
        """Generate URDF XML string."""
        root = ET.Element("robot", name=self.robot_name)

        # Materials (basic support)
        # Collect unique colors? For simplicity, define inline or reuse.

        for link in self.links:
            link_elem = ET.SubElement(root, "link", name=link["name"])

            # Visual
            visual = ET.SubElement(link_elem, "visual")
            geometry = ET.SubElement(visual, "geometry")

            g_type = link["geometry_type"]
            if g_type == "box":
                ET.SubElement(geometry, "box", size=link["size"])
            elif g_type == "cylinder":
                # Parse size "radius length"
                parts = link["size"].split()
                radius = parts[0] if len(parts) > 0 else "0.1"
                length = parts[1] if len(parts) > 1 else "0.5"
                ET.SubElement(geometry, "cylinder", radius=radius, length=length)
            elif g_type == "sphere":
                ET.SubElement(geometry, "sphere", radius=link["size"])

            material = ET.SubElement(visual, "material", name=f"mat_{link['name']}")
            ET.SubElement(material, "color", rgba=link.get("color", "0.8 0.8 0.8 1"))

            # Collision (duplicate visual)
            collision = ET.SubElement(link_elem, "collision")
            c_geometry = ET.SubElement(collision, "geometry")
            if g_type == "box":
                ET.SubElement(c_geometry, "box", size=link["size"])
            elif g_type == "cylinder":
                parts = link["size"].split()
                radius = parts[0] if len(parts) > 0 else "0.1"
                length = parts[1] if len(parts) > 1 else "0.5"
                ET.SubElement(c_geometry, "cylinder", radius=radius, length=length)
            elif g_type == "sphere":
                ET.SubElement(c_geometry, "sphere", radius=link["size"])

            # Inertial (dummy for now)
            inertial = ET.SubElement(link_elem, "inertial")
            ET.SubElement(inertial, "mass", value="1.0")
            ET.SubElement(
                inertial,
                "inertia",
                ixx="0.1",
                ixy="0",
                ixz="0",
                iyy="0.1",
                iyz="0",
                izz="0.1",
            )

        for joint in self.joints:
            joint_elem = ET.SubElement(
                root, "joint", name=joint["name"], type=joint["type"]
            )
            ET.SubElement(joint_elem, "parent", link=joint["parent"])
            ET.SubElement(joint_elem, "child", link=joint["child"])

            origin = joint.get("origin", "0 0 0")
            rpy = joint.get("rpy", "0 0 0")
            ET.SubElement(joint_elem, "origin", xyz=origin, rpy=rpy)

            if joint["type"] != "fixed":
                ET.SubElement(joint_elem, "axis", xyz=joint.get("axis", "1 0 0"))
                ET.SubElement(
                    joint_elem,
                    "limit",
                    lower="-3.14",
                    upper="3.14",
                    effort="100",
                    velocity="10",
                )

        # Pretty print
        try:
            import defusedxml.minidom as minidom

            rough_string = ET.tostring(root, "utf-8")
            reparsed = minidom.parseString(rough_string)
            return str(reparsed.toprettyxml(indent="  "))
        except ImportError:
            return str(ET.tostring(root, encoding="unicode"))

    def _update_xml_preview(self) -> None:
        self.xml_preview.setText(self._generate_urdf_xml())

    def _save_urdf(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save URDF", "", "URDF Files (*.urdf)"
        )
        if filename:
            with open(filename, "w") as f:
                f.write(self._generate_urdf_xml())
            QtWidgets.QMessageBox.information(
                self, "Saved", f"URDF saved to {filename}"
            )

    def _load_urdf(self) -> None:
        """Primitive URDF loader (basic parsing)."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load URDF", "", "URDF Files (*.urdf)"
        )
        if not filename:
            return

        try:
            tree = ET.parse(filename)
            root = tree.getroot()
            self.robot_name = root.attrib.get("name", "robot")
            self.name_input.setText(self.robot_name)

            self.links = []
            self.joints = []

            for link in root.findall("link"):
                name = link.attrib["name"]
                # Try to find visual geometry
                geom_type = "box"
                size = "0.1 0.1 0.1"
                color = "0.8 0.8 0.8 1"

                visual = link.find("visual")
                if visual is not None:
                    geometry = visual.find("geometry")
                    if geometry is not None:
                        box_elem = geometry.find("box")
                        if box_elem is not None:
                            geom_type = "box"
                            size = box_elem.attrib.get("size", size)
                        else:
                            cyl_elem = geometry.find("cylinder")
                            if cyl_elem is not None:
                                geom_type = "cylinder"
                                r = cyl_elem.attrib.get("radius", "0.1")
                                length_val = cyl_elem.attrib.get("length", "0.5")
                                size = f"{r} {length_val}"
                            else:
                                sphere_elem = geometry.find("sphere")
                                if sphere_elem is not None:
                                    geom_type = "sphere"
                                    size = sphere_elem.attrib.get("radius", "0.1")

                    material = visual.find("material")
                    if material is not None:
                        col = material.find("color")
                        if col is not None:
                            color = col.attrib.get("rgba", color)

                self.links.append(
                    {
                        "name": name,
                        "geometry_type": geom_type,
                        "size": size,
                        "color": color,
                    }
                )

            for joint in root.findall("joint"):
                name = joint.attrib["name"]
                jtype = joint.attrib.get("type", "revolute")
                parent_elem = joint.find("parent")
                child_elem = joint.find("child")

                if parent_elem is not None and child_elem is not None:
                    parent = parent_elem.attrib["link"]
                    child = child_elem.attrib["link"]
                else:
                    continue  # Skip malformed joints

                origin_elem = joint.find("origin")
                origin = "0 0 0"
                rpy = "0 0 0"
                if origin_elem is not None:
                    origin = origin_elem.attrib.get("xyz", origin)
                    rpy = origin_elem.attrib.get("rpy", rpy)

                axis = "1 0 0"
                axis_elem = joint.find("axis")
                if axis_elem is not None:
                    axis = axis_elem.attrib.get("xyz", axis)

                self.joints.append(
                    {
                        "name": name,
                        "type": jtype,
                        "parent": parent,
                        "child": child,
                        "origin": origin,
                        "rpy": rpy,
                        "axis": axis,
                    }
                )

            self._refresh_tree()
            self._update_xml_preview()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load URDF: {e}")

    def _visualize_model(self) -> None:
        """Launch a visualization window with the current URDF."""
        # Save to temp file
        temp_path = Path("temp_model.urdf")
        with open(temp_path, "w") as f:
            f.write(self._generate_urdf_xml())

        # Try to launch visualization
        try:
            # We can try to import MuJoCoSimWidget if available in pythonpath
            # Or assume we can launch it via subprocess if this tool is standalone

            # For now, let's try to import MuJoCoSimWidget from the suite
            sys.path.insert(0, str(Path(__file__).parents[2]))  # Add root to path
            from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import (
                MuJoCoSimWidget,
            )

            # Create a dialog to hold the widget
            self.viz_dialog = QtWidgets.QDialog(self)
            self.viz_dialog.setWindowTitle("Model Visualization")
            self.viz_dialog.resize(800, 600)
            layout = QtWidgets.QVBoxLayout(self.viz_dialog)

            sim_widget = MuJoCoSimWidget(width=800, height=600)
            layout.addWidget(sim_widget)

            # Load the model
            sim_widget.load_model_from_file(str(temp_path.absolute()))

            self.viz_dialog.show()

            # Remove temp file after closing? Or keep for debugging.

        except ImportError:
            QtWidgets.QMessageBox.warning(
                self,
                "Visualization Error",
                "Could not import MuJoCo engine components.",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Visualization Error", f"Error launching visualization: {e}"
            )


class LinkDialog(QtWidgets.QDialog):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Link")
        layout = QtWidgets.QFormLayout(self)

        self.name_input = QtWidgets.QLineEdit()
        layout.addRow("Name:", self.name_input)

        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["box", "cylinder", "sphere"])
        layout.addRow("Type:", self.type_combo)

        self.size_input = QtWidgets.QLineEdit("0.1 0.1 0.1")
        layout.addRow("Size (xyz / r l / r):", self.size_input)

        self.color_input = QtWidgets.QLineEdit("0.8 0.8 0.8 1")
        layout.addRow("Color (rgba):", self.color_input)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_data(self) -> dict:
        return {
            "name": self.name_input.text(),
            "geometry_type": self.type_combo.currentText(),
            "size": self.size_input.text(),
            "color": self.color_input.text(),
        }


class JointDialog(QtWidgets.QDialog):
    def __init__(self, links: list[str], parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Joint")
        layout = QtWidgets.QFormLayout(self)

        self.name_input = QtWidgets.QLineEdit()
        layout.addRow("Name:", self.name_input)

        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["revolute", "fixed", "continuous", "prismatic"])
        layout.addRow("Type:", self.type_combo)

        self.parent_combo = QtWidgets.QComboBox()
        self.parent_combo.addItems(links)
        layout.addRow("Parent Link:", self.parent_combo)

        self.child_combo = QtWidgets.QComboBox()
        self.child_combo.addItems(links)
        layout.addRow("Child Link:", self.child_combo)

        self.origin_input = QtWidgets.QLineEdit("0 0 0")
        layout.addRow("Origin (xyz):", self.origin_input)

        self.rpy_input = QtWidgets.QLineEdit("0 0 0")
        layout.addRow("RPY (rad):", self.rpy_input)

        self.axis_input = QtWidgets.QLineEdit("1 0 0")
        layout.addRow("Axis (xyz):", self.axis_input)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_data(self) -> dict:
        return {
            "name": self.name_input.text(),
            "type": self.type_combo.currentText(),
            "parent": self.parent_combo.currentText(),
            "child": self.child_combo.currentText(),
            "origin": self.origin_input.text(),
            "rpy": self.rpy_input.text(),
            "axis": self.axis_input.text(),
        }


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = URDFGenerator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
