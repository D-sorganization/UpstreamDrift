"""Synergy classes and dialogs for linked control of joints.

Used in the Grip Modelling Tab.
"""

from __future__ import annotations

import re

from PyQt6 import QtCore, QtWidgets


def get_descriptive_joint_name(name: str) -> str:
    """Convert raw joint name to a descriptive name.

    Args:
        name: Raw joint name from MuJoCo model.

    Returns:
        User-friendly descriptive joint label.
    """
    side_prefix = ""
    if name.startswith(("rh_", "right_")):
        side_prefix = "[Right] "
    elif name.startswith(("lh_", "left_")):
        side_prefix = "[Left] "

    clean_name = name
    for prefix in ["rh_", "lh_", "right_", "left_"]:
        if clean_name.lower().startswith(prefix):
            clean_name = clean_name[len(prefix) :]
            break

    finger_map = {
        "ff": "Index",
        "mf": "Middle",
        "rf": "Ring",
        "lf": "Little (Pinky)",
        "th": "Thumb",
        "wr": "Wrist",
    }

    match_shadow = re.match(r"^([a-zA-Z]+)J(\d+)$", clean_name)
    match_allegro = re.match(r"^([a-zA-Z]+)j(\d+)$", clean_name)

    if match_shadow:
        finger_code = match_shadow.group(1).lower()
        joint_num = int(match_shadow.group(2))
        finger_name = finger_map.get(finger_code, finger_code.upper())

        if finger_code == "wr":
            if joint_num == 1:
                return f"{side_prefix}Wrist Pitch / Flexion (WRJ1)"
            if joint_num == 2:
                return f"{side_prefix}Wrist Yaw / Abduction (WRJ2)"
        elif finger_code == "th":
            thumb_joints = {
                5: "CMC Abduction (THJ5)",
                4: "CMC Flexion (THJ4)",
                3: "MCP Flexion (THJ3)",
                2: "IP Flexion (THJ2)",
                1: "Distal Flexion (THJ1)",
            }
            desc = thumb_joints.get(joint_num, f"Joint {joint_num}")
            return f"{side_prefix}Thumb {desc}"
        elif finger_code == "lf" and joint_num == 5:
            return f"{side_prefix}Little (Pinky) CMC Flexion (LFJ5)"
        else:
            finger_joints = {
                4: "Knuckle Abduction (MCP) (J4)",
                3: "Knuckle Flexion (MCP) (J3)",
                2: "Middle Joint Flexion (PIP) (J2)",
                1: "Distal Joint Flexion (DIP) (J1)",
            }
            desc = finger_joints.get(joint_num, f"Joint {joint_num}")
            return f"{side_prefix}{finger_name} {desc}"

    elif match_allegro:
        finger_code = match_allegro.group(1).lower()
        joint_num = int(match_allegro.group(2))
        finger_name = finger_map.get(finger_code, finger_code.upper())

        if finger_code == "th":
            thumb_joints = {
                0: "CMC Abduction (thj0)",
                1: "CMC Flexion (thj1)",
                2: "MCP Flexion (thj2)",
                3: "IP Flexion (thj3)",
            }
            desc = thumb_joints.get(joint_num, f"Joint {joint_num}")
            return f"{side_prefix}Thumb {desc}"
        else:
            finger_joints = {
                0: "Knuckle Abduction (MCP) (j0)",
                1: "Knuckle Flexion (MCP) (j1)",
                2: "Middle Joint Flexion (PIP) (j2)",
                3: "Distal Joint Flexion (DIP) (j3)",
            }
            desc = finger_joints.get(joint_num, f"Joint {joint_num}")
            return f"{side_prefix}{finger_name} {desc}"

    return f"{side_prefix}{name}"


class SynergyJointBinding:
    """Binds a specific joint range to a synergy slider."""

    def __init__(self, qpos_adr: int, min_val: float, max_val: float) -> None:
        """Initialize binding.

        Args:
            qpos_adr: Address of the joint position in the MjData qpos array.
            min_val: Target angle at slider minimum (0.0).
            max_val: Target angle at slider maximum (1.0).
        """
        self.qpos_adr: int = qpos_adr
        self.min_val: float = min_val
        self.max_val: float = max_val


class Synergy:
    """Defines a linked joint synergy."""

    def __init__(self, name: str, bindings: list[SynergyJointBinding]) -> None:
        """Initialize synergy.

        Args:
            name: Synergy display name.
            bindings: List of SynergyJointBinding associations.
        """
        self.name: str = name
        self.bindings: list[SynergyJointBinding] = bindings


class AddSynergyDialog(QtWidgets.QDialog):
    """Dialog to define custom synergy bindings."""

    def __init__(
        self,
        joints: list[tuple[int, str, float, float]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize dialog.

        Args:
            joints: List of tuples (qpos_adr, descriptive_name, min_limit, max_limit).
            parent: Parent QWidget.
        """
        super().__init__(parent)
        self.setWindowTitle("Add Custom Synergy Slider")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.dialog_layout = QtWidgets.QVBoxLayout(self)

        # Name
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Synergy Name:"))
        self.txt_name = QtWidgets.QLineEdit("Custom Synergy")
        name_row.addWidget(self.txt_name)
        self.dialog_layout.addLayout(name_row)

        self.dialog_layout.addWidget(QtWidgets.QLabel("<b>Select Joints & Ranges:</b>"))

        # Scrollable joint list
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.rows: list[
            tuple[
                int,
                QtWidgets.QCheckBox,
                QtWidgets.QDoubleSpinBox,
                QtWidgets.QDoubleSpinBox,
            ]
        ] = []

        for q_adr, name, min_l, max_l in joints:
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            chk = QtWidgets.QCheckBox()
            row_layout.addWidget(chk)

            lbl = QtWidgets.QLabel(name)
            lbl.setFixedWidth(200)
            row_layout.addWidget(lbl)

            row_layout.addWidget(QtWidgets.QLabel("Min:"))
            spin_min = QtWidgets.QDoubleSpinBox()
            spin_min.setRange(min_l, max_l)
            spin_min.setValue(min_l)
            spin_min.setSingleStep(0.01)
            row_layout.addWidget(spin_min)

            row_layout.addWidget(QtWidgets.QLabel("Max:"))
            spin_max = QtWidgets.QDoubleSpinBox()
            spin_max.setRange(min_l, max_l)
            spin_max.setValue(max_l)
            spin_max.setSingleStep(0.01)
            row_layout.addWidget(spin_max)

            self.scroll_layout.addWidget(row_widget)
            self.rows.append((q_adr, chk, spin_min, spin_max))

        self.scroll_area.setWidget(self.scroll_widget)
        self.dialog_layout.addWidget(self.scroll_area)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        self.dialog_layout.addLayout(btn_layout)

    def get_synergy(self) -> Synergy | None:
        """Construct and return the Synergy mapping if accepted.

        Returns:
            A Synergy mapping, or None if no joints were selected.
        """
        name = self.txt_name.text().strip()
        if not name:
            name = "Custom Synergy"

        bindings: list[SynergyJointBinding] = []
        for q_adr, chk, spin_min, spin_max in self.rows:
            if chk.isChecked():
                bindings.append(
                    SynergyJointBinding(q_adr, spin_min.value(), spin_max.value())
                )

        if not bindings:
            return None

        return Synergy(name, bindings)
