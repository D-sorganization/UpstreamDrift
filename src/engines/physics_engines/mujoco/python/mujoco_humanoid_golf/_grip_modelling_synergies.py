"""Synergy classes and dialogs for linked control of joints.

Used in the Grip Modelling Tab.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from PyQt6 import QtCore, QtWidgets

# Named angle-limit constants (radians) for the default grip synergies.
# Previously these were inline literals scattered across six near-identical
# branches in ``rebuild_synergy_controls`` (issue #7723); centralising them
# gives a single source of truth.
SHADOW_FIST_MAX = 1.4
SHADOW_INDEX_MAX = 1.4
SHADOW_PINCH_FINGER_MAX = 1.0
SHADOW_PINCH_THUMB_MAX = 0.8
ALLEGRO_FIST_MAX = 1.5
ALLEGRO_INDEX_MAX = 1.5
ALLEGRO_PINCH_MAX = 1.0

# A joint spec is (joint_name, min_angle, max_angle); the resolver turns the
# joint name into a qpos address (or None when the joint is absent).
JointSpec = tuple[str, float, float]


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


def build_synergy_from_specs(
    name: str,
    specs: list[JointSpec],
    resolve: Callable[[str], int | None],
) -> Synergy | None:
    """Build a :class:`Synergy` from a list of joint specs.

    Single, data-driven replacement for the six previously duplicated
    Fist/Index/Pinch binding-build blocks (issue #7723). Each spec is resolved
    to a qpos address; specs whose joint is missing are skipped.

    Args:
        name: Display name of the synergy.
        specs: Ordered list of ``(joint_name, min_angle, max_angle)`` tuples.
        resolve: Callable mapping a joint name to a qpos address, or ``None``
            when the joint does not exist in the loaded model.

    Returns:
        A :class:`Synergy` when at least one spec resolves, else ``None``.
    """
    bindings: list[SynergyJointBinding] = []
    for joint_name, min_val, max_val in specs:
        q_adr = resolve(joint_name)
        if q_adr is not None:
            bindings.append(SynergyJointBinding(q_adr, min_val, max_val))
    if not bindings:
        return None
    return Synergy(name, bindings)


def shadow_synergy_specs(prefixes: list[str]) -> list[tuple[str, list[JointSpec]]]:
    """Build the ordered Shadow-hand synergy specs for the given prefixes.

    Behaviour-preserving extraction of the Shadow branch of
    ``rebuild_synergy_controls`` (issue #7723). Spec ordering matches the
    original nested-loop construction exactly so resolved binding order is
    unchanged.

    Args:
        prefixes: Hand prefixes such as ``["rh"]`` or ``["rh", "lh"]``.

    Returns:
        Ordered ``(synergy_name, specs)`` pairs for Fist/Index/Pinch.
    """
    fist: list[JointSpec] = [
        (f"{p}_{f}J{j}", 0.0, SHADOW_FIST_MAX)
        for p in prefixes
        for f in ("FF", "MF", "RF", "LF")
        for j in (3, 2, 1)
    ]
    index: list[JointSpec] = [
        (f"{p}_FFJ{j}", 0.0, SHADOW_INDEX_MAX) for p in prefixes for j in (3, 2, 1)
    ]
    pinch: list[JointSpec] = []
    for p in prefixes:
        pinch.extend((f"{p}_FFJ{j}", 0.0, SHADOW_PINCH_FINGER_MAX) for j in (3, 2, 1))
        pinch.extend((f"{p}_THJ{j}", 0.0, SHADOW_PINCH_THUMB_MAX) for j in (4, 3, 2, 1))
    return [
        ("Fist Curl", fist),
        ("Index Curl", index),
        ("Pinch Grip", pinch),
    ]


def allegro_synergy_specs(joint_names: list[str]) -> list[tuple[str, list[JointSpec]]]:
    """Build the ordered Allegro-hand synergy specs from model joint names.

    Behaviour-preserving extraction of the Allegro branch of
    ``rebuild_synergy_controls`` (issue #7723). Matching is substring-based
    against the lowercased joint name, preserving the original iteration order
    over ``joint_names``.

    Args:
        joint_names: All joint names enumerated from the loaded model.

    Returns:
        Ordered ``(synergy_name, specs)`` pairs for Fist/Index/Pinch.
    """
    fist_targets = (
        "ffj1",
        "ffj2",
        "ffj3",
        "mfj1",
        "mfj2",
        "mfj3",
        "rfj1",
        "rfj2",
        "rfj3",
    )
    index_targets = ("ffj1", "ffj2", "ffj3")
    pinch_thumb_targets = ("thj1", "thj2", "thj3")

    fist: list[JointSpec] = [
        (n, 0.0, ALLEGRO_FIST_MAX)
        for n in joint_names
        if any(t in n.lower() for t in fist_targets)
    ]
    index: list[JointSpec] = [
        (n, 0.0, ALLEGRO_INDEX_MAX)
        for n in joint_names
        if any(t in n.lower() for t in index_targets)
    ]
    pinch: list[JointSpec] = []
    for n in joint_names:
        if any(t in n.lower() for t in index_targets):
            pinch.append((n, 0.0, ALLEGRO_PINCH_MAX))
        if any(t in n.lower() for t in pinch_thumb_targets):
            pinch.append((n, 0.0, ALLEGRO_PINCH_MAX))
    return [
        ("Fist Curl", fist),
        ("Index Curl", index),
        ("Pinch Grip", pinch),
    ]


def resolve_hand_prefixes(model_name: str, *, is_shadow: bool) -> list[str]:
    """Resolve the joint-name prefixes for a hand model selection.

    Behaviour-preserving extraction of the prefix logic in
    ``rebuild_synergy_controls`` (issue #7723).

    Args:
        model_name: Lowercased hand-model display string.
        is_shadow: Whether the selected model is a Shadow hand.

    Returns:
        Ordered list of prefixes (e.g. ``["rh", "lh"]``).
    """
    if "both" in model_name:
        return ["rh", "lh"]
    if "right" in model_name:
        return ["rh" if is_shadow else "right"]
    if "left" in model_name:
        return ["lh" if is_shadow else "left"]
    return []


def build_default_synergies(
    model_name: str,
    resolve: Callable[[str], int | None],
    allegro_joint_names: list[str],
) -> list[Synergy]:
    """Build the default synergy list for a hand model (data-driven).

    Single entry point that replaces the CC-48 branch matrix in
    ``rebuild_synergy_controls`` (issue #7723). Pure: it depends only on the
    model name, a qpos resolver, and the enumerated joint names, so it is
    unit-testable without a live Qt tab.

    Args:
        model_name: Lowercased hand-model display string.
        resolve: Callable mapping a joint name to a qpos address (or ``None``).
        allegro_joint_names: Joint names from the model (only used for Allegro).

    Returns:
        Ordered list of resolvable :class:`Synergy` objects.
    """
    is_shadow = "shadow" in model_name
    is_allegro = "allegro" in model_name

    if is_shadow:
        prefixes = resolve_hand_prefixes(model_name, is_shadow=True)
        spec_groups = shadow_synergy_specs(prefixes)
    elif is_allegro:
        spec_groups = allegro_synergy_specs(allegro_joint_names)
    else:
        return []

    synergies: list[Synergy] = []
    for name, specs in spec_groups:
        synergy = build_synergy_from_specs(name, specs, resolve)
        if synergy is not None:
            synergies.append(synergy)
    return synergies
