"""
Pure-data constants and helpers for the TorqueHistoryWidget.

Extracted from torque_history_widget.py so they can be imported and
tested without PyQt6 or any display server.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Trace colour palette — cycles through for any number of joints
# ---------------------------------------------------------------------------
_DRIVE_COLORS = [
    (230, 120, 50),  # warm orange
    (120, 180, 230),  # cool blue
    (180, 120, 220),  # purple
    (60, 200, 120),  # green
    (220, 180, 60),  # gold
    (100, 200, 200),  # teal
    (220, 100, 160),  # pink
]
_FRICTION_COLORS = [
    (200, 80, 80),  # red
    (80, 160, 160),  # teal
    (160, 80, 180),  # purple
    (80, 160, 80),  # green
    (180, 160, 80),  # olive
    (80, 120, 180),  # blue
    (180, 80, 120),  # rose
]
_TOTAL_COLORS = [
    (255, 220, 80),  # gold
    (180, 255, 180),  # pale green
    (220, 180, 255),  # lavender
    (180, 255, 220),  # mint
    (255, 220, 180),  # peach
    (180, 220, 255),  # sky
    (255, 180, 200),  # blush
]

# Okabe-Ito colorblind-safe palette (https://jfly.uni-koeln.de/color/)
# Each tuple: (R, G, B)
_CB_DRIVE_COLORS = [
    (0, 114, 178),  # blue
    (230, 159, 0),  # orange
    (0, 158, 115),  # green
    (204, 121, 167),  # pink
    (86, 180, 233),  # sky blue
    (213, 94, 0),  # vermillion
    (240, 228, 66),  # yellow
]

_CB_FRICTION_COLORS = [
    (0, 80, 130),  # dark blue
    (170, 110, 0),  # dark orange
    (0, 110, 80),  # dark green
    (150, 85, 120),  # dark pink
    (60, 130, 170),  # dark sky blue
    (155, 65, 0),  # dark vermillion
    (175, 165, 45),  # dark yellow
]

_CB_TOTAL_COLORS = [
    (80, 170, 230),  # light blue
    (255, 200, 80),  # light orange
    (80, 210, 165),  # light green
    (230, 170, 200),  # light pink
    (140, 210, 255),  # light sky blue
    (240, 150, 80),  # light vermillion
    (255, 245, 130),  # light yellow
]

# Active palette selection (False = default, True = colorblind-safe)
_use_colorblind_palette: bool = False


def set_colorblind_mode(enabled: bool) -> None:
    """Toggle between default and colorblind-safe color palettes.

    Pre: enabled is a boolean.
    Post: global palette selection is updated.
    """
    global _use_colorblind_palette
    _use_colorblind_palette = bool(enabled)


def get_drive_colors() -> list[tuple[int, int, int]]:
    """Return the active drive torque color palette."""
    return list(_CB_DRIVE_COLORS if _use_colorblind_palette else _DRIVE_COLORS)


def get_friction_colors() -> list[tuple[int, int, int]]:
    """Return the active friction torque color palette."""
    return list(_CB_FRICTION_COLORS if _use_colorblind_palette else _FRICTION_COLORS)


def get_total_colors() -> list[tuple[int, int, int]]:
    """Return the active total torque color palette."""
    return list(_CB_TOTAL_COLORS if _use_colorblind_palette else _TOTAL_COLORS)


# Joint labels per model
_JOINT_LABELS_2 = ["Shoulder", "Wrist"]
_JOINT_LABELS_3 = ["Shoulder", "Elbow", "Wrist"]
_JOINT_LABELS_7 = [
    "Hub",
    "R Shoulder",
    "R Elbow",
    "R Wrist",
    "L Shoulder",
    "L Elbow",
    "L Wrist",
]


def _joint_labels_for_ndof(n_joints: int) -> list[str]:
    """Return joint labels based on DOF count.

    Preconditions:
        n_joints > 0
    Postconditions:
        Returns a list of length n_joints.
    """
    if not (n_joints > 0):
        raise ValueError(f"n_joints must be positive, got {n_joints}")
    if n_joints == 2:
        return list(_JOINT_LABELS_2)
    if n_joints == 3:
        return list(_JOINT_LABELS_3)
    if n_joints == 7:
        return list(_JOINT_LABELS_7)
    result = [f"Joint {i + 1}" for i in range(n_joints)]
    if not (len(result) == n_joints):
        raise ValueError("DbC Blocked: Precondition failed.")
    return result
