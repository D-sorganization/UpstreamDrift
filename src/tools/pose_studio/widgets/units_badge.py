"""Read-only badge showing the active engine's native pose convention.

Displayed in the top-right of the main window so the user can tell at
a glance what convention the engine reports its ``q`` vector in (e.g.
``"Drake URDF / RPY (rad)"``, ``"OpenSim / coordinates (rad)"``).

The badge does NOT change the values in the joint panel — those are
always displayed in the canonical convention (degrees by default,
toggleable to radians via :meth:`JointPanel.set_show_radians`).  The
badge simply tells the user which engine's Jacobian / FK is feeding
the live kinematics service under the hood.
"""

from __future__ import annotations

from PyQt6 import QtWidgets

from src.shared.python.theme.style_constants import Styles

# Hard-coded native-convention strings per engine.  These mirror the
# Joint-Slot units reported by each adapter and the engine's own
# documentation; they are not derived at runtime because the badge is
# purely informational.
_ENGINE_CONVENTIONS: dict[str, str] = {
    "drake": "Drake URDF / RPY (rad)",
    "mujoco": "MuJoCo MJCF / Euler (rad)",
    "pinocchio": "Pinocchio URDF / RPY (rad)",
    "opensim": "OpenSim .osim / coordinates (rad)",
    "simscape": "Simscape Parameters / RPY (deg)",
}


class UnitsBadge(QtWidgets.QLabel):
    """Read-only :class:`QLabel` showing the engine's native convention."""

    def __init__(
        self,
        engine_name: str = "drake",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(Styles.STATUS_INFO)
        self.setToolTip(
            "The active engine's native pose convention. Joint sliders "
            "below always show angles in the canonical convention "
            "(degrees by default; toggle radians from the View menu)."
        )
        self.set_engine(engine_name)

    def set_engine(self, engine_name: str) -> None:
        """Update the badge text for *engine_name*."""
        if not isinstance(engine_name, str):
            raise TypeError(
                f"engine_name must be str, got {type(engine_name).__name__}"
            )
        text = _ENGINE_CONVENTIONS.get(engine_name, f"{engine_name} (unknown)")
        self.setText(text)


__all__ = ["UnitsBadge"]
