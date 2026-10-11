"""Force and torque overlay manager for Simscape 3D viewer (ADR-0052, #11305)."""

from __future__ import annotations

import logging
from pathlib import Path

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from PyQt6 import QtWidgets

from src.engines.simscape.force_channels import load_simscape_force_series
from src.shared.python.force_overlay.contracts import (
    AxialLoadFrame,
    ForceTorqueFrame,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.matplotlib_glyphs import (
    draw_glyphs_3d,
    draw_legend,
)
from src.shared.python.force_overlay.series import ForceTorqueSeries

logger = logging.getLogger(__name__)


def format_legend_text(legend: LegendSpec) -> str:
    """Format LegendSpec into concise status/overlay text."""
    parts: list[str] = []
    if legend.force_reference_n is not None:
        parts.append(f"Scale: {legend.force_reference_n:g} N")
    if legend.torque_reference_nm is not None:
        parts.append(f"{legend.torque_reference_nm:g} N·m")
    if legend.kinds_present:
        parts.append(f"Kinds: {', '.join(k.value for k in legend.kinds_present)}")
    if legend.unavailable_labels:
        parts.append(f"Unavailable: {len(legend.unavailable_labels)}")
    return " | ".join(parts) if parts else "No active forces"


class ViewerForceOverlayManager:
    """Manages force/torque loading, filtering toggles, and 3D glyph rendering."""

    def __init__(self, parent_widget: QtWidgets.QWidget | None = None) -> None:
        self._parent = parent_widget
        self._series: ForceTorqueSeries | None = None
        self._missing_channels: tuple[str, ...] = ()
        self._artists: list[Artist] = []
        self._legend_inset: Axes | None = None
        self._status_note: str = ""

        # UI Toggles
        self.check_forces = QtWidgets.QCheckBox("Forces")
        self.check_forces.setObjectName("check_force_glyphs")
        self.check_forces.setChecked(True)
        self.check_forces.setToolTip("Toggle force arrow display")

        self.check_torques = QtWidgets.QCheckBox("Torques")
        self.check_torques.setObjectName("check_torque_glyphs")
        self.check_torques.setChecked(True)
        self.check_torques.setToolTip("Toggle torque arc display")

        self.check_grip = QtWidgets.QCheckBox("Grip / Hand")
        self.check_grip.setObjectName("check_grip_glyphs")
        self.check_grip.setChecked(True)
        self.check_grip.setToolTip("Toggle grip/hand force & couple display")

        self.label_legend = QtWidgets.QLabel("No force data")
        self.label_legend.setObjectName("label_force_legend")
        self.label_legend.setStyleSheet(
            "color: #888888; font-size: 11px; padding: 2px;"
        )

        # Style with default palette
        self._style = ForceGlyphStyle(
            force_scale_m_per_n=0.002,
            torque_scale_m_per_nm=0.005,
            min_length_m=0.02,
            max_length_m=0.6,
        )

    @property
    def series(self) -> ForceTorqueSeries | None:
        return self._series

    @property
    def missing_channels(self) -> tuple[str, ...]:
        return self._missing_channels

    @property
    def artists(self) -> list[Artist]:
        return list(self._artists)

    @property
    def status_note(self) -> str:
        return self._status_note

    def active_kinds(self) -> frozenset[WrenchKind]:
        """Compute the active set of WrenchKinds permitted by current toggles."""
        kinds: set[WrenchKind] = set()
        if self.check_forces.isChecked():
            kinds.update(
                {
                    WrenchKind.JOINT_REACTION,
                    WrenchKind.EXTERNAL,
                    WrenchKind.CONTACT,
                    WrenchKind.GRAVITY,
                    WrenchKind.MUSCLE,
                }
            )
        if self.check_torques.isChecked():
            kinds.add(WrenchKind.JOINT_ACTUATOR)
            kinds.update({WrenchKind.JOINT_REACTION, WrenchKind.EXTERNAL})
        if self.check_grip.isChecked():
            kinds.add(WrenchKind.GRIP)
        return frozenset(kinds)

    def load_force_series(
        self, path: str | Path, rotation_tol: float = 1e-2
    ) -> tuple[ForceTorqueSeries, tuple[str, ...]]:
        """Load logged Simscape force series from a trial CSV."""
        p = Path(path)
        if not p.is_file():
            self._series = ForceTorqueSeries(engine="simscape", frames=())
            self._missing_channels = ()
            self._status_note = f"Force dataset not found: {p.name}"
            self._notify_host(self._status_note)
            return self._series, self._missing_channels

        try:
            series, missing = load_simscape_force_series(p, rotation_tol=rotation_tol)
            self._series = series
            self._missing_channels = missing
            if missing:
                self._status_note = (
                    f"Loaded {len(series)} frames ({len(missing)} channels missing)"
                )
            else:
                self._status_note = f"Loaded {len(series)} force frames"
            self._notify_host(self._status_note)
            return series, missing
        except (ValueError, TypeError, OSError) as err:
            logger.warning("Failed to load force series from %s: %s", p, err)
            self._series = ForceTorqueSeries(engine="simscape", frames=())
            self._missing_channels = ()
            self._status_note = f"Force series unavailable: {err}"
            self._notify_host(self._status_note)
            return self._series, self._missing_channels

    def set_force_series(
        self, series: ForceTorqueSeries | None, missing: tuple[str, ...] = ()
    ) -> None:
        """Assign an existing series directly."""
        self._series = series
        self._missing_channels = tuple(missing)
        if series is not None and len(series) > 0:
            self._status_note = f"Attached {len(series)} force frames"
        else:
            self._status_note = "No force frames"

    def clear(self) -> None:
        """Clear all series data and artists."""
        self.clear_artists()
        self._series = None
        self._missing_channels = ()
        self._status_note = ""
        self.label_legend.setText("No force data")

    def clear_artists(self) -> None:
        """Remove active Matplotlib glyph artists."""
        for a in self._artists:
            try:
                a.remove()
            except Exception:  # noqa: BLE001
                pass
        self._artists.clear()
        if self._legend_inset is not None:
            try:
                self._legend_inset.remove()
            except Exception:  # noqa: BLE001
                pass
            self._legend_inset = None

    def update_frame(
        self, time_s: float, dt: float, ax: Axes | None
    ) -> tuple[int, AxialLoadFrame | None]:
        """Build and render glyphs for the specified timestamp.

        Returns (number of artists rendered, axial loads frame or None).
        """
        if self._series is None or len(self._series) == 0:
            self.clear_artists()
            self.label_legend.setText("No force data")
            return 0, None

        max_gap_s = max(0.001, 2.0 * dt)
        frame: ForceTorqueFrame | None = self._series.frame_at(
            time_s, max_gap_s=max_gap_s
        )
        if frame is None:
            self.clear_artists()
            self.label_legend.setText("Force gap frame")
            return 0, None

        axial_loads = frame.axial_loads
        kinds = self.active_kinds()

        # If all relevant toggles off, clear artists but still return axial loads
        if not kinds or (
            not self.check_forces.isChecked() and not self.check_torques.isChecked()
        ):
            self.clear_artists()
            self.label_legend.setText("Forces hidden")
            return 0, axial_loads

        if ax is None:
            return 0, axial_loads

        # Build glyphs with kind filtering
        style = ForceGlyphStyle(
            force_scale_m_per_n=self._style.force_scale_m_per_n,
            torque_scale_m_per_nm=self._style.torque_scale_m_per_nm,
            min_length_m=self._style.min_length_m,
            max_length_m=self._style.max_length_m,
            kinds=kinds,
        )
        glyphs: GlyphSet = build_glyphs(frame, style)

        # Apply specific arrow/arc visibility switches
        arrows = glyphs.arrows if self.check_forces.isChecked() else ()
        arcs = glyphs.torque_arcs if self.check_torques.isChecked() else ()

        glyphs = GlyphSet(
            time_s=glyphs.time_s,
            arrows=arrows,
            torque_arcs=arcs,
            legend=glyphs.legend,
        )

        self.clear_artists()
        self._artists = draw_glyphs_3d(ax, glyphs)
        if glyphs.legend is not None and (arrows or arcs):
            try:
                self._legend_inset = draw_legend(ax, glyphs.legend)
            except Exception as e:  # noqa: BLE001
                logger.debug("Inset legend skipped: %s", e)
        legend_text = format_legend_text(glyphs.legend)
        self.label_legend.setText(legend_text)

        return len(self._artists), axial_loads

    def _notify_host(self, message: str) -> None:
        if self._parent is None:
            return
        host = self._parent.window()
        if host is not None and hasattr(host, "statusBar"):
            sb = host.statusBar()
            if sb is not None:
                sb.showMessage(message, 5000)
