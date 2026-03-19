"""
Custom QWidget that draws the double/triple pendulum animation.

Renders the segments, joint markers, a tip trail, force vectors,
ellipsoids, and an interactive zoom/pan canvas.

Inherits shared zoom/pan, grid, trail, and coordinate mapping from
BasePendulumWidget.  Specializes: pendulum segment drawing, force
vectors (regular + zero-torque), ellipsoid overlays, COM, zoom controls.

Design by Contract
------------------
- Pre:  ``set_simulation(result)`` requires ``result is not None``.
- Pre:  ``set_frame(idx)`` requires loaded simulation.
- Inv:  ``_force_scale > 0`` (enforced by base class).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..jacobians import ellipsoids_double, ellipsoids_triple
from ..joint_moments import double_pendulum_moments, triple_pendulum_moments
from ..simulation import SimulationResult
from .base_pendulum_widget import BasePendulumWidget

logger = logging.getLogger(__name__)


class PendulumWidget(BasePendulumWidget):
    """Animated zoomable/pannable visualization of the double (or triple) pendulum.

    Controls
    --------
    Scroll wheel   : zoom in / out centred on cursor
    Left drag      : pan the view
    Double-click   : reset zoom & pan
    """

    # Model-specific colors (beyond what base provides)
    COLOR_ARM = QColor(70, 140, 240)
    COLOR_CLUB = QColor(240, 130, 50)
    COLOR_SHOULDER = QColor(210, 210, 220)
    COLOR_WRIST = QColor(255, 225, 80)
    COLOR_WRIST2 = QColor(120, 225, 185)
    COLOR_TIP = QColor(255, 80, 80)
    COLOR_OVERLAY_BG = QColor(25, 25, 42, 200)

    # Ellipsoid colors
    COLOR_MOB_ELLIPSOID = QColor(100, 200, 255, 70)
    COLOR_MOB_OUTLINE = QColor(100, 200, 255, 180)
    COLOR_FORCE_ELLIPSOID = QColor(255, 160, 80, 60)
    COLOR_FORCE_OUTLINE = QColor(255, 160, 80, 180)

    # Zero-torque force vector color (violet/purple)
    COLOR_ZERO_TORQUE = QColor(210, 120, 255)

    # Torque vector color (cyan)
    COLOR_TORQUE = QColor(0, 220, 220)
    # Moment of force color (orange-gold)
    COLOR_MOMENT = QColor(255, 180, 60)
    # Sum of moments color (hot pink)
    COLOR_SUM_MOMENTS = QColor(255, 80, 180)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._result: SimulationResult | None = None
        self._current_idx: int = 0

        # Pre-computed counterfactual forces (list[dict] or None)
        self._zero_torque_forces: list[dict] | None = None

        # Perf: precomputed tip position cache (n_steps x 2)
        self._tip_positions_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def _get_total_length(self) -> float:
        if self._result is not None:
            p = self._result.params
            total = p.L1 + p.L2
            if hasattr(p, "L3"):
                total += p.L3
            return total
        return 2.0

    def _has_result(self) -> bool:
        return self._result is not None

    def _draw_model(self, painter: QPainter) -> None:
        self._draw_pendulum(painter)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_simulation(self, result: SimulationResult) -> None:
        """Load a new simulation result, precompute caches, and reset display.

        Pre: result is not None
        """
        assert result is not None, "SimulationResult must not be None"
        self._result = result
        self._current_idx = 0
        self._trail.clear()

        self._tip_positions_cache = self._precompute_tips(result)
        self._zero_torque_forces = self._precompute_zero_torque_forces(result)

        self.update()

    @staticmethod
    def _precompute_tips(result: SimulationResult) -> np.ndarray:
        """Vectorised forward kinematics for the tip joint across all frames.

        Returns shape (n_steps, 2) float64 array of (x, y) world coords.
        """
        states = result.states
        params = result.params
        L1 = params.L1
        L2 = params.L2

        theta1 = states[:, 0]
        phi_or_1 = states[:, 1]

        if states.shape[1] == 6:
            L3 = getattr(params, "L3", L2)
            phi2 = states[:, 2]
            wx1 = L1 * np.sin(theta1)
            wy1 = -L1 * np.cos(theta1)
            abs_phi1 = theta1 + phi_or_1
            wx2 = wx1 + L2 * np.sin(abs_phi1)
            wy2 = wy1 - L2 * np.cos(abs_phi1)
            abs_phi2 = theta1 + phi_or_1 + phi2
            tx = wx2 + L3 * np.sin(abs_phi2)
            ty = wy2 - L3 * np.cos(abs_phi2)
        else:
            abs_angle2 = theta1 + phi_or_1
            wx = L1 * np.sin(theta1)
            wy = -L1 * np.cos(theta1)
            tx = wx + L2 * np.sin(abs_angle2)
            ty = wy - L2 * np.cos(abs_angle2)

        return np.column_stack([tx, ty])

    @staticmethod
    def _precompute_zero_torque_forces(result: SimulationResult) -> list[dict]:
        """Pre-compute zero-torque counterfactual joint forces for every frame."""
        from ..counterfactual import (
            zero_torque_joint_forces_double,
            zero_torque_joint_forces_triple,
        )

        forces: list[dict] = []
        params = result.params
        for state in result.states:
            try:
                if state.shape[0] >= 6:
                    forces.append(zero_torque_joint_forces_triple(state, params))  # type: ignore[arg-type]
                else:
                    forces.append(zero_torque_joint_forces_double(state, params))
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                logger.warning("zero_torque_joint_forces failed for state: %s", exc)
                forces.append({})
        return forces

    def set_frame(self, idx: int) -> None:
        """Set the displayed frame and rebuild the trail up to that frame.

        Instead of appending one point per call (which breaks when scrubbing
        back and forth), rebuild the trail from the precomputed tip cache
        so it always shows frames [max(0, idx-TRAIL_LENGTH)..idx].
        """
        assert idx is not None, "idx must be provided"
        if self._result is None:
            return
        idx = max(0, min(idx, self._result.n_steps - 1))
        self._current_idx = idx

        # Rebuild trail from precomputed cache: show last TRAIL_LENGTH frames
        self._trail.clear()
        if self._tip_positions_cache is not None:
            start = max(0, idx - self.TRAIL_LENGTH + 1)
            for i in range(start, idx + 1):
                self._trail.append(tuple(self._tip_positions_cache[i]))
        else:
            # Fallback: compute the current tip position only
            pos = self._result.positions_at(idx)
            self._trail.append(pos["tip"])

        self.update()

    def clear(self) -> None:
        """Reset to blank state."""
        self._result = None
        self._current_idx = 0
        self._trail.clear()
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: object) -> None:
        assert event is not None, "event must be provided"
        base_scale = self._compute_base_scale()
        self._pixels_per_meter = base_scale * self._zoom

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        self._draw_grid(painter)

        # Ground plane + tilt plane + ball visualization
        if self._result is not None:
            L_total = self._get_total_length()
            self._draw_ground_plane(painter, -L_total)
            self._draw_tilt_plane(painter)
            self._draw_ball(painter, 0.0, -L_total)

        if self._result is None:
            self._draw_placeholder(painter)
            self._draw_zoom_controls(painter)
            painter.end()
            return

        self._draw_trail(painter)
        self._draw_pendulum(painter)

        if self._show_forces:
            pos = self._result.positions_at(self._current_idx)
            self._draw_force_vectors(painter, pos)

        if self._show_zero_torque_forces:
            pos = self._result.positions_at(self._current_idx)
            self._draw_zero_torque_force_vectors(painter, pos)

        if self._show_torque_vectors:
            self._draw_torque_vectors(painter)

        if self._show_moment_of_force or self._show_sum_moments:
            self._draw_moment_of_force(painter)

        if self._show_mob_ellipsoids or self._show_force_ellipsoids:
            self._draw_ellipsoids_at_frame(painter)

        if self._show_com:
            self._draw_com(painter)

        self._draw_info(painter)
        self._draw_zoom_controls(painter)

        if not self._gravity_on:
            self._draw_no_gravity_badge(painter)

        painter.end()

    # ------------------------------------------------------------------
    # Pendulum segment drawing
    # ------------------------------------------------------------------

    def _draw_pendulum(self, painter: QPainter) -> None:
        """Draw the segments and joint markers.

        When 3D mode is enabled (#1155), uses tapered gradient segments.
        Otherwise falls back to flat-line rendering.
        """
        assert self._result is not None
        pos = self._result.positions_at(self._current_idx)
        shoulder = self._world_to_pixel(*pos["shoulder"])
        tip = self._world_to_pixel(*pos["tip"])

        wrist2 = None
        if "wrist2" in pos:
            wrist1 = self._world_to_pixel(*pos["wrist1"])
            wrist2 = self._world_to_pixel(*pos["wrist2"])
        else:
            wrist1 = self._world_to_pixel(*pos["wrist"])

        if self._3d_mode:
            # 3D tapered segment rendering (#1155)
            self._draw_3d_segment(
                painter,
                shoulder,
                wrist1,
                14,
                10,
                self.COLOR_ARM,
            )
            if wrist2 is not None:
                self._draw_3d_segment(
                    painter,
                    wrist1,
                    wrist2,
                    10,
                    7,
                    self.COLOR_CLUB,
                )
                self._draw_3d_segment(
                    painter,
                    wrist2,
                    tip,
                    7,
                    5,
                    self.COLOR_WRIST2,
                )
            else:
                self._draw_3d_segment(
                    painter,
                    wrist1,
                    tip,
                    10,
                    6,
                    self.COLOR_CLUB,
                )
        else:
            # Flat-line rendering (default)
            pen = QPen(self.COLOR_ARM, 5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(shoulder, wrist1)

            pen = QPen(self.COLOR_CLUB, 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(wrist1, wrist2 if wrist2 is not None else tip)

            if wrist2 is not None:
                pen2 = QPen(self.COLOR_WRIST2, 3)
                pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen2)
                painter.drawLine(wrist2, tip)

        # Joints (always drawn)
        self._draw_joint(painter, shoulder, 8, self.COLOR_SHOULDER)
        self._draw_joint(painter, wrist1, 6, self.COLOR_WRIST)
        if wrist2 is not None:
            self._draw_joint(painter, wrist2, 5, self.COLOR_WRIST2)
        self._draw_joint(painter, tip, 5, self.COLOR_TIP)

    # ------------------------------------------------------------------
    # Info and placeholder
    # ------------------------------------------------------------------

    def _draw_info(self, painter: QPainter) -> None:
        assert self._result is not None
        t = self._result.t[self._current_idx]
        s = self._result.states[self._current_idx]
        theta1_deg = np.degrees(s[0])
        phi_deg = np.degrees(s[1])

        painter.setFont(QFont("Monospace", 9))
        painter.setPen(self.COLOR_TEXT)

        lines = [f"t = {t:.3f} s", f"\u03b81 = {theta1_deg:+.1f}\u00b0"]
        if s.shape[0] >= 6:
            phi2_deg = np.degrees(s[2])
            lines.append(f"\u03c61 = {phi_deg:+.1f}\u00b0")
            lines.append(f"\u03c62 = {phi2_deg:+.1f}\u00b0")
        else:
            lines.append(f"\u03c6 = {phi_deg:+.1f}\u00b0")

        lines.append(f"zoom {self._zoom:.1f}\u00d7")

        y = 18
        for line in lines:
            painter.drawText(8, y, line)
            y += 15

    def _draw_placeholder(self, painter: QPainter) -> None:
        assert painter is not None, "painter must be provided"
        painter.setPen(QColor(80, 80, 110))
        painter.setFont(QFont("Sans", 12))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Configure parameters\nand click 'Run Simulation'\n\n"
            "Scroll to zoom \u00b7 Drag to pan \u00b7 Double-click to reset",
        )

    # ------------------------------------------------------------------
    # Force vectors
    # ------------------------------------------------------------------

    def _draw_force_vectors(self, painter: QPainter, pos: dict) -> None:
        """Draw net force vectors at joints."""
        assert painter is not None, "painter must be provided"
        if self._result is None or not hasattr(self._result, "joint_forces_at"):
            return
        forces = self._result.joint_forces_at(self._current_idx)
        if not forces:
            return

        magnitudes = [np.hypot(f[0], f[1]) for f in forces.values()]
        max_mag = max(1.0, max(magnitudes))
        scale = 0.4 * self._pixels_per_meter * self._force_scale / max_mag

        joint_map = {
            "shoulder": pos.get("shoulder"),
            "wrist": pos.get("wrist"),
            "wrist1": pos.get("wrist1"),
            "wrist2": pos.get("wrist2"),
        }

        painter.setPen(QPen(self.COLOR_FORCE, 2))
        for key, force in forces.items():
            if self._visible_segments is not None and key not in self._visible_segments:
                continue
            joint_pos = joint_map.get(key)
            if joint_pos is None:
                continue
            fx, fy = force
            end = (
                joint_pos[0] + fx * scale / self._pixels_per_meter,
                joint_pos[1] + fy * scale / self._pixels_per_meter,
            )
            self._draw_arrow(painter, joint_pos, end)

    def _draw_zero_torque_force_vectors(self, painter: QPainter, pos: dict) -> None:
        """Draw zero-torque (passive drift) force vectors at each joint."""
        assert painter is not None, "painter must be provided"
        if self._zero_torque_forces is None or not self._zero_torque_forces:
            return
        forces = self._zero_torque_forces[self._current_idx]
        if not forces:
            return

        magnitudes = [np.hypot(f[0], f[1]) for f in forces.values()]
        max_mag = max(1.0, max(magnitudes))
        scale = 0.4 * self._pixels_per_meter * self._force_scale / max_mag

        joint_map = {
            "shoulder": pos.get("shoulder"),
            "wrist": pos.get("wrist"),
            "wrist1": pos.get("wrist1"),
            "wrist2": pos.get("wrist2"),
        }

        pen = QPen(self.COLOR_ZERO_TORQUE, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for key, force in forces.items():
            if self._visible_segments is not None and key not in self._visible_segments:
                continue
            joint_pos = joint_map.get(key)
            if joint_pos is None:
                continue
            fx, fy = force
            end = (
                joint_pos[0] + fx * scale / self._pixels_per_meter,
                joint_pos[1] + fy * scale / self._pixels_per_meter,
            )
            self._draw_arrow(painter, joint_pos, end)

    def _draw_arrow(self, painter: QPainter, origin: tuple, end: tuple) -> None:
        """Draw a force/torque vector with a filled triangular arrowhead."""
        assert painter is not None, "painter must be provided"
        p0 = self._world_to_pixel(origin[0], origin[1])
        p1 = self._world_to_pixel(end[0], end[1])
        painter.drawLine(p0, p1)

        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        length = max(1.0, np.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        arrow_len = 10.0
        arrow_w = 4.0

        left = QPointF(
            p1.x() - arrow_len * ux + arrow_w * uy,
            p1.y() - arrow_len * uy - arrow_w * ux,
        )
        right = QPointF(
            p1.x() - arrow_len * ux - arrow_w * uy,
            p1.y() - arrow_len * uy + arrow_w * ux,
        )

        path = QPainterPath()
        path.moveTo(p1)
        path.lineTo(left)
        path.lineTo(right)
        path.closeSubpath()

        old_brush = painter.brush()
        painter.setBrush(QBrush(painter.pen().color()))
        painter.drawPath(path)
        painter.setBrush(old_brush)

    # ------------------------------------------------------------------
    # Torque vector drawing (#1119, #1170)
    # ------------------------------------------------------------------

    def _draw_torque_vectors(self, painter: QPainter) -> None:
        """Draw applied torque as curved arcs at each joint.

        Convention: clockwise = negative, counterclockwise = positive.
        Arc radius scales with torque magnitude.
        """
        assert painter is not None, "painter must be provided"
        if self._result is None:
            return
        try:
            torques = self._result.torques_at(self._current_idx)
        except (AttributeError, IndexError):
            return

        pos = self._result.positions_at(self._current_idx)
        joint_names = ["shoulder"]
        if "wrist1" in pos:
            joint_names.extend(["wrist1", "wrist2"])
        else:
            joint_names.append("wrist")

        torque_list = list(torques) if not isinstance(torques, list) else torques
        max_tau = max(1e-6, max(abs(t) for t in torque_list))

        for i, jname in enumerate(joint_names):
            if i >= len(torque_list):
                break
            if (
                self._visible_segments is not None
                and jname not in self._visible_segments
            ):
                continue
            jp = pos.get(jname)
            if jp is None:
                continue

            tau = torque_list[i]
            if abs(tau) < 1e-10:
                continue

            center = self._world_to_pixel(*jp)
            radius = int(15 + 25 * abs(tau) / max_tau)

            # Arc parameters: torque sign determines direction
            start_angle = 30 * 16  # 30° in 1/16 degree units for Qt
            span = int(np.sign(tau) * 240 * 16 * abs(tau) / max_tau)

            pen = QPen(self.COLOR_TORQUE, 2.5)
            painter.setPen(pen)
            rect = QRect(
                int(center.x()) - radius,
                int(center.y()) - radius,
                2 * radius,
                2 * radius,
            )
            painter.drawArc(rect, start_angle, span)

            # Draw arrowhead at arc end
            end_angle_rad = np.radians((start_angle + span) / 16)
            arrow_x = center.x() + radius * np.cos(end_angle_rad)
            arrow_y = center.y() - radius * np.sin(end_angle_rad)

            # Small direction indicator
            painter.setBrush(QBrush(self.COLOR_TORQUE))
            painter.drawEllipse(QPointF(arrow_x, arrow_y), 3, 3)

            # Label
            painter.setFont(QFont("Monospace", 7))
            painter.drawText(
                QPointF(center.x() + radius + 3, center.y() - 2),
                f"τ={tau:.1f}",
            )

    def _draw_moment_of_force(self, painter: QPainter) -> None:
        """Draw moment-of-force and/or sum of moments at each joint.

        Uses the joint_moments module for proper proximal-on-distal computation.
        """
        assert painter is not None, "painter must be provided"
        if self._result is None:
            return
        try:
            torques = self._result.torques_at(self._current_idx)
            forces = self._result.joint_forces_at(self._current_idx)
        except (AttributeError, IndexError):
            return

        pos = self._result.positions_at(self._current_idx)
        state = self._result.states[self._current_idx]
        params = self._result.params

        try:
            if state.shape[0] >= 6:
                moments = triple_pendulum_moments(pos, forces, tuple(torques), params)  # type: ignore[arg-type]
            else:
                moments = double_pendulum_moments(pos, forces, tuple(torques), params)  # type: ignore[arg-type]
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            logger.debug("Moment computation failed: %s", exc)
            return

        # Draw total moments at each joint
        joint_names = ["shoulder"]
        if "wrist1" in pos:
            joint_names.extend(["wrist1", "wrist2"])
        else:
            joint_names.append("wrist")

        for jname in joint_names:
            if (
                self._visible_segments is not None
                and jname not in self._visible_segments
            ):
                continue
            jp = pos.get(jname)
            if jp is None:
                continue

            center = self._world_to_pixel(*jp)

            if self._show_sum_moments:
                key = f"{jname}_total_moment"
                total = moments.get(key, 0.0)
                if abs(total) > 1e-10:
                    painter.setPen(QPen(self.COLOR_SUM_MOMENTS, 2))
                    painter.setFont(QFont("Monospace", 7))
                    painter.drawText(
                        QPointF(center.x() - 30, center.y() + 18),
                        f"ΣM={total:.2f}",
                    )

            if self._show_moment_of_force:
                key_applied = f"{jname}_applied_torque"
                key_gravity = f"{jname}_gravity_moment"
                tau_applied = moments.get(key_applied, 0.0)
                tau_grav = moments.get(key_gravity, 0.0)

                y_offset = 0
                for label, val, color in [
                    ("τ_a", tau_applied, self.COLOR_TORQUE),
                    ("M_g", tau_grav, self.COLOR_MOMENT),
                ]:
                    if abs(val) > 1e-10:
                        painter.setPen(QPen(color, 1.5))
                        painter.setFont(QFont("Monospace", 6))
                        painter.drawText(
                            QPointF(center.x() + 12, center.y() + y_offset - 8),
                            f"{label}={val:.2f}",
                        )
                        y_offset += 10

    # ------------------------------------------------------------------
    # Ellipsoid drawing
    # ------------------------------------------------------------------

    def _draw_ellipsoids_at_frame(self, painter: QPainter) -> None:
        """Compute and draw mobility/force ellipsoids for the current frame."""
        assert self._result is not None
        state = self._result.states[self._current_idx]
        params = self._result.params
        ppm = self._pixels_per_meter

        if state.shape[0] >= 6:
            theta1, phi1, phi2 = float(state[0]), float(state[1]), float(state[2])
            data = ellipsoids_triple(
                theta1,
                phi1,
                phi2,
                params.L1,
                params.L2,
                params.L3,  # type: ignore[attr-defined]
            )
            pos = self._result.positions_at(self._current_idx)
            endpoint_map = {
                "wrist1": pos.get("wrist1", pos.get("wrist", (0.0, 0.0))),
                "wrist2": pos.get("wrist2", (0.0, 0.0)),
                "tip": pos["tip"],
            }
        else:
            theta1, phi = float(state[0]), float(state[1])
            data = ellipsoids_double(theta1, phi, params.L1, params.L2)
            pos = self._result.positions_at(self._current_idx)
            endpoint_map = {
                "wrist": pos.get("wrist", (0.0, 0.0)),
                "tip": pos["tip"],
            }

        for name, ell in data.items():
            if (
                self._visible_segments is not None
                and name not in self._visible_segments
            ):
                continue
            world_pos = endpoint_map.get(name)
            if world_pos is None:
                continue
            cx_px = self._world_to_pixel(*world_pos).x()
            cy_px = self._world_to_pixel(*world_pos).y()

            dirs = ell["directions"]
            if self._show_mob_ellipsoids:
                mob = ell["mob_semi_axes"]
                mob_scale = self._mob_ellipsoid_scale * ppm * 0.3
                self._draw_ellipse_axes(
                    painter,
                    cx_px,
                    cy_px,
                    dirs,
                    mob * mob_scale,
                    fill=self.COLOR_MOB_ELLIPSOID,
                    outline=self.COLOR_MOB_OUTLINE,
                    label="M",
                )
            if self._show_force_ellipsoids:
                force = ell["force_semi_axes"]
                force_scale = self._force_ellipsoid_scale * ppm * 0.3
                if force is not None:
                    self._draw_ellipse_axes(
                        painter,
                        cx_px,
                        cy_px,
                        dirs,
                        force * force_scale,
                        fill=self.COLOR_FORCE_ELLIPSOID,
                        outline=self.COLOR_FORCE_OUTLINE,
                        label="F",
                    )
                else:
                    # Degenerate (singular) — draw a line along the major
                    # mobility axis to show the direction of force capability
                    # (#1133: force ellipsoid at wrist always visible)
                    mob = ell["mob_semi_axes"]
                    line_len = float(mob[0]) * force_scale * 0.5
                    line_len = max(10.0, min(line_len, 200.0))
                    dx_line = float(dirs[0, 0]) * line_len
                    dy_line = -float(dirs[1, 0]) * line_len  # screen Y is inverted
                    pen = QPen(self.COLOR_FORCE_OUTLINE, 1.5, Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(
                        QPointF(cx_px - dx_line, cy_px - dy_line),
                        QPointF(cx_px + dx_line, cy_px + dy_line),
                    )
                    painter.setFont(QFont("Monospace", 7))
                    painter.drawText(
                        QPointF(cx_px + dx_line + 4, cy_px + dy_line), "F∞"
                    )

    def _draw_ellipse_axes(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        directions: np.ndarray,
        semi_axes_px: np.ndarray,
        fill: QColor,
        outline: QColor,
        label: str = "",
    ) -> None:
        """Draw a 2-D ellipse defined by principal axes and semi-axis lengths.

        Contract
        --------
        Pre: directions.shape == (2, 2)
        Pre: semi_axes_px.shape == (2,)
        """
        assert directions.shape == (2, 2), "directions must be (2, 2)"
        assert semi_axes_px.shape == (2,), "semi_axes_px must be (2,)"

        a = float(semi_axes_px[0])
        b = float(semi_axes_px[1])

        max_px = min(self.width(), self.height()) * 0.8
        a = max(2.0, min(a, max_px))
        b = max(2.0, min(b, max_px))

        dx, dy = float(directions[0, 0]), float(directions[1, 0])
        angle_deg = float(np.degrees(np.arctan2(-dy, dx)))

        painter.save()
        painter.translate(cx, cy)
        painter.rotate(angle_deg)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(outline, 1.2))
        painter.drawEllipse(QPointF(0.0, 0.0), a, b)
        painter.restore()

        if label:
            lbl_x = cx + float(directions[0, 0]) * a + 4
            lbl_y = cy - float(directions[1, 0]) * a
            painter.setFont(QFont("Monospace", 7))
            painter.setPen(outline)
            painter.drawText(QPointF(lbl_x, lbl_y), label)

    # ------------------------------------------------------------------
    # Center of Mass drawing
    # ------------------------------------------------------------------

    def _draw_com(self, painter: QPainter) -> None:
        """Draw the combined center of mass of the system."""
        assert painter is not None, "painter must be provided"
        if self._result is None:
            return

        state = self._result.states[self._current_idx]
        params = self._result.params

        if state.shape[0] >= 6:
            pos = self._result.positions_at(self._current_idx)
            masses = [params.m1, params.m2, getattr(params, "m3", params.m2)]
            wrist1 = np.array(pos.get("wrist1", pos.get("wrist", (0, 0))))
            wrist2 = np.array(pos.get("wrist2", (0, 0)))
            tip = np.array(pos["tip"])
            shoulder = np.array(pos["shoulder"])
            com1 = 0.5 * (shoulder + wrist1)
            com2 = 0.5 * (wrist1 + wrist2)
            com3 = 0.5 * (wrist2 + tip)
            total_m = sum(masses)
            com = (masses[0] * com1 + masses[1] * com2 + masses[2] * com3) / total_m
        else:
            theta1 = state[0]
            phi = state[1]
            abs2 = theta1 + phi
            c1x = 0.5 * params.L1 * np.sin(theta1)
            c1y = -0.5 * params.L1 * np.cos(theta1)
            wx = params.L1 * np.sin(theta1)
            wy = -params.L1 * np.cos(theta1)
            c2x = wx + 0.5 * params.L2 * np.sin(abs2)
            c2y = wy - 0.5 * params.L2 * np.cos(abs2)
            total_m = params.m1 + params.m2
            com = np.array(
                [
                    (params.m1 * c1x + params.m2 * c2x) / total_m,
                    (params.m1 * c1y + params.m2 * c2y) / total_m,
                ]
            )

        com_px = self._world_to_pixel(float(com[0]), float(com[1]))

        r = 6
        painter.setPen(QPen(self.COLOR_COM, 2))
        painter.setBrush(QBrush(QColor(255, 255, 80, 100)))
        painter.drawEllipse(com_px, r, r)
        painter.drawLine(
            QPointF(com_px.x() - r * 1.5, com_px.y()),
            QPointF(com_px.x() + r * 1.5, com_px.y()),
        )
        painter.drawLine(
            QPointF(com_px.x(), com_px.y() - r * 1.5),
            QPointF(com_px.x(), com_px.y() + r * 1.5),
        )
        painter.setFont(QFont("Monospace", 7))
        painter.drawText(QPointF(com_px.x() + r + 2, com_px.y() - 2), "COM")

    # ------------------------------------------------------------------
    # Zoom controls overlay
    # ------------------------------------------------------------------

    def _draw_zoom_controls(self, painter: QPainter) -> None:
        """Draw a small zoom toolbar in the top-right corner."""
        assert painter is not None, "painter must be provided"
        r = self.rect()
        btn_size = 24
        margin = 6
        x = r.right() - btn_size - margin
        y_start = margin

        buttons = [
            ("\u2295", "Zoom in"),
            ("\u2296", "Zoom out"),
            ("\u2922", "Reset view"),
        ]
        painter.setFont(QFont("Sans", 11))

        for i, (icon, _) in enumerate(buttons):
            bx = x
            by = y_start + i * (btn_size + 3)
            rect = QRect(bx, by, btn_size, btn_size)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.COLOR_OVERLAY_BG))
            painter.drawRoundedRect(rect, 4, 4)

            painter.setPen(QColor(180, 180, 210))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, icon)

        self._zoom_btn_rects = [
            QRect(x, y_start + i * (btn_size + 3), btn_size, btn_size)
            for i in range(len(buttons))
        ]

    def _handle_zoom_button_click(self, pos: QPoint) -> bool:
        assert pos is not None, "pos must be provided"
        if not hasattr(self, "_zoom_btn_rects"):
            return False
        for i, rect in enumerate(self._zoom_btn_rects):
            if rect.contains(pos):
                if i == 0:
                    self._zoom = min(20.0, self._zoom * 1.3)
                elif i == 1:
                    self._zoom = max(0.1, self._zoom / 1.3)
                else:
                    self._zoom = 1.0
                    self._pan_x = 0.0
                    self._pan_y = 0.0
                self.update()
                return True
        return False
