"""Rendering / drawing mixin for :class:`MainWidget`.

Owns the matplotlib axes setup and every ``_draw_*`` helper. Split out
of ``gui.py`` (Subtask 5 / #4998 of EPIC #4993) to keep the per-file
line budget under 1200 lines.

The mixin assumes the host class exposes the attributes constructed in
:meth:`MainWidget.__init__`: ``self.ax``, ``self.canvas``,
``self.poses``, ``self.transform``, ``self.df``, ``self.events``,
``self.current_frame``, the ``self.show_*`` toggles, and the various
``cb_*`` checkboxes that the build-mixin wires up.
"""

from __future__ import annotations

from contextlib import suppress

import math
import numpy as np

from src.shared.python.motion_matching.diagnostics._skeleton_render import (
    equalize_3d_axes as _shared_equalize_3d_axes,
)
from src.tools.starting_pose_matcher.core import (
    DEFAULT_PHASE as _DEFAULT_PHASE,
    PHASE_BOUNDS as _PHASE_BOUNDS,
    PoseSlot,
    Skeleton,
    phase_key_from_label as _phase_key_from_label,
)

from ._gui_common import _CAMERA_PRESETS, _DEFAULT_CAMERA


class _RenderMixin:
    """Drawing + helper logic for :class:`MainWidget`.

    Pure presentation: it reads the host's data attributes and writes
    only to the matplotlib axes / status labels. No widget construction
    happens here.
    """

    # ---------- axis setup ----------------------------------------------- #

    def _setup_axes(self) -> None:
        ax = self.ax
        ax.set_xlabel("X (target line)", color="#cbd5e1")
        ax.set_ylabel("Y (ball direction)", color="#cbd5e1")
        ax.set_zlabel("Z (vertical)", color="#cbd5e1")

        # Default static bounds; if we have data, we'll re-fit via the shared
        # ``equalize_3d_axes`` helper (see #4376) so the view always tracks
        # the loaded mocap + skeleton extents.
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 2.0)
        ax.set_zlim(-1.5, 2.5)

        with suppress(AttributeError):
            ax.set_box_aspect((4, 3.5, 4))
        # Dark-theme tick & pane colours
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.16, 0.18, 0.22, 0.85))
            axis.label.set_color("#cbd5e1")
            for t in axis.get_ticklabels():
                t.set_color("#a3a8b3")
            axis._axinfo["grid"]["color"] = (0.35, 0.40, 0.48, 0.45)

    def _autoscale_axes_to_data(self) -> None:
        """Use the shared ``equalize_3d_axes`` helper to fit the view to
        whatever joints / mocap targets are currently visible.  Called from
        ``_redraw`` after the static defaults are set, so the user gets a
        cube-aspect view tightly cropped around the body.
        """
        pts: list[np.ndarray] = []
        # Visible pose skeletons (incl. trajectory frame if active)
        for slot in self.poses.values():
            if not slot.visible:
                continue
            skel = self._effective_skeleton(slot)
            for v in skel.joints.values():
                pts.append(self.transform.apply(v[None, :])[0])
            mp = self._mocap_pos_for(slot, "mid")
            ch = self._mocap_pos_for(slot, "club")
            if mp is not None:
                pts.append(mp)
            if ch is not None:
                pts.append(ch)
        if not pts:
            return
        with suppress(ValueError, AttributeError):
            _shared_equalize_3d_axes(self.ax, np.asarray(pts))

    def _apply_camera_preset(self, name: str) -> None:
        elev, azim = _CAMERA_PRESETS.get(name, _CAMERA_PRESETS[_DEFAULT_CAMERA])
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw()

    # ---------- helpers --------------------------------------------------- #

    def _frame_for(self, label: str) -> int | None:
        f = self.events.frame_for(label)
        if f is None or self.df is None:
            return None
        return max(0, min(f, len(self.df) - 1))

    def _first_visible_pose(self) -> PoseSlot | None:
        for slot in self.poses.values():
            if slot.visible:
                return slot
        return None

    def _mocap_pos_for(self, slot: PoseSlot, kind: str) -> np.ndarray | None:
        if self.df is None:
            return None
        if self.frame_override_active:
            f: int | None = self.current_frame
        else:
            f = self._frame_for(slot.target_event)
        if f is None:
            return None
        f = max(0, min(int(f), len(self.df) - 1))
        row = self.df.iloc[f]
        if kind == "mid":
            return np.array([-row["mid_X"], row["mid_Y"], row["mid_Z"]])
        return np.array([-row["club_X"], row["club_Y"], row["club_Z"]])

    def _compute_residuals_mm(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, slot in self.poses.items():
            if "mp" not in slot.skeleton.joints:
                continue
            target = self._mocap_pos_for(slot, "mid")
            if target is None:
                continue
            moved = self.transform.apply(slot.skeleton.joints["mp"][None, :])[0]
            d_mid = (moved - target) * 1000.0
            entry = {
                "dx_mm": float(d_mid[0]),
                "dy_mm": float(d_mid[1]),
                "dz_mm": float(d_mid[2]),
                # ⚡ Bolt: math.hypot is faster than np.linalg.norm for small 1D arrays
                "norm_mm": float(math.hypot(d_mid[0], d_mid[1], d_mid[2])),
            }
            ch_target = self._mocap_pos_for(slot, "club")
            if ch_target is not None and "ch" in slot.skeleton.joints:
                moved_ch = self.transform.apply(slot.skeleton.joints["ch"][None, :])[0]
                diff_ch = (moved_ch - ch_target) * 1000.0
                entry["clubhead_norm_mm"] = float(
                    math.hypot(diff_ch[0], diff_ch[1], diff_ch[2])
                )
            out[key] = entry
        return out

    def _notify(self, msg: str) -> None:
        # Reuse the residual line as a status display
        self.lbl_residual.setText(msg + "\n" + self._residual_text())

    def _residual_text(self) -> str:
        residuals = self._compute_residuals_mm()
        if not residuals:
            return "Residuals: (no data)"
        lines = []
        for key, r in residuals.items():
            line = (
                f"{key}:  |Δmid|={r['norm_mm']:5.0f} mm  "
                f"(Δ=[{r['dx_mm']:+5.0f}, {r['dy_mm']:+5.0f}, "
                f"{r['dz_mm']:+5.0f}])"
            )
            if "clubhead_norm_mm" in r:
                line += f"   |Δclub|={r['clubhead_norm_mm']:5.0f} mm"
            lines.append(line)
        return "\n".join(lines)

    def _phase_window_key(self) -> str:
        return _phase_key_from_label(str(self.phase_window)) or _DEFAULT_PHASE

    # ===================================================================== #
    # Drawing                                                               #
    # ===================================================================== #

    def _redraw(self) -> None:
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self._setup_axes()
        self.ax.view_init(elev=elev, azim=azim)

        self._draw_floor_and_ball()
        self._draw_traces()
        self._draw_visible_poses()

        # Re-fit axes to the actually drawn data via the shared helper.
        # Keeps the view cropped tightly around the body and the mocap
        # target as the user scrubs through frames or loads a trajectory.
        if getattr(self, "auto_fit_axes", True):
            self._autoscale_axes_to_data()

        # Re-attach the live-view layer artists after the axes were
        # cleared by ``ax.clear()`` in this method. The controller keeps
        # its target data, so re-binding is just rebuilding artists.
        if (
            getattr(self, "_live_view", None) is not None
            and getattr(self, "_live_body_target", None) is not None
        ):
            self._live_view.set_target(body=self._live_body_target)
            self._live_view.set_frame(int(self.current_frame))

        self.lbl_residual.setText(self._residual_text())

        leg = self.ax.legend(loc="upper right", fontsize=8, ncol=1, framealpha=0.85)
        if leg is not None:
            for text in leg.get_texts():
                text.set_color("#e6e6e6")
            leg.get_frame().set_facecolor("#1f242b")
            leg.get_frame().set_edgecolor("#404652")
        self.canvas.draw()

    def _draw_floor_and_ball(self) -> None:
        if self.show_ground:
            x = np.linspace(-1.5, 1.5, 5)
            y = np.linspace(-1.5, 1.5, 5)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            self.ax.plot_surface(X, Y, Z, alpha=0.10, color="#22c55e")
        if self.show_ball:
            self.ax.scatter(
                [0], [0], [0.021], c="white", edgecolor="black", s=40, label="ball"
            )

    def _trace_window(self) -> tuple[int, int]:
        """Return [start, end) frame indices for trace drawing per phase setting."""
        if self.df is None:
            return (0, 0)
        n = len(self.df)
        bounds = _PHASE_BOUNDS.get(self._phase_window_key(), (None, None))
        # "None" -> draw across full data
        if bounds == (None, None):
            return (0, n)
        # "Manual range"
        if bounds == ("manual", "manual"):
            i0 = max(0, min(self.manual_window_start, n - 1))
            i1 = max(0, min(self.manual_window_end + 1, n))
            if i1 <= i0:
                i1 = i0 + 1
            return (i0, i1)
        # Event-bounded
        a_label, b_label = bounds
        a = self._frame_for(str(a_label)) if a_label is not None else 0
        b = self._frame_for(str(b_label)) if b_label is not None else (n - 1)
        if a is None:
            a = 0
        if b is None:
            b = n - 1
        if b < a:
            a, b = 0, n - 1
        return (a, b + 1)

    def _draw_traces(self) -> None:
        if self.df is None:
            return
        i0, i1 = self._trace_window()
        sub = self.df.iloc[i0:i1]
        if self.show_midhands_trace and len(sub) > 1:
            self.ax.plot(
                -sub["mid_X"].values,
                sub["mid_Y"].values,
                sub["mid_Z"].values,
                color="#7dd3fc",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label="mocap mid-hands trace",
            )
        if self.show_clubhead_trace and len(sub) > 1:
            self.ax.plot(
                -sub["club_X"].values,
                sub["club_Y"].values,
                sub["club_Z"].values,
                color="#fb7185",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label="mocap clubhead trace",
            )
        # Phase boundary markers (start / end of selected window)
        if (self.show_midhands_trace or self.show_clubhead_trace) and len(sub) > 0:
            for idx, _marker_label, color in [
                (i0, "start", "#22c55e"),
                (min(i1 - 1, len(self.df) - 1), "end", "#a855f7"),
            ]:
                row = self.df.iloc[idx]
                if self.show_midhands_trace:
                    self.ax.scatter(
                        -row["mid_X"],
                        row["mid_Y"],
                        row["mid_Z"],
                        color=color,
                        s=40,
                        marker="^",
                        edgecolor="black",
                        linewidth=0.5,
                    )
                if self.show_clubhead_trace:
                    self.ax.scatter(
                        -row["club_X"],
                        row["club_Y"],
                        row["club_Z"],
                        color=color,
                        s=40,
                        marker="^",
                        edgecolor="black",
                        linewidth=0.5,
                    )
        # Current-frame marker (cross)
        if (
            getattr(self, "cb_frame_marker", None) is not None
            and self.cb_frame_marker.isChecked()
            and self.df is not None
            and 0 <= self.current_frame < len(self.df)
        ):
            row = self.df.iloc[self.current_frame]
            if self.show_midhands_trace:
                self.ax.scatter(
                    -row["mid_X"],
                    row["mid_Y"],
                    row["mid_Z"],
                    color="#fde047",
                    s=120,
                    marker="x",
                    linewidth=2,
                    label="current frame (mid)",
                )
            if self.show_clubhead_trace:
                self.ax.scatter(
                    -row["club_X"],
                    row["club_Y"],
                    row["club_Z"],
                    color="#fde047",
                    s=140,
                    marker="x",
                    linewidth=2,
                    label="current frame (clubhead)",
                )

    def _draw_visible_poses(self) -> None:
        for slot in self.poses.values():
            if not slot.visible:
                continue
            self._draw_one_pose(slot)
        # Live "current-frame mocap club" — always drawn so playback is
        # visible without needing to toggle traces on or set the override.
        # When the override is active OR the slider differs from every
        # visible pose's event frame, draw it as a yellow accent line.
        self._draw_current_frame_club()

    def _draw_current_frame_club(self) -> None:
        """Draw a yellow "live" mocap club at the current frame.

        This is what makes playback visible — the skeleton is static per pose
        slot, so without this draw the only thing that would change as the
        playback timer fires is the spinbox value.  We always show it; when
        the slider matches a visible pose's event frame it lands on top of
        the bold red/orange mocap club anyway.
        """
        if self.df is None or len(self.df) == 0:
            return
        f = max(0, min(self.current_frame, len(self.df) - 1))
        row = self.df.iloc[f]
        mp = np.array([-row["mid_X"], row["mid_Y"], row["mid_Z"]])
        ch = np.array([-row["club_X"], row["club_Y"], row["club_Z"]])
        # Draw thin yellow club so it doesn't obscure the bold pose-targets.
        self.ax.plot(
            [mp[0], ch[0]],
            [mp[1], ch[1]],
            [mp[2], ch[2]],
            color="#fde047",
            linewidth=2.0,
            alpha=0.95,
            label=f"current frame ({self._event_label_for_frame(f)})",
        )
        self.ax.scatter(
            *mp, color="#fde047", s=60, marker="o", edgecolor="black", linewidth=0.6
        )
        self.ax.scatter(
            *ch, color="#fde047", s=110, marker="s", edgecolor="black", linewidth=0.6
        )

    def _event_label_for_frame(self, f: int) -> str:
        """Return 'A', 'T', 'I', 'F' if frame matches an event, else 'frame N'."""
        for label in ("A", "T", "I", "F"):
            ef = self._frame_for(label)
            if ef is not None and ef == f:
                return self.event_labels.get(label, label)
        return f"frame {f}"

    def _draw_one_pose(self, slot: PoseSlot) -> None:
        # Pick which skeleton to draw: trajectory frame when active, else
        # the slot's static pose.
        skel = self._effective_skeleton(slot)
        if not skel.joints:
            return

        mp = self._mocap_pos_for(slot, "mid")
        ch = self._mocap_pos_for(slot, "club")
        if mp is not None and ch is not None:
            pts = np.array([mp, ch])
            self.ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=slot.mocap_color,
                linewidth=4.5,
                label=f"mocap {slot.name}",
            )
            self.ax.scatter(
                *mp,
                color=slot.mocap_color,
                s=70,
                marker="o",
                edgecolor="black",
                linewidth=0.6,
            )
            self.ax.scatter(
                *ch,
                color=slot.mocap_color,
                s=130,
                marker="s",
                edgecolor="black",
                linewidth=0.6,
            )

        names = list(skel.joints.keys())
        pts = np.array([skel.joints[n] for n in names])
        moved = self.transform.apply(pts)
        pos = {n: moved[i] for i, n in enumerate(names)}

        for parent, child in skel.segments:
            if parent in pos and child in pos:
                a, b = pos[parent], pos[child]
                width = 4.5 if (parent, child) == ("mp", "ch") else 2.6
                self.ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color=slot.color,
                    linewidth=width,
                )

        # Torso-twist indicator: draw a small disk at the torso joint
        # whose plane normal matches the spine-to-hub direction and whose
        # in-plane "+X" axis is aligned with the LS-RS line.  Makes the
        # body coil visible at a glance.
        if self.show_torso_disk and "torso" in pos and "ls" in pos and "rs" in pos:
            self._draw_torso_disk(
                pos["torso"],
                pos["ls"],
                pos["rs"],
                pos.get("hub"),
                pos.get("spine"),
                slot.color,
            )
        # Indicate that this is a trajectory frame (not the static pose)
        # by appending the frame index to the legend label.
        legend = f"sim {slot.name}"
        if slot.trajectory is not None and self.playback_target in ("Skeleton", "Both"):
            legend = (
                f"sim {slot.name} (trajectory frame "
                f"{slot.trajectory_frame_index}/{len(slot.trajectory) - 1})"
            )
        self.ax.scatter(
            moved[:, 0], moved[:, 1], moved[:, 2], color=slot.color, s=24, label=legend
        )
        if "mp" in pos:
            self.ax.scatter(
                *pos["mp"],
                color=slot.color,
                s=70,
                marker="o",
                edgecolor="black",
                linewidth=0.6,
            )
        if "ch" in pos:
            self.ax.scatter(
                *pos["ch"],
                color=slot.color,
                s=130,
                marker="s",
                edgecolor="black",
                linewidth=0.6,
            )

    def _draw_torso_disk(
        self,
        torso: np.ndarray,
        ls: np.ndarray,
        rs: np.ndarray,
        hub: np.ndarray | None,
        spine: np.ndarray | None,
        color: str,
        radius: float = 0.18,
    ) -> None:
        """Draw a small disc at the torso joint to visualise the twist.

        The disc's normal is the spine→hub direction (or world +Z if those
        are missing); the disc is oriented so a marker arrow points in the
        LS direction along the disc plane.  This makes it instantly obvious
        which way the body has coiled.
        """
        # Build an orthonormal frame at the torso joint.
        if hub is not None and spine is not None:
            n = hub - spine
        elif hub is not None:
            n = hub - torso
        else:
            n = np.array([0.0, 0.0, 1.0])
        # ⚡ Bolt: math.hypot is faster than np.linalg.norm for small 1D arrays
        nn = float(math.hypot(n[0], n[1], n[2]))
        n = np.array([0.0, 0.0, 1.0]) if nn < 1e-6 else n / nn

        # In-plane axis: project (rs - ls) onto the plane orthogonal to n.
        rs_dir = rs - ls
        rs_dir = rs_dir - np.dot(rs_dir, n) * n
        rd = float(math.hypot(rs_dir[0], rs_dir[1], rs_dir[2]))
        if rd < 1e-6:
            # Pick any perpendicular if shoulders are degenerate.
            rs_dir = np.array([1.0, 0.0, 0.0])
            rs_dir = rs_dir - np.dot(rs_dir, n) * n
            rd = float(math.hypot(rs_dir[0], rs_dir[1], rs_dir[2]))
            if rd < 1e-6:
                rs_dir = np.array([0.0, 1.0, 0.0])
                rs_dir = rs_dir - np.dot(rs_dir, n) * n
                rd = float(math.hypot(rs_dir[0], rs_dir[1], rs_dir[2]))
                if rd < 1e-6:
                    return
        rs_dir = rs_dir / rd
        n_perp = np.cross(n, rs_dir)
        # Disc points
        thetas = np.linspace(0.0, 2.0 * np.pi, 24)
        disc = torso + radius * (
            np.cos(thetas)[:, None] * rs_dir + np.sin(thetas)[:, None] * n_perp
        )
        self.ax.plot(
            disc[:, 0], disc[:, 1], disc[:, 2], color=color, linewidth=1.5, alpha=0.9
        )
        # Twist-indicator arrow from torso center toward right shoulder.
        tip = torso + (radius * 1.05) * rs_dir
        self.ax.plot(
            [torso[0], tip[0]],
            [torso[1], tip[1]],
            [torso[2], tip[2]],
            color=color,
            linewidth=2.6,
            alpha=0.95,
        )
        self.ax.scatter(
            *tip, color=color, s=24, marker=">", edgecolor="black", linewidth=0.4
        )

    def _effective_skeleton(self, slot: PoseSlot) -> Skeleton:
        """Return the skeleton to draw for this slot.

        When playback target is Skeleton or Both AND the slot has a
        trajectory loaded, returns the trajectory's current frame.
        Otherwise returns the slot's static skeleton.
        """
        if (
            slot.trajectory is not None
            and len(slot.trajectory) > 0
            and self.playback_target in ("Skeleton", "Both")
        ):
            i = max(0, min(slot.trajectory_frame_index, len(slot.trajectory) - 1))
            return slot.trajectory.frames[i]
        return slot.skeleton


__all__ = ["_RenderMixin"]
