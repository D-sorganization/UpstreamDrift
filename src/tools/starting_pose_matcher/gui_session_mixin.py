"""Session save / load + offsets-export mixin for :class:`MainWidget`.

Holds the JSON serialisation entry points that used to live on
``StartingPoseMatcher`` directly. Split out of ``gui.py`` (Subtask 5 /
#4998 of EPIC #4993) so the per-file budget stays under 1200 lines.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QFileDialog

from src.tools.starting_pose_matcher.core import (
    EVENT_KEYS as _EVENT_KEYS,
    PHASE_KEYS as _PHASE_KEYS,
    SESSION_SCHEMA_VERSION as _SESSION_SCHEMA_VERSION,
    load_simscape_trajectory_csv,
    phase_key_from_label as _phase_key_from_label,
)
from src.tools.starting_pose_matcher.session_schema import (
    BodySkeletonBlock,
    serialize_body_skeleton,
)

logger = logging.getLogger(__name__)


class _SessionMixin:
    """Save/load helpers — JSON offsets export and full-session round-trip."""

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save offsets",
            str(Path(__file__).parent / "starting_pose_offsets.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        out = {
            "transform": {
                "tx": self.transform.tx,
                "ty": self.transform.ty,
                "tz": self.transform.tz,
                "rx": self.transform.rx,
                "ry": self.transform.ry,
                "rz": self.transform.rz,
                "scale": self.transform.scale,
                "pivot": list(self.transform.pivot),
                "lock_xy_rotation": self.lock_xy_rotation,
                "units": {
                    "translation": "metres",
                    "rotation": "degrees",
                    "rotation_order": "Rz @ Ry @ Rx (intrinsic XYZ)",
                },
            },
            "poses": {
                key: {
                    "visible": slot.visible,
                    "event": slot.target_event,
                    "skeleton_source": str(
                        Path(__file__).parent / f"simscape_skeleton_{key}.json"
                    ),
                }
                for key, slot in self.poses.items()
            },
            "events": {
                "A_sample": self.events.A_sample,
                "T_sample": self.events.T_sample,
                "I_sample": self.events.I_sample,
                "F_sample": self.events.F_sample,
                "CHS_mph": self.events.CHS_mph,
            },
            "residuals_mm": self._compute_residuals_mm(),
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=float)
        self._notify(f"Saved: {Path(path).name}")
        logger.info("Wrote %s", path)

    # ---------- session save / load --------------------------------------- #

    def _serialize_session(self) -> dict[str, Any]:
        """Snapshot the entire UI state to a JSON-serialisable dict."""
        return {
            "schema_version": _SESSION_SCHEMA_VERSION,
            "saved_at": pd.Timestamp.now().isoformat(),
            "xlsx_path": self._xlsx_path,
            "sheet": self.sheet_combo.currentText(),
            "transform": {
                "tx": self.transform.tx,
                "ty": self.transform.ty,
                "tz": self.transform.tz,
                "rx": self.transform.rx,
                "ry": self.transform.ry,
                "rz": self.transform.rz,
                "scale": self.transform.scale,
                "pivot": list(self.transform.pivot),
            },
            "lock_xy_rotation": self.lock_xy_rotation,
            "poses": {
                key: {
                    "visible": slot.visible,
                    "event": slot.target_event,
                    "skeleton_path": str(
                        Path(__file__).parent / f"simscape_skeleton_{key}.json"
                    ),
                    "trajectory_path": (
                        slot.trajectory.source_path
                        if slot.trajectory is not None
                        else None
                    ),
                    "trajectory_frame_index": slot.trajectory_frame_index,
                }
                for key, slot in self.poses.items()
            },
            "view": {"elev": float(self.ax.elev), "azim": float(self.ax.azim)},
            "traces": {
                "clubhead": self.show_clubhead_trace,
                "midhands": self.show_midhands_trace,
                "phase": self.phase_window,
                "manual_start": self.manual_window_start,
                "manual_end": self.manual_window_end,
                "frame_marker": self.cb_frame_marker.isChecked(),
            },
            "scene": {
                "ball": self.show_ball,
                "ground": self.show_ground,
                "torso_disk": self.show_torso_disk,
            },
            "playback": {
                "current_frame": self.current_frame,
                "frame_override_active": self.frame_override_active,
                "loop": self.loop_playback,
                "fps": int(self.spin_speed.value()),
                "speed": float(self.playback_speed),
                "trail_frames": int(self.trail_frames),
                "show_trail": bool(self.show_trail),
                "target": self.playback_target,
            },
            "event_overrides": dict(self.event_overrides),
            "event_labels": {
                "preset": self.event_label_preset,
                "labels": dict(self.event_labels),
            },
            # Issue #4480: multi-source toggle state.  Older sessions will
            # not have this block; ``_apply_session`` treats absence as the
            # empty default.
            "data_sources": self._serialize_data_sources(),
            # Issue #4767: body_part_viz renderer style.  Pre-v5 sessions
            # do not carry this block; loaders fall back to "lines".
            "body_skeleton": serialize_body_skeleton(
                BodySkeletonBlock(style=self.body_skeleton_style)
            ),
        }

    def _on_save_session_clicked(self) -> None:
        ses_dir = Path(__file__).parent / "sessions"
        ses_dir.mkdir(exist_ok=True)
        sheet = self.sheet_combo.currentText() or "session"
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save session",
            str(ses_dir / f"{sheet}_{ts}.session.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(self._serialize_session(), f, indent=2, default=float)
        self._notify(f"Saved session: {Path(path).name}")
        logger.info("Wrote session %s", path)

    def _on_load_session_clicked(self) -> None:
        ses_dir = Path(__file__).parent / "sessions"
        start = str(ses_dir) if ses_dir.exists() else str(Path(__file__).parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load session", start, "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Load failed: {exc}")
            return
        self._apply_session(d)
        self._notify(f"Loaded session: {Path(path).name}")

    def _apply_session(self, d: dict[str, Any]) -> None:
        """Restore UI state from a session dict.  Forward-compatible: missing
        keys keep current values, unknown keys are ignored.
        """
        ver = d.get("schema_version", 1)
        if ver > _SESSION_SCHEMA_VERSION:
            logger.warning(
                "Session schema_version=%s newer than supported %s "
                "— ignoring unknown keys.",
                ver,
                _SESSION_SCHEMA_VERSION,
            )

        # 1. Re-load xlsx + sheet (this resets a lot of widgets, so do it first).
        xlsx = d.get("xlsx_path")
        sheet = d.get("sheet")
        if sheet:
            with QSignalBlocker(self.sheet_combo):
                self.sheet_combo.setCurrentText(sheet)
        if xlsx and Path(xlsx).exists():
            self._load_xlsx(xlsx)
        elif xlsx:
            logger.warning("Saved xlsx not found: %s", xlsx)

        # 2. Event overrides (applied on top of the freshly-loaded events).
        evo = d.get("event_overrides") or {}
        for ev, sample in evo.items():
            self.event_overrides[ev] = int(sample)
            setattr(self.events, f"{ev}_sample", float(sample))
        if evo:
            self.lbl_event_info.setText(self._events_summary() + "  (overrides active)")

        # 3. Pose visibility + events + trajectory.
        for key, slot_d in (d.get("poses") or {}).items():
            if key not in self.poses:
                continue
            cb = self._pose_visible_checks.get(key)
            ec = self._pose_event_combos.get(key)
            if cb is not None:
                with QSignalBlocker(cb):
                    cb.setChecked(bool(slot_d.get("visible", True)))
                self.poses[key].visible = cb.isChecked()
            if ec is not None and slot_d.get("event") in ("A", "T", "I", "F"):
                with QSignalBlocker(ec):
                    for i in range(ec.count()):
                        if ec.itemText(i).startswith(slot_d["event"] + " "):
                            ec.setCurrentIndex(i)
                            break
                self.poses[key].target_event = slot_d["event"]
            # Trajectory CSV (optional)
            traj_path = slot_d.get("trajectory_path")
            if traj_path:
                p = Path(traj_path)
                if p.exists():
                    try:
                        self.poses[key].trajectory = load_simscape_trajectory_csv(p)
                        self.poses[key].trajectory_frame_index = int(
                            slot_d.get("trajectory_frame_index", 0)
                        )
                        btn = self._pose_trajectory_buttons.get(key)
                        if btn is not None:
                            btn.setText(f"✓ {len(self.poses[key].trajectory)}f")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not reload trajectory %s: %s", p, exc)

        # 4. Transform sliders.
        tf = d.get("transform") or {}
        for attr, widget in [
            ("tx", self.s_tx),
            ("ty", self.s_ty),
            ("tz", self.s_tz),
            ("rx", self.s_rx),
            ("ry", self.s_ry),
            ("rz", self.s_rz),
            ("scale", self.s_scale),
        ]:
            if attr in tf:
                with QSignalBlocker(widget.spin):
                    widget.set_value(float(tf[attr]))
                with QSignalBlocker(widget.slider):
                    widget.slider.setValue(int(round(float(tf[attr]) / widget._scale)))
                setattr(self.transform, attr, float(tf[attr]))
        if "pivot" in tf:
            self.transform.pivot = tuple(tf["pivot"])

        # 5. Lock-XY rotation.
        if "lock_xy_rotation" in d:
            allow_xy = not bool(d["lock_xy_rotation"])
            with QSignalBlocker(self.cb_lock_xy):
                self.cb_lock_xy.setChecked(allow_xy)
            self.lock_xy_rotation = not allow_xy
            self.s_rx.setEnabled(allow_xy)
            self.s_ry.setEnabled(allow_xy)

        # 6. Camera.
        view = d.get("view") or {}
        if "elev" in view and "azim" in view:
            self.ax.view_init(elev=float(view["elev"]), azim=float(view["azim"]))

        # 7. Traces / phase.
        tr = d.get("traces") or {}
        if "clubhead" in tr:
            with QSignalBlocker(self.cb_clubhead_trace):
                self.cb_clubhead_trace.setChecked(bool(tr["clubhead"]))
            self.show_clubhead_trace = bool(tr["clubhead"])
        if "midhands" in tr:
            with QSignalBlocker(self.cb_midhands_trace):
                self.cb_midhands_trace.setChecked(bool(tr["midhands"]))
            self.show_midhands_trace = bool(tr["midhands"])
        phase_in = tr.get("phase")
        if phase_in is not None:
            # Support both v1 (legacy display strings like "Backswing (A → T)")
            # and v2 (logical keys like "backswing").
            key = _phase_key_from_label(str(phase_in)) if phase_in else None
            if key is None and phase_in in _PHASE_KEYS:
                key = phase_in
            if key in _PHASE_KEYS:
                with QSignalBlocker(self.phase_combo):
                    for i in range(self.phase_combo.count()):
                        if self.phase_combo.itemData(i) == key:
                            self.phase_combo.setCurrentIndex(i)
                            break
                self.phase_window = str(phase_in)
                self.manual_range_widget.setVisible(key == "manual")
        if "manual_start" in tr:
            with QSignalBlocker(self.spin_phase_start):
                self.spin_phase_start.setValue(int(tr["manual_start"]))
            self.manual_window_start = int(tr["manual_start"])
        if "manual_end" in tr:
            with QSignalBlocker(self.spin_phase_end):
                self.spin_phase_end.setValue(int(tr["manual_end"]))
            self.manual_window_end = int(tr["manual_end"])
        if "frame_marker" in tr:
            with QSignalBlocker(self.cb_frame_marker):
                self.cb_frame_marker.setChecked(bool(tr["frame_marker"]))

        # Scene toggles
        scene = d.get("scene") or {}
        for attr, cb_name in (
            ("ball", "cb_show_ball"),
            ("ground", "cb_show_ground"),
            ("torso_disk", "cb_show_torso_disk"),
        ):
            if attr in scene:
                val = bool(scene[attr])
                setattr(self, f"show_{attr}", val)
                cb = getattr(self, cb_name, None)
                if cb is not None:
                    with QSignalBlocker(cb):
                        cb.setChecked(val)

        # 8. Playback.
        pb = d.get("playback") or {}
        if "current_frame" in pb:
            with QSignalBlocker(self.spin_frame):
                self.spin_frame.setValue(int(pb["current_frame"]))
            with QSignalBlocker(self.frame_slider):
                self.frame_slider.setValue(int(pb["current_frame"]))
            self.current_frame = int(pb["current_frame"])
            self._update_time_label()
        if "frame_override_active" in pb:
            with QSignalBlocker(self.cb_use_current_frame):
                self.cb_use_current_frame.setChecked(bool(pb["frame_override_active"]))
            self.frame_override_active = bool(pb["frame_override_active"])
        if "loop" in pb:
            with QSignalBlocker(self.cb_loop):
                self.cb_loop.setChecked(bool(pb["loop"]))
            self.loop_playback = bool(pb["loop"])
        if "fps" in pb:
            with QSignalBlocker(self.spin_speed):
                self.spin_speed.setValue(int(pb["fps"]))
        if "speed" in pb:
            with suppress(TypeError, ValueError):
                self.playback_speed = float(pb["speed"])
            if hasattr(self, "combo_speed"):
                # Snap to closest allowed speed.
                from .session_schema import ALLOWED_SPEEDS as _ALLOWED_SPEEDS

                snap = min(_ALLOWED_SPEEDS, key=lambda s: abs(s - self.playback_speed))
                idx = self.combo_speed.findText(f"{snap}×")
                if idx >= 0:
                    with QSignalBlocker(self.combo_speed):
                        self.combo_speed.setCurrentIndex(idx)
                self.playback_speed = float(snap)
        if "trail_frames" in pb:
            with suppress(TypeError, ValueError):
                self.trail_frames = int(pb["trail_frames"])
            if hasattr(self, "spin_trail"):
                with QSignalBlocker(self.spin_trail):
                    self.spin_trail.setValue(int(self.trail_frames))
        if "show_trail" in pb:
            self.show_trail = bool(pb["show_trail"])
            if hasattr(self, "cb_show_trail"):
                with QSignalBlocker(self.cb_show_trail):
                    self.cb_show_trail.setChecked(self.show_trail)
        if "target" in pb and pb["target"] in ("Mocap", "Skeleton", "Both"):
            with QSignalBlocker(self.combo_playback_target):
                self.combo_playback_target.setCurrentText(pb["target"])
            self.playback_target = pb["target"]

        # 9. Event labels.
        el = d.get("event_labels") or {}
        if "labels" in el and isinstance(el["labels"], dict):
            for k in _EVENT_KEYS:
                if k in el["labels"]:
                    self.event_labels[k] = str(el["labels"][k])
                    if hasattr(self, "_event_label_edits"):
                        with QSignalBlocker(self._event_label_edits[k]):
                            self._event_label_edits[k].setText(self.event_labels[k])
        if "preset" in el:
            self.event_label_preset = str(el["preset"])
            if hasattr(self, "event_preset_combo"):
                idx = self.event_preset_combo.findText(self.event_label_preset)
                if idx < 0:
                    idx = self.event_preset_combo.findText("Custom…")
                if idx >= 0:
                    with QSignalBlocker(self.event_preset_combo):
                        self.event_preset_combo.setCurrentIndex(idx)
        self._refresh_event_label_dependents()

        # Issue #4480: data-sources panel.  Missing block → empty default.
        self._apply_data_sources(d.get("data_sources"))

        # Issue #4767: body-skeleton renderer style.  Missing block on
        # pre-v5 sessions falls back to the default ("lines").
        self._apply_body_skeleton_block(d.get("body_skeleton"))

        self._redraw()


__all__ = ["_SessionMixin"]
