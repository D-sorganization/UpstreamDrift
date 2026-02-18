"""Pinocchio GUI simulation mixin.

Extracts simulation loop, physics integration, recording, and live analysis
computation from PinocchioGUI (gui.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pinocchio as pin  # type: ignore

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["SimulationMixin"]


class SimulationMixin:
    """Mixin providing simulation control methods for PinocchioGUI.

    Provides:
    - ``_toggle_run``: Start/pause simulation
    - ``_reset_simulation``: Reset state to neutral
    - ``_toggle_recording``: Toggle data recording
    - ``_game_loop``: Main timer callback
    - ``_advance_physics``: Single physics step (ABA integration)
    - ``_record_frame``: Record one frame of simulation data
    - ``_find_club_head_state``: Locate club head position/velocity
    - ``_is_analysis_enabled``: Check live analysis flag
    - ``_compute_live_analysis``: Compute induced/counterfactual data
    - ``_compute_specific_sources``: Per-actuator induced acceleration
    - ``_resolve_source_to_tau``: Map source string to torque vector
    """

    def _toggle_run(self: Any, checked: bool = False) -> None:  # noqa: FBT001, FBT002
        self.is_running = not self.is_running
        self.btn_run.setText(
            "Pause Simulation" if self.is_running else "Run Simulation"
        )
        self.btn_run.setChecked(self.is_running)

    def _reset_simulation(self: Any) -> None:
        if self.model is None:
            return
        self.q = pin.neutral(self.model)
        self.v = np.zeros(self.model.nv)
        self.is_running = False
        self.sim_time = 0.0
        self.btn_run.setText("Run Simulation")
        self.btn_run.setChecked(False)
        self._update_viewer()
        self._sync_kinematic_controls()

        # Reset recording
        self.recorder.reset()
        self.lbl_rec_status.setText("Frames: 0")
        if self.btn_record.isChecked():
            self.btn_record.setChecked(False)
            self.btn_record.setText("Record")

    def _toggle_recording(self: Any) -> None:
        """Toggle recording state."""
        if self.btn_record.isChecked():
            self.recorder.start_recording()
            self.log_write("Recording started.")
            self.btn_record.setText("Stop Recording")
        else:
            self.recorder.stop_recording()
            self.log_write(
                f"Recording stopped. Frames: {self.recorder.get_num_frames()}"
            )
            self.btn_record.setText("Record")

    def _game_loop(self: Any) -> None:
        if self.model is None or self.data is None or self.q is None or self.v is None:
            return

        # Always update Live Plot (even if paused, to redraw last frame/resize)
        if hasattr(self, "live_plot"):
            self.live_plot.update_plot()

        if self.operating_mode == "dynamic" and self.is_running:
            self._advance_physics()

            if self.recorder.is_recording:
                self._record_frame()

            self._update_viewer()

    def _advance_physics(self: Any) -> None:
        """Integrate physics forward by one time step."""
        assert self.model is not None
        assert self.data is not None
        assert self.q is not None
        assert self.v is not None
        tau = np.zeros(self.model.nv)
        a = pin.aba(self.model, self.data, self.q, self.v, tau)
        self.v += a * self.dt
        self.q = pin.integrate(self.model, self.q, self.v * self.dt)
        self.sim_time += self.dt

    def _record_frame(self: Any) -> None:
        """Record a single frame of simulation data."""
        assert self.model is not None
        assert self.data is not None
        assert self.q is not None
        assert self.v is not None
        tau = np.zeros(self.model.nv)

        # Compute energies for recording
        pin.computeKineticEnergy(self.model, self.data, self.q, self.v)
        pin.computePotentialEnergy(self.model, self.data, self.q)

        # Capture club head data
        club_head_pos, club_head_vel = self._find_club_head_state()

        q_for_recording = self.q if self.q is not None else np.array([])

        # Induced / Counterfactuals
        induced, counterfactuals = self._compute_live_analysis(tau)

        self.recorder.record_frame(
            time=self.sim_time,
            q=q_for_recording,
            v=self.v,
            tau=tau,
            kinetic_energy=self.data.kinetic_energy,
            potential_energy=self.data.potential_energy,
            club_head_position=club_head_pos,
            club_head_velocity=club_head_vel,
            induced_accelerations=induced,
            counterfactuals=counterfactuals,
        )
        self.lbl_rec_status.setText(f"Frames: {self.recorder.get_num_frames()}")

    def _find_club_head_state(
        self: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Find the club head frame and return its position and velocity."""
        assert self.model is not None
        assert self.data is not None
        club_head_pos = None
        club_head_vel = None

        club_id = -1
        for fid in range(self.model.nframes):
            name = self.model.frames[fid].name.lower()
            if "club" in name or "head" in name:
                club_id = fid
                break

        if club_id == -1 and self.model.nframes > 0:
            club_id = self.model.nframes - 1

        if club_id >= 0:
            pin.forwardKinematics(self.model, self.data, self.q, self.v)
            pin.updateFramePlacements(self.model, self.data)

            frame = self.data.oMf[club_id]
            club_head_pos = frame.translation.copy()

            v_frame = pin.getFrameVelocity(
                self.model,
                self.data,
                club_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            club_head_vel = v_frame.linear.copy()

        return club_head_pos, club_head_vel

    def _is_analysis_enabled(self: Any) -> bool:
        """Check whether live analysis should run based on UI and config."""
        config_requests_analysis = False
        if hasattr(self.recorder, "analysis_config") and isinstance(
            self.recorder.analysis_config, dict
        ):
            cfg = self.recorder.analysis_config
            if (
                cfg.get("ztcf")
                or cfg.get("zvcf")
                or cfg.get("track_drift")
                or cfg.get("track_total_control")
                or cfg.get("induced_accel_sources")
            ):
                config_requests_analysis = True

        return self.chk_live_analysis.isChecked() or config_requests_analysis

    def _compute_live_analysis(
        self: Any, tau: np.ndarray
    ) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
        """Compute induced accelerations and counterfactuals if analysis is enabled.

        Returns:
            Tuple of (induced_accelerations, counterfactuals), each may be None.
        """
        induced = None
        counterfactuals = None

        if not self._is_analysis_enabled():
            return induced, counterfactuals

        if self.analyzer and self.q is not None and self.v is not None:
            induced = self.analyzer.compute_components(self.q, self.v, tau)
            self.latest_induced = induced

            # Compute specific actuator sources
            self._compute_specific_sources(induced)

            if hasattr(self.analyzer, "compute_counterfactuals"):
                counterfactuals = self.analyzer.compute_counterfactuals(self.q, self.v)
                self.latest_cf = counterfactuals

        return induced, counterfactuals

    def _compute_specific_sources(self: Any, induced: dict[str, np.ndarray]) -> None:
        """Compute induced accelerations for specific actuator sources."""
        assert self.analyzer is not None
        assert self.q is not None
        sources_to_compute: list[str] = []
        txt = self.combo_induced.currentText()
        if txt:
            sources_to_compute.append(txt)

        # From config
        has_config = hasattr(self.recorder, "analysis_config")
        config = getattr(self.recorder, "analysis_config", None)
        if has_config and isinstance(config, dict):
            sources = config.get("induced_accel_sources", [])
            if isinstance(sources, list):
                sources_to_compute.extend(sources)

        unique_sources = set()
        for s in sources_to_compute:
            if s:
                unique_sources.add(str(s))

        for src in unique_sources:
            if src in ["gravity", "velocity", "total"]:
                continue

            spec_tau = self._resolve_source_to_tau(src)
            if spec_tau is not None:
                spec_acc = self.analyzer.compute_specific_control(self.q, spec_tau)
                induced[src] = spec_acc

    def _resolve_source_to_tau(self: Any, src: str) -> np.ndarray | None:
        """Resolve an actuator source string to a torque vector.

        Tries joint name, integer index, and comma-separated vector in order.
        Returns None if the source cannot be resolved.
        """
        assert self.model is not None
        spec_tau = np.zeros(self.model.nv)

        # Check if it's a joint name
        if self.model.existJointName(src):
            j_id = self.model.getJointId(src)
            joint = self.model.joints[j_id]
            if joint.nv == 1:
                spec_tau[joint.idx_v] = 1.0
                return spec_tau

        # Check if it's an int index
        try:
            act_idx = int(src)
            if 0 <= act_idx < self.model.nv:
                spec_tau[act_idx] = 1.0
                return spec_tau
        except ValueError:
            pass

        # Check if it's a comma-separated vector
        try:
            parts = [float(x) for x in src.split(",")]
            if len(parts) == self.model.nv:
                return np.array(parts)
        except ValueError:
            pass

        return None
