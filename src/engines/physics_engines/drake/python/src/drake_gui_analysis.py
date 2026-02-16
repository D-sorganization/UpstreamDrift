"""Drake GUI Analysis & Plotting Mixin.

Contains induced acceleration analysis, counterfactual computation,
and all post-hoc plotting methods.
"""

from __future__ import annotations

import contextlib

import numpy as np

from src.shared.python.engine_core.engine_availability import (
    MATPLOTLIB_AVAILABLE,
    PYQT6_AVAILABLE,
)
from src.shared.python.logging_pkg.logging_config import get_logger

HAS_QT = PYQT6_AVAILABLE
HAS_MATPLOTLIB = MATPLOTLIB_AVAILABLE

if HAS_QT:
    from PyQt6 import QtCore, QtWidgets

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt


from .drake_analysis import DrakeInducedAccelerationAnalyzer  # noqa: E402

LOGGER = get_logger(__name__)


class AnalysisMixin:
    """Mixin providing analysis and plotting methods for DrakeSimApp."""

    def _is_analysis_enabled(self) -> bool:
        """Check whether live analysis should run based on UI and config."""
        config_requests_analysis = False
        if hasattr(self.recorder, "analysis_config") and isinstance(  # type: ignore[attr-defined]
            self.recorder.analysis_config,
            dict,  # type: ignore[attr-defined]
        ):
            cfg = self.recorder.analysis_config  # type: ignore[attr-defined]
            if (
                cfg.get("ztcf")
                or cfg.get("zvcf")
                or cfg.get("track_drift")
                or cfg.get("track_total_control")
                or cfg.get("induced_accel_sources")
            ):
                config_requests_analysis = True

        return self.chk_live_analysis.isChecked() or config_requests_analysis  # type: ignore[attr-defined]

    def _compute_live_analysis(self, q: np.ndarray, v: np.ndarray) -> None:
        """Compute induced accelerations and counterfactuals for the current state."""
        assert self.plant is not None  # type: ignore[attr-defined]
        assert self.eval_context is not None  # type: ignore[attr-defined]
        # Update eval context
        self.plant.SetPositions(self.eval_context, q)  # type: ignore[attr-defined]
        self.plant.SetVelocities(self.eval_context, v)  # type: ignore[attr-defined]

        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)  # type: ignore[attr-defined]

        # Compute Induced
        res = analyzer.compute_components(self.eval_context)  # type: ignore[attr-defined]

        # Compute specific actuator sources
        self._compute_specific_sources(analyzer, res)

        # Append to recorder lists (DrakeRecorder uses dict[str, list[np.ndarray]])
        for k, val in res.items():
            if k not in self.recorder.induced_accelerations:  # type: ignore[attr-defined]
                self.recorder.induced_accelerations[k] = []  # type: ignore[attr-defined]
            self.recorder.induced_accelerations[k].append(val)  # type: ignore[attr-defined]

        # Compute Counterfactuals
        cf_res = analyzer.compute_counterfactuals(self.eval_context)  # type: ignore[attr-defined]
        for k, val in cf_res.items():
            if k not in self.recorder.counterfactuals:  # type: ignore[attr-defined]
                self.recorder.counterfactuals[k] = []  # type: ignore[attr-defined]
            self.recorder.counterfactuals[k].append(val)  # type: ignore[attr-defined]

    def _compute_specific_sources(
        self,
        analyzer: DrakeInducedAccelerationAnalyzer,
        res: dict[str, np.ndarray],
    ) -> None:
        """Compute induced accelerations for specific actuator sources."""
        sources_to_compute: list[str] = []

        # 1. From GUI combo
        if self.chk_induced_vec.isChecked():  # type: ignore[attr-defined]
            sources_to_compute.append(self.combo_induced_source.currentText())  # type: ignore[attr-defined]

        # 2. From LivePlotWidget config
        if hasattr(self.recorder, "analysis_config") and isinstance(  # type: ignore[attr-defined]
            self.recorder.analysis_config,
            dict,  # type: ignore[attr-defined]
        ):
            sources = self.recorder.analysis_config.get("induced_accel_sources", [])  # type: ignore[attr-defined]
            if isinstance(sources, list):
                sources_to_compute.extend(sources)

        # Deduplicate and compute specific sources
        unique_sources = set()
        for src in sources_to_compute:
            if src:
                unique_sources.add(str(src))

        assert self.plant is not None  # type: ignore[attr-defined]
        assert self.eval_context is not None  # type: ignore[attr-defined]
        for source in unique_sources:
            if source in ["gravity", "velocity", "total"]:
                continue

            try:
                act_idx = -1
                try:
                    act_idx = int(source)
                except ValueError:
                    if self.plant.HasJointNamed(source):  # type: ignore[attr-defined]
                        joint = self.plant.GetJointByName(source)  # type: ignore[attr-defined]
                        if joint.num_velocities() == 1:
                            act_idx = joint.velocity_start()

                if act_idx >= 0:
                    tau_vec = np.zeros(self.plant.num_velocities())  # type: ignore[attr-defined]
                    if 0 <= act_idx < len(tau_vec):
                        tau_vec[act_idx] = 1.0
                        accels = analyzer.compute_specific_control(
                            self.eval_context,
                            tau_vec,  # type: ignore[attr-defined]
                        )
                        res[source] = accels
            except (ValueError, TypeError, RuntimeError):
                pass

    def _show_induced_acceleration_plot(self) -> None:
        """Calculate and plot induced accelerations."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")  # type: ignore[arg-type]
            return

        from shared.python.plotting import GolfSwingPlotter

        if not self.recorder.times:  # type: ignore[attr-defined]
            QtWidgets.QMessageBox.warning(
                self,  # type: ignore[arg-type]
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        if not self.plant or not self.eval_context:  # type: ignore[attr-defined]
            return

        spec_act_idx = -1
        txt = self.combo_induced_source.currentText()  # type: ignore[attr-defined]
        if txt and txt not in ["gravity", "velocity", "total"]:
            with contextlib.suppress(ValueError):
                spec_act_idx = int(txt)

        g_induced = []
        c_induced = []
        spec_induced = []
        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)  # type: ignore[attr-defined]

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

            for _i, (q, v) in enumerate(
                zip(self.recorder.q_history, self.recorder.v_history, strict=False)  # type: ignore[attr-defined]
            ):
                self.plant.SetPositions(self.eval_context, q)  # type: ignore[attr-defined]
                self.plant.SetVelocities(self.eval_context, v)  # type: ignore[attr-defined]

                res = analyzer.compute_components(self.eval_context)  # type: ignore[attr-defined]

                g_induced.append(res["gravity"])
                c_induced.append(res["velocity"])

                if spec_act_idx >= 0:
                    tau = np.zeros(self.plant.num_velocities())  # type: ignore[attr-defined]
                    if spec_act_idx < len(tau):
                        tau[spec_act_idx] = 1.0
                    spec = analyzer.compute_specific_control(self.eval_context, tau)  # type: ignore[attr-defined]
                    spec_induced.append(spec)

        except (ValueError, TypeError, RuntimeError) as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Analysis Error", str(e))  # type: ignore[arg-type]
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        g_induced_arr = np.array(g_induced)
        c_induced_arr = np.array(c_induced)
        total_arr = g_induced_arr + c_induced_arr

        self.recorder.induced_accelerations["gravity"] = list(g_induced_arr)  # type: ignore[attr-defined]
        self.recorder.induced_accelerations["velocity"] = list(c_induced_arr)  # type: ignore[attr-defined]
        self.recorder.induced_accelerations["total"] = list(total_arr)  # type: ignore[attr-defined]

        if spec_induced:
            self.recorder.induced_accelerations["control"] = list(  # type: ignore[attr-defined]
                np.array(spec_induced)
            )

        joint_idx = 0
        if g_induced_arr.shape[1] > 2:
            joint_idx = 2

        plotter = GolfSwingPlotter(self.recorder)  # type: ignore[attr-defined]
        fig = plt.figure(figsize=(10, 6))

        plotter.plot_induced_acceleration(
            fig, "breakdown", joint_idx=joint_idx, breakdown_mode=True
        )
        plt.show()

    def _show_counterfactuals_plot(self) -> None:
        """Calculate and plot Counterfactuals (ZTCF/ZVCF)."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")  # type: ignore[arg-type]
            return

        from shared.python.plotting import GolfSwingPlotter

        if not self.recorder.times:  # type: ignore[attr-defined]
            QtWidgets.QMessageBox.warning(
                self,  # type: ignore[arg-type]
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        if not self.plant or not self.eval_context:  # type: ignore[attr-defined]
            return

        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)  # type: ignore[attr-defined]
        ztcf_list = []
        zvcf_list = []

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

            for q, v in zip(
                self.recorder.q_history,
                self.recorder.v_history,
                strict=False,  # type: ignore[attr-defined]
            ):
                self.plant.SetPositions(self.eval_context, q)  # type: ignore[attr-defined]
                self.plant.SetVelocities(self.eval_context, v)  # type: ignore[attr-defined]

                res = analyzer.compute_counterfactuals(self.eval_context)  # type: ignore[attr-defined]
                ztcf_list.append(res["ztcf_accel"])
                zvcf_list.append(res["zvcf_torque"])

        except (RuntimeError, ValueError, OSError) as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Analysis Error", str(e))  # type: ignore[arg-type]
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.recorder.counterfactuals["ztcf_accel"] = list(np.array(ztcf_list))  # type: ignore[attr-defined]
        self.recorder.counterfactuals["zvcf_torque"] = list(np.array(zvcf_list))  # type: ignore[attr-defined]

        joint_idx = 0
        if np.array(ztcf_list).shape[1] > 2:
            joint_idx = 2

        plotter = GolfSwingPlotter(self.recorder)  # type: ignore[attr-defined]
        fig = plt.figure(figsize=(10, 6))

        plotter.plot_counterfactual_comparison(fig, "dual", metric_idx=joint_idx)
        plt.show()

    def _show_swing_plane_analysis(self) -> None:
        """Show swing plane analysis using shared plotter."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")  # type: ignore[arg-type]
            return

        from shared.python.plotting import GolfSwingPlotter

        if not self.recorder.times:  # type: ignore[attr-defined]
            QtWidgets.QMessageBox.warning(
                self,  # type: ignore[arg-type]
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        plotter = GolfSwingPlotter(self.recorder)  # type: ignore[attr-defined]
        fig = plt.figure(figsize=(10, 8))
        plotter.plot_swing_plane(fig)
        plt.show()

    def _show_advanced_plots(self) -> None:
        """Show advanced analysis plots."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")  # type: ignore[arg-type]
            return

        from shared.python.plotting import GolfSwingPlotter
        from shared.python.validation_pkg.statistical_analysis import (
            StatisticalAnalyzer,
        )

        if not self.recorder.times:  # type: ignore[attr-defined]
            QtWidgets.QMessageBox.warning(
                self,  # type: ignore[arg-type]
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        plotter = GolfSwingPlotter(self.recorder)  # type: ignore[attr-defined]

        times = np.array(self.recorder.times)  # type: ignore[attr-defined]
        q_history = np.array(self.recorder.q_history)  # type: ignore[attr-defined]
        v_history = np.array(self.recorder.v_history)  # type: ignore[attr-defined]
        tau_history = np.zeros((len(times), v_history.shape[1]))

        _, club_pos = self.recorder.get_time_series("club_head_position")  # type: ignore[attr-defined]

        # Convert club position to numpy array if needed
        club_head_pos = np.array(club_pos) if isinstance(club_pos, list) else club_pos

        analyzer = StatisticalAnalyzer(
            times, q_history, v_history, tau_history, club_head_position=club_head_pos
        )
        report = analyzer.generate_comprehensive_report()

        metrics = {
            "Club Speed": 0.0,
            "Swing Efficiency": 0.0,
            "Tempo": 0.0,
            "Consistency": 0.8,
            "Power Transfer": 0.0,
        }

        if "club_head_speed" in report:
            peak_speed = report["club_head_speed"]["peak_value"]
            metrics["Club Speed"] = min(peak_speed / 50.0, 1.0)

        if "tempo" in report:
            ratio = report["tempo"]["ratio"]
            error = abs(ratio - 3.0)
            metrics["Tempo"] = max(0, 1.0 - error)

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        plotter.plot_radar_chart(fig, metrics)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, "CoP Data Not Available in Drake", ha="center", va="center")

        ax3 = fig.add_subplot(gs[1, :])
        ax3.text(
            0.5, 0.5, "Power Data Not Available in Drake", ha="center", va="center"
        )

        plt.tight_layout()
        plt.show()
