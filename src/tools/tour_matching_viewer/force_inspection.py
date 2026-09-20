"""Force and torque telemetry and counterfactual inspection widget (MV-06, #10482).

Provides:
- Synchronized GRF, Fz, and CoP telemetry readouts.
- Joint torque telemetry displaying instantaneous peak actuator efforts.
- Interactive counterfactual fork triggering and divergence feedback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.counterfactual import (
    CounterfactualFork,
    CounterfactualStrategy,
)

logger = get_logger(__name__)

__all__ = ["ForceInspectionWidget"]


class ForceInspectionWidget(QtWidgets.QWidget):
    """Panel for inspecting force/torque telemetry and launching counterfactual rollouts."""

    counterfactualForked = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: Any = None
        self._current_frame_idx: int = 0
        self._current_fork: CounterfactualFork | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1. Ground Reaction & CoP Box
        grf_box = QtWidgets.QGroupBox("Ground Reaction & CoP")
        grf_layout = QtWidgets.QVBoxLayout(grf_box)
        grf_layout.setSpacing(2)

        self._fz_label = QtWidgets.QLabel("Fz (Vertical): —")
        self._fnet_label = QtWidgets.QLabel("|F_net|: —")
        self._cop_label = QtWidgets.QLabel("CoP (x, y): —")
        grf_layout.addWidget(self._fz_label)
        grf_layout.addWidget(self._fnet_label)
        grf_layout.addWidget(self._cop_label)
        layout.addWidget(grf_box)

        # 2. Joint Torques Box
        torque_box = QtWidgets.QGroupBox("Actuator Torques")
        torque_layout = QtWidgets.QVBoxLayout(torque_box)
        torque_layout.setSpacing(2)

        self._peak_torque_label = QtWidgets.QLabel("Peak |tau|: —")
        self._torques_summary_label = QtWidgets.QLabel("Active Channels: —")
        torque_layout.addWidget(self._peak_torque_label)
        torque_layout.addWidget(self._torques_summary_label)
        layout.addWidget(torque_box)

        # 3. Counterfactual Rollout Box
        cf_box = QtWidgets.QGroupBox("Counterfactual Rollout")
        cf_layout = QtWidgets.QVBoxLayout(cf_box)
        cf_layout.setSpacing(4)

        cf_layout.addWidget(QtWidgets.QLabel("Intervention Strategy:"))
        self._strategy_combo = QtWidgets.QComboBox()
        self._strategy_combo.addItem(
            "Zero Trail Arm Torque", CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE
        )
        self._strategy_combo.addItem(
            "Clamped Actuator Torque", CounterfactualStrategy.CLAMPED_ACTUATOR_TORQUE
        )
        self._strategy_combo.addItem(
            "Nullspace Exploration", CounterfactualStrategy.NULLSPACE_EXPLORATION
        )
        cf_layout.addWidget(self._strategy_combo)

        self._fork_btn = QtWidgets.QPushButton("Fork Rollout at Current Frame")
        self._fork_btn.setEnabled(False)
        self._fork_btn.clicked.connect(self._on_fork_clicked)
        cf_layout.addWidget(self._fork_btn)

        self._fork_status_label = QtWidgets.QLabel("No active fork")
        self._fork_status_label.setStyleSheet("color: #666666; font-size: 11px;")
        self._divergence_label = QtWidgets.QLabel("Divergence RMS: —")
        cf_layout.addWidget(self._fork_status_label)
        cf_layout.addWidget(self._divergence_label)
        layout.addWidget(cf_box)

        layout.addStretch()

    def set_candidate_session(self, session: Any) -> None:
        """Connect candidate session and update capability states."""
        self._session = session
        self._current_fork = None
        self._fork_status_label.setText("No active fork")
        self._divergence_label.setText("Divergence RMS: —")

        if session is None or not getattr(session, "supports_forces", False):
            self._fork_btn.setEnabled(False)
            self._fork_status_label.setText("Forces unsupported (kinematic candidate)")
            self._clear_telemetry()
            return

        can_fork = getattr(session, "supports_counterfactuals", False)
        self._fork_btn.setEnabled(can_fork)
        if not can_fork:
            self._fork_status_label.setText("Counterfactuals unsupported on session")

        self.update_frame(0)

    def _clear_telemetry(self) -> None:
        self._fz_label.setText("Fz (Vertical): —")
        self._fnet_label.setText("|F_net|: —")
        self._cop_label.setText("CoP (x, y): —")
        self._peak_torque_label.setText("Peak |tau|: —")
        self._torques_summary_label.setText("Active Channels: —")

    def update_frame(self, frame_idx: int) -> None:
        """Update synchronized telemetry display for the given frame index."""
        self._current_frame_idx = frame_idx
        if self._session is None or not getattr(
            self._session, "supports_forces", False
        ):
            return

        session = self._session
        wrench = session.get_wrench_at(frame_idx)
        if wrench is not None:
            fn = wrench.force_n
            fz = fn[2]
            fnet = float(np.linalg.norm(fn))
            self._fz_label.setText(f"Fz (Vertical): {fz:.1f} N")
            self._fnet_label.setText(f"|F_net|: {fnet:.1f} N")
        else:
            self._fz_label.setText("Fz (Vertical): —")
            self._fnet_label.setText("|F_net|: —")

        cop = session.get_center_of_pressure(frame_idx)
        if cop is not None:
            self._cop_label.setText(
                f"CoP (x, y): ({cop[0] * 1000.0:.1f}, {cop[1] * 1000.0:.1f}) mm"
            )
        else:
            self._cop_label.setText("CoP (x, y): Undefined (Fz <= 5.0 N)")

        torques = session.get_joint_torques_at(frame_idx)
        if torques is not None and len(torques) > 0:
            max_j = max(torques.items(), key=lambda item: abs(item[1]))
            self._peak_torque_label.setText(
                f"Peak |tau|: {abs(max_j[1]):.1f} N*m ({max_j[0]})"
            )
            self._torques_summary_label.setText(f"Active Channels: {len(torques)}")
        else:
            self._peak_torque_label.setText("Peak |tau|: —")
            self._torques_summary_label.setText("Active Channels: —")

    def _on_fork_clicked(self) -> None:
        """Trigger counterfactual rollout at the current playback frame."""
        if self._session is None or not getattr(
            self._session, "supports_counterfactuals", False
        ):
            return

        strategy = self._strategy_combo.currentData()
        if strategy is None:
            strategy = CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE

        try:
            fork = self._session.create_counterfactual_fork(
                fork_frame_idx=self._current_frame_idx,
                strategy=strategy,
            )
            self._current_fork = fork
            strat_val = fork.strategy.value
            self._fork_status_label.setText(
                f"Forked @ frame {fork.fork_frame_idx} ({strat_val})"
            )
            self._divergence_label.setText(f"Divergence RMS: {fork.divergence_rms:.4f}")
            self.counterfactualForked.emit(fork)
        except Exception as exc:
            logger.exception("Counterfactual fork failed: %s", exc)
            self._fork_status_label.setText(f"Fork failed: {exc}")
