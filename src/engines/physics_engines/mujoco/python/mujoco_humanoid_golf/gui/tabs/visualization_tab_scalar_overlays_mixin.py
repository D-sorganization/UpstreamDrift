"""Stored-KE and net-power overlay controls for the visualization tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6 import QtCore, QtWidgets

from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale
from src.shared.python.body_part_viz.segment_net_power import (
    SEGMENT_NET_POWER_LABEL,
    default_net_power_color_scale,
)
from src.shared.python.core.contracts import require
from src.shared.python.segment_kinetic_energy import SEGMENT_KE_PROXY_DISCLAIMER
from src.shared.python.segment_kinetic_energy.shading import (
    KE_COMPONENTS,
    KE_PROXY_GLOBE_LABEL,
    STORED_KINETIC_ENERGY_VIEW_LABEL,
)

if TYPE_CHECKING:
    from ...sim_widget import MuJoCoSimWidget


class VisualizationTabScalarOverlaysMixin:
    """Force-tab overlays: stored segment KE, net power, and status-bar legends."""

    sim_widget: MuJoCoSimWidget

    def _connect_overlay_legends_to_status_bar(self) -> None:
        """Show force, stored-KE, and net-power legend lines in the status bar."""
        status_bar = getattr(self.window(), "statusBar", None)
        if status_bar is None:
            return
        force_signal = getattr(self.sim_widget, "force_legend_changed", None)
        ke_signal = getattr(self.sim_widget, "ke_legend_changed", None)
        net_signal = getattr(self.sim_widget, "net_power_legend_changed", None)
        self._legend_lines = {"force": "", "ke": "", "net_power": ""}

        def _refresh(
            *,
            force: str | None = None,
            ke: str | None = None,
            net_power: str | None = None,
        ) -> None:
            if force is not None:
                self._legend_lines["force"] = force
            if ke is not None:
                self._legend_lines["ke"] = ke
            if net_power is not None:
                self._legend_lines["net_power"] = net_power
            combined = " · ".join(
                self._legend_lines[key]
                for key in ("net_power", "ke", "force")
                if self._legend_lines[key]
            )
            status_bar().showMessage(combined)

        if force_signal is not None:
            force_signal.connect(lambda text: _refresh(force=text))
        if ke_signal is not None:
            ke_signal.connect(lambda text: _refresh(ke=text))
        if net_signal is not None:
            net_signal.connect(lambda text: _refresh(net_power=text))

    def _create_stored_ke_shading_controls(
        self, force_layout: QtWidgets.QVBoxLayout
    ) -> None:
        if force_layout is None:
            raise ValueError("force_layout must be provided")
        self.show_stored_ke_cb = QtWidgets.QCheckBox(STORED_KINETIC_ENERGY_VIEW_LABEL)
        self.show_stored_ke_cb.setObjectName("stored_kinetic_energy_enabled")
        self.show_stored_ke_cb.setToolTip(
            "Shade segment surfaces by stored kinetic energy (J) with fixed cividis "
            "limits across the swing."
        )
        scale = getattr(self.sim_widget, "stored_ke_color_scale", None)
        if isinstance(scale, ScalarColorScale) and scale.enabled:
            self.show_stored_ke_cb.setChecked(True)
        self.show_stored_ke_cb.stateChanged.connect(self._on_stored_ke_enabled_changed)
        force_layout.addWidget(self.show_stored_ke_cb)

        form = QtWidgets.QFormLayout()
        self.stored_ke_mode_combo = QtWidgets.QComboBox()
        for mode in KE_COMPONENTS:
            self.stored_ke_mode_combo.addItem(mode.replace("_", " ").title(), mode)
        current_mode = getattr(self.sim_widget, "stored_ke_component_mode", "total")
        if current_mode not in KE_COMPONENTS:
            current_mode = "total"
        idx = self.stored_ke_mode_combo.findData(current_mode)
        if isinstance(idx, int) and idx >= 0:
            self.stored_ke_mode_combo.setCurrentIndex(idx)
        self.stored_ke_mode_combo.currentIndexChanged.connect(
            self._on_stored_ke_mode_changed
        )
        form.addRow("KE component:", self.stored_ke_mode_combo)

        self.stored_ke_sqrt_cb = QtWidgets.QCheckBox("Sqrt display scale (ticks in J)")
        if isinstance(scale, ScalarColorScale):
            self.stored_ke_sqrt_cb.setChecked(scale.display_transform == "sqrt")
        self.stored_ke_sqrt_cb.stateChanged.connect(self._on_stored_ke_sqrt_changed)
        form.addRow("", self.stored_ke_sqrt_cb)

        from src.shared.python.segment_kinetic_energy.shading import (
            REFERENCE_SWING_PEAK_SEGMENT_KE_J,
        )

        self.stored_ke_high_limit_spin = QtWidgets.QDoubleSpinBox()
        self.stored_ke_high_limit_spin.setDecimals(1)
        self.stored_ke_high_limit_spin.setMinimum(1.0)
        self.stored_ke_high_limit_spin.setMaximum(1e7)
        self.stored_ke_high_limit_spin.setSuffix(" J")
        self.stored_ke_high_limit_spin.setToolTip(
            "Upper colour-scale limit (J) for stored kinetic-energy shading."
        )
        high = (
            float(scale.high_limit)
            if isinstance(scale, ScalarColorScale)
            else float(REFERENCE_SWING_PEAK_SEGMENT_KE_J)
        )
        self.stored_ke_high_limit_spin.setValue(high)
        self.stored_ke_high_limit_spin.valueChanged.connect(
            self._on_stored_ke_high_limit_changed
        )
        form.addRow("KE high limit:", self.stored_ke_high_limit_spin)
        force_layout.addLayout(form)

    def _on_stored_ke_enabled_changed(self, state: int) -> None:
        from dataclasses import replace

        scale = getattr(self.sim_widget, "stored_ke_color_scale", None)
        if scale is None:
            return
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_stored_ke_color_scale(replace(scale, enabled=enabled))

    def _on_stored_ke_mode_changed(self, index: int) -> None:
        mode = self.stored_ke_mode_combo.itemData(index)
        if isinstance(mode, str):
            self.sim_widget.set_stored_ke_component_mode(mode)

    def _on_stored_ke_sqrt_changed(self, state: int) -> None:
        from dataclasses import replace

        scale = getattr(self.sim_widget, "stored_ke_color_scale", None)
        if scale is None:
            return
        transform = "sqrt" if state == QtCore.Qt.CheckState.Checked.value else "linear"
        self.sim_widget.set_stored_ke_color_scale(
            replace(scale, display_transform=transform)
        )

    def _on_stored_ke_high_limit_changed(self, value: float) -> None:
        from dataclasses import replace

        scale = getattr(self.sim_widget, "stored_ke_color_scale", None)
        if scale is None or value <= 0.0:
            return
        self.sim_widget.set_stored_ke_color_scale(
            replace(scale, high_limit=float(value))
        )

    def _create_segment_ke_globe_controls(
        self, force_layout: QtWidgets.QVBoxLayout
    ) -> None:
        if force_layout is None:
            raise ValueError("force_layout must be provided")
        self.show_segment_ke_globes_cb = QtWidgets.QCheckBox(KE_PROXY_GLOBE_LABEL)
        self.show_segment_ke_globes_cb.setToolTip(SEGMENT_KE_PROXY_DISCLAIMER)
        self._restore_checkbox(  # type: ignore[attr-defined]
            self.show_segment_ke_globes_cb,
            getattr(self.sim_widget, "show_segment_ke_globes", None),
        )
        self.show_segment_ke_globes_cb.stateChanged.connect(
            self.on_show_segment_ke_globes_changed
        )
        force_layout.addWidget(self.show_segment_ke_globes_cb)

    def _sim_net_power_scale(self) -> ScalarColorScale | None:
        scale = getattr(self.sim_widget, "net_power_color_scale", None)
        return scale if isinstance(scale, ScalarColorScale) else None

    def _create_net_power_controls(self, force_layout: QtWidgets.QVBoxLayout) -> None:
        if force_layout is None:
            raise ValueError("force_layout must be provided")
        self.show_net_power_shading_cb = QtWidgets.QCheckBox(SEGMENT_NET_POWER_LABEL)
        self.show_net_power_shading_cb.setToolTip(
            "Diverging segment shading for mechanical energy rate (watts)"
        )
        scale = self._sim_net_power_scale()
        stored = bool(scale is not None and scale.enabled)
        self._restore_checkbox(self.show_net_power_shading_cb, stored)  # type: ignore[attr-defined]
        self.show_net_power_shading_cb.stateChanged.connect(
            self.on_show_net_power_shading_changed
        )
        force_layout.addWidget(self.show_net_power_shading_cb)

        self.show_interface_transfer_cb = QtWidgets.QCheckBox(
            "Interface transfer markers"
        )
        self.show_interface_transfer_cb.setToolTip(
            "Robertson–Winter interface power glyphs on parent/child joints"
        )
        self._restore_checkbox(  # type: ignore[attr-defined]
            self.show_interface_transfer_cb,
            getattr(self.sim_widget, "show_interface_power_glyphs", None),
        )
        self.show_interface_transfer_cb.stateChanged.connect(
            self.on_show_interface_transfer_changed
        )
        force_layout.addWidget(self.show_interface_transfer_cb)

        limit_row = QtWidgets.QHBoxLayout()
        limit_row.addWidget(QtWidgets.QLabel("Net-power ±W limit:"))
        self.net_power_limit_spin = QtWidgets.QDoubleSpinBox()
        self.net_power_limit_spin.setRange(1.0, 100_000.0)
        self.net_power_limit_spin.setDecimals(1)
        self.net_power_limit_spin.setSingleStep(50.0)
        scale = self._sim_net_power_scale() or default_net_power_color_scale()
        self.net_power_limit_spin.setValue(abs(float(scale.high_limit)))
        self.net_power_limit_spin.valueChanged.connect(self.on_net_power_limit_changed)
        limit_row.addWidget(self.net_power_limit_spin, stretch=1)
        force_layout.addLayout(limit_row)

    def on_net_power_limit_changed(self, value: float) -> None:
        from dataclasses import replace

        limit = float(value)
        if limit <= 0.0:
            return
        require(self.sim_widget is not None, "sim_widget is set")
        scale = self._sim_net_power_scale() or default_net_power_color_scale()
        setter = getattr(self.sim_widget, "set_net_power_color_scale", None)
        if callable(setter):
            setter(replace(scale, low_limit=-limit, high_limit=limit))

    def on_show_net_power_shading_changed(self, state: int) -> None:
        if state is None:
            raise ValueError("state must be provided")
        require(self.sim_widget is not None, "sim_widget is set")
        enabled = state == QtCore.Qt.CheckState.Checked.value
        from dataclasses import replace

        scale = self._sim_net_power_scale() or default_net_power_color_scale()
        setter = getattr(self.sim_widget, "set_net_power_color_scale", None)
        if callable(setter):
            setter(replace(scale, enabled=enabled))

    def on_show_interface_transfer_changed(self, state: int) -> None:
        if state is None:
            raise ValueError("state must be provided")
        require(self.sim_widget is not None, "sim_widget is set")
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_interface_power_glyphs(enabled)

    def on_show_segment_ke_globes_changed(self, state: int) -> None:
        if state is None:
            raise ValueError("state must be provided")
        require(self.sim_widget is not None, "sim_widget is set")
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_segment_ke_globe_visualization(enabled)


__all__ = ["VisualizationTabScalarOverlaysMixin"]
