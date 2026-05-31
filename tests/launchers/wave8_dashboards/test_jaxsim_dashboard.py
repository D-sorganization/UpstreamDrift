"""Tests for ``src.launchers.jaxsim_dashboard`` (issue #6658).

These tests never import ``jaxsim``/``jax`` (Linux-only): the dashboard reads
the backend's declared ``EngineCapabilities`` dataclass, which is produced
without loading a model. They therefore run on every platform, including the
Windows dev box.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.engines.physics_engines.jaxsim import JaxSimBackend  # noqa: E402
from src.shared.python.engine_core.capabilities import (  # noqa: E402
    CapabilityLevel,
    EngineCapabilities,
)
from src.launchers.jaxsim_dashboard import (  # noqa: E402
    PARAMETER_SENSITIVITY_GATED_ISSUE,
    JaxSimDashboard,
)


class TestJaxSimDashboard:
    def test_full_capabilities_enable_their_controls(self, qapp) -> None:
        win = JaxSimDashboard(exercise_filter="gait")
        try:
            caps = JaxSimBackend().get_capabilities()
            # forward_sim and mass_matrix are FULL -> enabled.
            assert caps.forward_sim == CapabilityLevel.FULL
            assert win.feature_controls["Forward simulation"].isEnabled()
            assert win.feature_controls["Mass matrix M(q)"].isEnabled()
            assert win.feature_controls["Inverse dynamics"].isEnabled()
            assert win.feature_controls["Spatial Jacobian"].isEnabled()
        finally:
            win.deleteLater()

    def test_partial_capabilities_are_greyed_out(self, qapp) -> None:
        win = JaxSimDashboard(exercise_filter="gait")
        try:
            caps = JaxSimBackend().get_capabilities()
            # contact_forces and drift_acceleration are PARTIAL -> disabled.
            assert caps.contact_forces == CapabilityLevel.PARTIAL
            assert caps.drift_acceleration == CapabilityLevel.PARTIAL
            assert not win.feature_controls["Contact forces"].isEnabled()
            assert not win.feature_controls["Drift / ZTCF acceleration"].isEnabled()
        finally:
            win.deleteLater()

    def test_parameter_sensitivity_entry_is_stubbed_and_gated(self, qapp) -> None:
        win = JaxSimDashboard(exercise_filter="gait")
        try:
            button = win.parameter_sensitivity_button
            assert not button.isEnabled()
            assert str(PARAMETER_SENSITIVITY_GATED_ISSUE) in button.toolTip()
        finally:
            win.deleteLater()

    def test_capability_injection_drives_enable_state(self, qapp) -> None:
        # Inject a capability report flipping a normally-enabled feature off
        # and a normally-disabled feature on, proving the wiring is data-driven.
        caps = EngineCapabilities(
            engine_name="JaxSim",
            forward_sim=CapabilityLevel.NONE,
            contact_forces=CapabilityLevel.FULL,
        )
        win = JaxSimDashboard(exercise_filter="run", capabilities=caps)
        try:
            assert not win.feature_controls["Forward simulation"].isEnabled()
            assert win.feature_controls["Contact forces"].isEnabled()
        finally:
            win.deleteLater()

    def test_blank_exercise_filter_rejected(self, qapp) -> None:
        with pytest.raises(ValueError, match="exercise_filter"):
            JaxSimDashboard(exercise_filter="   ")
