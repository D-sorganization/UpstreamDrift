"""Tests for issue #8829: engine dashboards swallowing model-load failures.

Two behaviours are covered:

1. A model-load failure produces a visible status banner instead of a
   silently-successful-looking dashboard (``MuJoCoDashboard``,
   ``PinocchioDashboard``, ``DrakeDashboard`` all previously caught the
   loader exception with ``except Exception: pass`` /
   ``contextlib.suppress(Exception)`` while unconditionally setting a
   title that implied success).
2. The engine + model identity is rendered as an in-body label (not just
   the window title), so it survives ``ExerciseDashboard`` embedding,
   which strips the ``Qt.WindowType.Window`` flag and therefore hides the
   title bar entirely.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.unit

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _StubPhysicsEngine:
    """Minimal PhysicsEngine stand-in sufficient to build a real dashboard.

    Only the methods actually exercised by ``UnifiedDashboardWindow``,
    ``GenericPhysicsRecorder``, and ``GolfSwingPlotter`` construction are
    implemented; that is enough to build a real window without touching
    MuJoCo/Pinocchio/Drake's native bindings.
    """

    def __init__(self) -> None:
        self._time = 0.0
        self._q = np.zeros(6)
        self._v = np.zeros(6)

    @property
    def model_name(self) -> str:
        return "StubModel"

    def load_from_path(self, path: str) -> None:
        pass

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        pass

    def reset(self) -> None:
        self._time = 0.0
        self._q = np.zeros(6)
        self._v = np.zeros(6)

    def step(self, dt: float | None = None) -> None:
        self._time += dt or 0.01

    def forward(self) -> None:
        pass

    def get_state(self):
        return self._q, self._v

    def set_state(self, q, v) -> None:
        self._q = q
        self._v = v

    def set_control(self, u) -> None:
        pass

    def get_time(self) -> float:
        return self._time

    def compute_mass_matrix(self):
        return np.eye(6)

    def compute_bias_forces(self):
        return np.zeros(6)

    def compute_gravity_forces(self):
        return np.zeros(6)

    def compute_inverse_dynamics(self, qacc):
        return np.zeros(6)

    def compute_jacobian(self, body_name):
        return None

    def compute_drift_acceleration(self):
        return np.zeros(6)

    def compute_control_acceleration(self, tau):
        return np.zeros(6)

    def compute_ztcf(self, q, v):
        return np.zeros(6)

    def compute_zvcf(self, q):
        return np.zeros(6)


# ---------------------------------------------------------------------------
# 1. UnifiedDashboardWindow identity strip / failure banner
# ---------------------------------------------------------------------------


class TestIdentityStripAndFailureBanner:
    def test_successful_load_shows_identity_no_banner(self, qapp) -> None:
        from src.shared.python.dashboard.window import (
            ModelLoadStatus,
            UnifiedDashboardWindow,
        )

        status = ModelLoadStatus(engine_name="MuJoCo", model_name="/x/gait.xml")
        win = UnifiedDashboardWindow(_StubPhysicsEngine(), model_status=status)
        try:
            assert "MuJoCo" in win.identity_label.text()
            assert "gait.xml" in win.identity_label.text()
            assert win.status_banner is None
        finally:
            win.close()

    def test_load_failure_produces_visible_banner(self, qapp) -> None:
        from src.shared.python.dashboard.window import (
            ModelLoadStatus,
            UnifiedDashboardWindow,
        )

        status = ModelLoadStatus(
            engine_name="MuJoCo",
            model_name="/x/gait.xml",
            loaded=False,
            error="bad xml",
        )
        win = UnifiedDashboardWindow(_StubPhysicsEngine(), model_status=status)
        try:
            # Identity strip must not silently claim a model loaded.
            assert "not loaded" in win.identity_label.text().lower()
            # A distinct, visible banner communicates the failure -- this
            # is not just a log line, it is a widget in the layout.
            assert win.status_banner is not None
            assert win.status_banner.isVisibleTo(win) or not win.isVisible()
            assert "bad xml" in win.status_banner.text()
            assert win.status_banner.objectName() == "dashboard-status-banner"
        finally:
            win.close()

    def test_no_status_provided_still_identifies_engine(self, qapp) -> None:
        """Backward-compat default: callers that don't pass model_status."""
        from src.shared.python.dashboard.window import UnifiedDashboardWindow

        win = UnifiedDashboardWindow(_StubPhysicsEngine())
        try:
            assert "_StubPhysicsEngine" in win.identity_label.text()
            assert win.status_banner is None
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 2. Engine-specific dashboards surface load failures via ModelLoadStatus
# ---------------------------------------------------------------------------


def _install_fake_engine_module(monkeypatch, modpath: str, attr: str, instance):
    from types import ModuleType
    from unittest.mock import MagicMock

    module = ModuleType(modpath)
    engine_cls = MagicMock(name=attr, return_value=instance)
    engine_cls.__name__ = attr
    setattr(module, attr, engine_cls)
    monkeypatch.setitem(sys.modules, modpath, module)


class TestEngineDashboardsSurfaceLoadFailure:
    def test_mujoco_load_failure_is_visible_not_swallowed(
        self, qapp, monkeypatch, tmp_path
    ) -> None:
        from unittest.mock import MagicMock

        ex_dir = tmp_path / "exercises" / "gait"
        ex_dir.mkdir(parents=True)
        (ex_dir / "scene.xml").write_text("<mujoco/>")

        instance = MagicMock()
        instance.load_from_path.side_effect = RuntimeError("corrupt model")
        _install_fake_engine_module(
            monkeypatch,
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine",
            "MuJoCoPhysicsEngine",
            instance,
        )
        import src.shared.python.config.model_source_providers as _msp

        monkeypatch.setattr(_msp, "mujoco_models_source", lambda: tmp_path)

        from src.launchers.mujoco_dashboard import MuJoCoDashboard

        win = MuJoCoDashboard(exercise_filter="gait")
        try:
            # The title alone used to be the only signal, and it lied.
            # The dashboard must now also carry a non-loaded status.
            assert win.model_status is not None
            assert win.model_status.loaded is False
            assert "corrupt model" in (win.model_status.error or "")
            assert win.status_banner is not None
            assert "corrupt model" in win.status_banner.text()
        finally:
            win.close()

    def test_pinocchio_load_failure_is_visible_not_swallowed(
        self, qapp, monkeypatch, tmp_path
    ) -> None:
        from unittest.mock import MagicMock

        ex_dir = tmp_path / "exercises" / "gait"
        ex_dir.mkdir(parents=True)
        (ex_dir / "leg.urdf").write_text("<robot/>")

        instance = MagicMock()
        instance.load_from_path.side_effect = ValueError("bad urdf")
        _install_fake_engine_module(
            monkeypatch,
            "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine",
            "PinocchioPhysicsEngine",
            instance,
        )
        import src.shared.python.config.model_source_providers as _msp

        monkeypatch.setattr(_msp, "pinocchio_models_source", lambda: tmp_path)

        from src.launchers.pinocchio_dashboard import PinocchioDashboard

        win = PinocchioDashboard(exercise_filter="gait")
        try:
            assert win.model_status is not None
            assert win.model_status.loaded is False
            assert win.status_banner is not None
        finally:
            win.close()

    def test_drake_load_failure_is_visible_not_swallowed(self, qapp, monkeypatch):
        from unittest.mock import MagicMock

        instance = MagicMock()
        instance.load_from_path.side_effect = RuntimeError("bad model")
        _install_fake_engine_module(
            monkeypatch,
            "src.engines.physics_engines.drake.python.drake_physics_engine",
            "DrakePhysicsEngine",
            instance,
        )

        from src.launchers.drake_dashboard import DrakeDashboard

        win = DrakeDashboard(model_path="/oops.urdf")
        try:
            assert win.model_status is not None
            assert win.model_status.loaded is False
            assert win.status_banner is not None
            assert "bad model" in win.status_banner.text()
        finally:
            win.close()

    def test_mujoco_successful_load_has_no_banner(
        self, qapp, monkeypatch, tmp_path
    ) -> None:
        from unittest.mock import MagicMock

        ex_dir = tmp_path / "exercises" / "gait"
        ex_dir.mkdir(parents=True)
        (ex_dir / "scene.xml").write_text("<mujoco/>")

        instance = MagicMock()
        _install_fake_engine_module(
            monkeypatch,
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine",
            "MuJoCoPhysicsEngine",
            instance,
        )
        import src.shared.python.config.model_source_providers as _msp

        monkeypatch.setattr(_msp, "mujoco_models_source", lambda: tmp_path)

        from src.launchers.mujoco_dashboard import MuJoCoDashboard

        win = MuJoCoDashboard(exercise_filter="gait")
        try:
            assert win.model_status is not None
            assert win.model_status.loaded is True
            assert win.status_banner is None
            assert "MuJoCo" in win.identity_label.text()
        finally:
            win.close()


# ---------------------------------------------------------------------------
# 3. The identity strip survives ExerciseDashboard embedding
# ---------------------------------------------------------------------------


class TestIdentityStripSurvivesEmbedding:
    def test_identity_label_present_when_embedded(self, qapp, monkeypatch) -> None:
        """ExerciseDashboard strips the Window flag on embed (issue #8829);
        the identity strip lives in the dashboard's own layout, not the
        title bar, so it must remain a real, attached child widget.
        """
        from PyQt6.QtWidgets import QMainWindow

        from src.shared.python.dashboard.window import (
            ModelLoadStatus,
            UnifiedDashboardWindow,
        )

        status = ModelLoadStatus(engine_name="MuJoCo", model_name="/x/gait.xml")
        dashboard = UnifiedDashboardWindow(_StubPhysicsEngine(), model_status=status)
        try:
            # Mirror ExerciseDashboard._on_engine_changed's embedding step.
            assert isinstance(dashboard, QMainWindow)
            from PyQt6.QtCore import Qt

            dashboard.setWindowFlags(dashboard.windowFlags() & ~Qt.WindowType.Window)

            # The window title is now irrelevant for identification; the
            # in-body label must still carry the engine/model identity.
            assert dashboard.identity_label.parent() is not None
            assert "MuJoCo" in dashboard.identity_label.text()
            assert "gait.xml" in dashboard.identity_label.text()
        finally:
            dashboard.close()

    def test_exercise_dashboard_embeds_real_mujoco_identity(
        self, qapp, monkeypatch, tmp_path
    ) -> None:
        """End-to-end: swap ExerciseDashboard to MuJoCo and confirm the
        embedded child widget exposes its own identity label rather than
        relying solely on the (now-hidden) window title.
        """
        from unittest.mock import MagicMock

        ex_dir = tmp_path / "exercises" / "gait"
        ex_dir.mkdir(parents=True)
        (ex_dir / "scene.xml").write_text("<mujoco/>")

        instance = MagicMock()
        _install_fake_engine_module(
            monkeypatch,
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine",
            "MuJoCoPhysicsEngine",
            instance,
        )
        import src.shared.python.config.model_source_providers as _msp

        monkeypatch.setattr(_msp, "mujoco_models_source", lambda: tmp_path)
        monkeypatch.setattr(
            "src.launchers.exercise_dashboard.discover_exercise",
            lambda x: ["MuJoCo_Models"],
        )

        from src.launchers.exercise_dashboard import ExerciseDashboard

        win = ExerciseDashboard("gait")
        try:
            embedded = win._current_widget
            assert embedded is not None
            # Embedding stripped the Window flag (title bar hidden), but
            # the identity strip is a normal child widget of the layout.
            identity_label = getattr(embedded, "identity_label", None)
            assert identity_label is not None
            assert "MuJoCo" in identity_label.text()
        finally:
            win.close()
