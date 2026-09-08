"""``MuJoCoSimWidget`` must not outlive its own simulation timer (UD #9474).

Why this matters
----------------
``MuJoCoSimWidget.__init__`` starts a 60 fps ``QTimer`` unconditionally, and
until this change the class had no ``closeEvent``, no ``hideEvent`` and no
teardown method. ``QWidget.close()`` only *hides* a widget, so every sim widget
a test constructed kept stepping MuJoCo and rendering frames for the remaining
lifetime of the ``QApplication`` -- which, because the unit lane's
``QApplication`` is a process-wide singleton, means the rest of the pytest
session.

In CI that leaked timer fired during an unrelated later test, reached
``SimRenderingMixin._add_force_torque_overlays`` -> ``get_cv2()``, and the
``import cv2`` there raised ``AttributeError`` (a half-initialised OpenCV, not
an ``ImportError``). An exception escaping a Qt slot is fatal under PyQt6, so
the interpreter aborted:

    AttributeError: partially initialized module 'cv2' has no attribute
    'mat_wrapper' (most likely due to a circular import)
    Fatal Python error: Aborted

The lane produced no test summary at all. Both halves are covered here: the
widget must stop its own timer when it is closed, and ``get_cv2`` must honour
its documented contract of returning the module or ``None``.

The timer *body* responsibilities (UD #9492) are covered below as well: the
dynamic-mode stepping loop, control application, the live-analysis/recording
hook and the live-kinematics overlays are pinned with fakes so the
decomposition of ``_on_timer`` and
``SimRenderingMixin._add_live_kinematics_overlays`` cannot change behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

pytest.importorskip("PyQt6", reason="PyQt6 required for Qt lifetime tests")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo required for the sim widget")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (  # noqa: E402
    sim_rendering_mixin,
    sim_widget,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_rendering_mixin import (  # noqa: E402
    SimRenderingMixin,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import (  # noqa: E402
    MuJoCoSimWidget,
)
from src.shared.python.ui.qt.utils import get_qapp  # noqa: E402


@pytest.fixture
def qapp():
    try:
        return get_qapp()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qt initialisation failed (headless environment?): {exc}")


@pytest.fixture
def widget(qapp):
    """A sim widget that is always torn down, even when a test fails."""
    made = MuJoCoSimWidget(width=64, height=48, fps=30)
    try:
        yield made
    finally:
        made.stop_simulation()
        made.deleteLater()


def test_timer_is_running_after_construction(widget) -> None:
    """Baseline: the widget really does start a timer in ``__init__``."""
    assert widget.timer.isActive()


def test_close_stops_the_simulation_timer(widget) -> None:
    """Closing the widget must stop the timer it owns.

    Before UD #9474 this failed: ``close()`` only hid the widget, so the
    timer kept firing ``_on_timer`` for the rest of the process.
    """
    assert widget.timer.isActive()

    widget.close()

    assert not widget.timer.isActive(), (
        "close() left the simulation timer running; the widget will keep "
        "stepping MuJoCo during unrelated later tests"
    )
    assert widget.running is False


def test_stop_simulation_is_idempotent(widget) -> None:
    """Teardown runs from several paths, so it must tolerate repetition."""
    widget.stop_simulation()
    widget.stop_simulation()

    assert not widget.timer.isActive()
    assert widget.running is False


def test_close_is_idempotent(widget) -> None:
    widget.close()
    widget.close()

    assert not widget.timer.isActive()


class TestGetCv2Contract:
    """``get_cv2`` promises "the cv2 module, or None". It must keep that."""

    @staticmethod
    @pytest.fixture(autouse=True)
    def _reset_cv2_cache():
        original = dict(sim_rendering_mixin._cv2_state)
        sim_rendering_mixin._cv2_state.update({"lib": None, "invalid": False})
        try:
            yield
        finally:
            sim_rendering_mixin._cv2_state.clear()
            sim_rendering_mixin._cv2_state.update(original)

    @staticmethod
    def test_returns_none_when_opencv_is_half_initialised(monkeypatch) -> None:
        """A partially initialised OpenCV raises AttributeError, not ImportError.

        This is the exact CI failure: cv2's own ``bootstrap()`` reached
        ``cv2.typing`` before ``cv2.mat_wrapper`` was bound and raised
        ``AttributeError``. Before UD #9474 ``get_cv2`` guarded only
        ``ImportError``, so the AttributeError escaped through the Qt slot and
        aborted the interpreter.
        """
        real_import = __import__

        def broken_cv2_import(name, *args, **kwargs):
            if name == "cv2":
                raise AttributeError(
                    "partially initialized module 'cv2' has no attribute "
                    "'mat_wrapper' (most likely due to a circular import)"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", broken_cv2_import)

        assert sim_rendering_mixin.get_cv2() is None

    @staticmethod
    def test_returns_none_when_opencv_is_absent(monkeypatch) -> None:
        real_import = __import__

        def missing_cv2_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", missing_cv2_import)

        assert sim_rendering_mixin.get_cv2() is None

    @staticmethod
    def test_failure_is_remembered_so_the_import_is_attempted_once(
        monkeypatch,
    ) -> None:
        """A broken OpenCV must not be re-imported on every rendered frame."""
        real_import = __import__
        attempts = []

        def broken_cv2_import(name, *args, **kwargs):
            if name == "cv2":
                attempts.append(name)
                raise AttributeError("half-initialised")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", broken_cv2_import)

        assert sim_rendering_mixin.get_cv2() is None
        assert sim_rendering_mixin.get_cv2() is None
        assert len(attempts) == 1


# ---------------------------------------------------------------------------
# UD #9492: timer-body behaviour (stepping / control / recording / rendering)
# ---------------------------------------------------------------------------


class FakeModel:
    """Minimal ``MjModel`` stand-in exposing only what the timer path reads."""

    def __init__(self, nu: int = 2, timestep: float = 1.0 / 120.0) -> None:
        self.nu = nu
        self.opt = SimpleNamespace(timestep=timestep)


class FakeData:
    """Minimal ``MjData`` stand-in for the timer and overlay paths."""

    def __init__(self, nu: int = 2) -> None:
        self.time = 0.25
        self.qvel = np.zeros(6)
        self.qpos = np.zeros(6)
        self.ctrl = np.zeros(nu)
        self.xpos = np.array([[0.0, 0.0, 0.0], [0.5, 0.25, 0.75]])
        self.xquat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
        self.xmat = np.tile(np.eye(3).reshape(9), (2, 1))


class FakeControlSystem:
    """Records the time updates and velocity queries the widget performs."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.velocity_queries: list[np.ndarray | None] = []
        self.torque = np.array([0.5, -0.25])

    def update_time(self, time: float) -> None:
        self.times.append(time)

    def compute_control_vector(self, velocities: np.ndarray | None) -> np.ndarray:
        self.velocity_queries.append(
            None if velocities is None else np.array(velocities)
        )
        return self.torque


class FakeTelemetry:
    """Records every stepped simulation state."""

    def __init__(self) -> None:
        self.steps: list[object] = []

    def record_step(self, data: object) -> None:
        self.steps.append(data)


class FakeAnalyzer:
    """Records ``extract_full_state`` keyword calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, bool]] = []

    def extract_full_state(
        self,
        *,
        selected_actuator_name: str | None = None,
        compute_advanced_metrics: bool = False,
    ) -> dict[str, str | None]:
        self.calls.append((selected_actuator_name, compute_advanced_metrics))
        return {"actuator": selected_actuator_name}


class FakeRecorder:
    """Recorder fake carrying the attributes the timer path inspects."""

    def __init__(self, analysis_config: dict | None = None) -> None:
        self.analysis_config = analysis_config
        self.is_recording = False
        self.frames: list[dict] = []

    def record_frame(self, bio_data: dict) -> None:
        self.frames.append(bio_data)


class FakeCV2:
    """``putText``/``line``/``circle`` recorder standing in for OpenCV."""

    FONT_HERSHEY_SIMPLEX = "FONT_HERSHEY_SIMPLEX"

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def putText(self, img, msg, org, font_face, font_scale, color, thickness):
        self.calls.append(("text", msg, org, color))

    def line(self, img, p1, p2, color, thickness):
        self.calls.append(("line", p1, p2, color))

    def circle(self, img, center, radius, color, thickness):
        self.calls.append(("circle", center, color))


class FakeScrewAnalyzer:
    """Deterministic screw-kinematics stand-in for the ISA overlay."""

    def compute_twist(self, qpos, qvel, body_id):
        return ("twist", body_id)

    def compute_screw_axis(self, twist):
        return SimpleNamespace(pitch=0.125)

    def visualize_screw_axis(self, screw, length=0.5):
        return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


@pytest.fixture
def timer_widget(widget):
    """A widget wired to fakes so the timer body can run hermetically."""
    widget.timer.stop()
    widget.meshcat_adapter = None
    widget.operating_mode = "dynamic"
    widget.control_system = None
    widget.control_vector = None
    widget.telemetry = FakeTelemetry()
    widget.enable_live_analysis = False
    widget.show_induced_vectors = False
    widget.model = FakeModel()
    widget.data = FakeData()
    return widget


@pytest.fixture
def render_log(timer_widget, monkeypatch):
    """Stub every render-path collaborator; return ``(widget, call log)``."""
    widget = timer_widget
    log: list[str] = []
    monkeypatch.setattr(widget, "_render_once", lambda: log.append("render"))
    monkeypatch.setattr(
        widget, "_enforce_interactive_constraints", lambda: log.append("enforce")
    )
    monkeypatch.setattr(widget, "compute_ellipsoids", lambda: log.append("ellipsoids"))
    monkeypatch.setattr(
        widget, "_record_club_trajectory_point", lambda: log.append("trajectory")
    )
    monkeypatch.setattr(
        widget, "_update_swing_plane_overlays", lambda: log.append("overlays")
    )
    return widget, log


@pytest.fixture
def mj_steps(timer_widget, monkeypatch):
    """Record every ``mj_step`` call the timer path makes."""
    steps: list[object] = []
    monkeypatch.setattr(
        sim_widget.mujoco, "mj_step", lambda model, data: steps.append(data)
    )
    return steps


@pytest.fixture
def overlay_widget(widget, monkeypatch):
    """A widget with a fake selected body and a fake world-to-screen map."""
    widget.timer.stop()
    widget.meshcat_adapter = None
    widget.model = FakeModel()
    widget.data = FakeData()
    widget.manipulator = SimpleNamespace(selected_body_id=1)
    widget.show_live_euler = False
    widget.show_live_quat = False
    widget.show_live_screw = False
    monkeypatch.setattr(
        widget, "_world_to_screen", lambda pos: (int(pos[0]) + 10, int(pos[1]) + 20)
    )
    return widget


@pytest.fixture
def fake_cv2(overlay_widget, monkeypatch):
    """Route the overlay path's OpenCV calls into a recorder."""
    fake = FakeCV2()
    monkeypatch.setattr(sim_rendering_mixin, "get_cv2", lambda: fake)
    return fake


class TestOnTimerWithoutModel:
    """A timer tick before any model load must be a no-op."""

    def test_returns_without_stepping_or_rendering(self, widget, monkeypatch) -> None:
        widget.timer.stop()
        log: list[str] = []
        monkeypatch.setattr(widget, "_render_once", lambda: log.append("render"))
        monkeypatch.setattr(
            sim_widget.mujoco, "mj_step", lambda model, data: log.append("step")
        )
        widget._on_timer()
        assert log == []


class TestOnTimerKinematicMode:
    """Kinematic mode only enforces constraints and renders."""

    def test_renders_without_stepping(self, render_log, mj_steps) -> None:
        widget, log = render_log
        widget.operating_mode = "kinematic"
        widget._on_timer()
        assert log == ["enforce", "render"]
        assert mj_steps == []

    def test_does_not_touch_the_control_system(self, render_log, mj_steps) -> None:
        widget, log = render_log
        widget.operating_mode = "kinematic"
        control_system = FakeControlSystem()
        widget.control_system = control_system
        widget._on_timer()
        assert log == ["enforce", "render"]
        assert control_system.times == []


class TestOnTimerDynamicMode:
    """Dynamic mode steps physics, applies control, records and renders."""

    def test_steps_frame_and_applies_control_system(self, render_log, mj_steps) -> None:
        widget, log = render_log
        control_system = FakeControlSystem()
        widget.control_system = control_system
        widget._on_timer()
        # fps 30 with a 1/120 s model timestep -> 4 physics steps per frame.
        assert len(mj_steps) == 4
        assert all(data is widget.data for data in mj_steps)
        assert np.array_equal(widget.data.ctrl, control_system.torque)
        assert control_system.times == [0.25] * 4
        assert len(control_system.velocity_queries) == 4
        assert np.array_equal(control_system.velocity_queries[0], widget.data.qvel[:2])
        assert [data is widget.data for data in widget.telemetry.steps] == [True] * 4
        assert log[-1] == "render"

    def test_falls_back_to_static_control_vector(self, render_log, mj_steps) -> None:
        widget, _ = render_log
        widget.control_vector = np.array([0.1, 0.2])
        widget._on_timer()
        assert np.array_equal(widget.data.ctrl, [0.1, 0.2])
        assert len(mj_steps) == 4

    def test_memory_error_pauses_simulation(
        self, render_log, mj_steps, monkeypatch
    ) -> None:
        widget, _ = render_log

        def raise_memory_error() -> None:
            raise MemoryError("renderer out of memory")

        monkeypatch.setattr(widget, "_render_once", raise_memory_error)
        widget._on_timer()
        assert widget.running is False
        assert "Out of memory" in widget.label.text()

    def test_invokes_the_recording_hook(self, render_log, mj_steps) -> None:
        widget, _ = render_log
        widget.analyzer = FakeAnalyzer()
        widget.recorder = FakeRecorder()
        widget.recorder.is_recording = True
        widget._on_timer()
        assert widget.recorder.frames == [{"actuator": None}]


class TestApplyControl:
    """The per-step control write extracted from the timer body."""

    def test_prefers_the_control_system(self, timer_widget) -> None:
        widget = timer_widget
        control_system = FakeControlSystem()
        widget.control_system = control_system
        widget._apply_control()
        assert np.array_equal(widget.data.ctrl, control_system.torque)
        assert control_system.times == [0.25]
        assert np.array_equal(control_system.velocity_queries[0], widget.data.qvel[:2])

    def test_falls_back_to_the_static_vector(self, timer_widget) -> None:
        widget = timer_widget
        widget.control_vector = np.array([0.3, -0.4])
        widget._apply_control()
        assert np.array_equal(widget.data.ctrl, [0.3, -0.4])

    def test_leaves_ctrl_untouched_without_control_sources(self, timer_widget) -> None:
        widget = timer_widget
        widget._apply_control()
        assert np.array_equal(widget.data.ctrl, np.zeros(2))


class TestUpdateRecordingAnalysis:
    """The per-frame biomechanical analysis/recording hook."""

    def test_records_frame_while_recording(self, timer_widget) -> None:
        widget = timer_widget
        analyzer = FakeAnalyzer()
        recorder = FakeRecorder()
        recorder.is_recording = True
        widget.analyzer = analyzer
        widget.recorder = recorder
        widget._update_recording_analysis()
        assert analyzer.calls == [(None, False)]
        assert recorder.frames == [{"actuator": None}]
        assert widget.latest_bio_data == {"actuator": None}

    def test_computes_without_recording_when_enabled(self, timer_widget) -> None:
        widget = timer_widget
        widget.enable_live_analysis = True
        analyzer = FakeAnalyzer()
        recorder = FakeRecorder()
        widget.analyzer = analyzer
        widget.recorder = recorder
        widget._update_recording_analysis()
        assert analyzer.calls == [(None, True)]
        assert recorder.frames == []
        assert widget.latest_bio_data == {"actuator": None}

    def test_is_idle_without_analyzer_or_request(self, timer_widget) -> None:
        widget = timer_widget
        analyzer = FakeAnalyzer()
        widget.analyzer = analyzer
        widget.recorder = FakeRecorder()
        widget._update_recording_analysis()
        assert analyzer.calls == []
        assert widget.latest_bio_data is None


class TestResolveLiveAnalysis:
    """The recorder-config and toggle resolution extracted from the timer."""

    def test_config_requests_analysis_and_selects_actuator(self, timer_widget) -> None:
        widget = timer_widget
        widget.recorder = FakeRecorder(
            analysis_config={
                "ztcf": True,
                "induced_accel_sources": ["gravity", "psoas"],
            }
        )
        assert widget._resolve_live_analysis() == (True, "psoas")

    def test_config_with_only_standard_sources_selects_nothing(
        self, timer_widget
    ) -> None:
        widget = timer_widget
        widget.recorder = FakeRecorder(
            analysis_config={"induced_accel_sources": ["gravity", "actuator"]}
        )
        assert widget._resolve_live_analysis() == (True, None)

    def test_live_analysis_flag_alone_requests_analysis(self, timer_widget) -> None:
        widget = timer_widget
        widget.recorder = FakeRecorder()
        widget.enable_live_analysis = True
        assert widget._resolve_live_analysis() == (True, None)

    def test_induced_vector_source_fallback(self, timer_widget) -> None:
        widget = timer_widget
        widget.recorder = FakeRecorder()
        widget.show_induced_vectors = True
        widget.induced_vector_source = "velocity"
        assert widget._resolve_live_analysis() == (False, "velocity")

    def test_induced_vector_source_excluded_from_fallback(self, timer_widget) -> None:
        widget = timer_widget
        widget.recorder = FakeRecorder()
        widget.show_induced_vectors = True
        widget.induced_vector_source = "gravity"
        assert widget._resolve_live_analysis() == (False, None)


class TestEulerFromMatrix:
    """The rotation-matrix to xyz-Euler helper used by the euler overlay."""

    def test_identity_is_zero(self) -> None:
        assert SimRenderingMixin._euler_from_matrix(np.eye(3)) == (0.0, 0.0, 0.0)

    def test_rotation_about_z(self) -> None:
        r_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        x_e, y_e, z_e = SimRenderingMixin._euler_from_matrix(r_z)
        assert np.isclose(np.rad2deg(x_e), 0.0)
        assert np.isclose(np.rad2deg(y_e), 0.0)
        assert np.isclose(np.rad2deg(z_e), 90.0)

    def test_singular_branch(self) -> None:
        # A 90 deg rotation about Y sends sqrt(m00^2 + m10^2) to zero and must
        # take the gimbal-locked branch without producing NaNs.
        r_y = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        x_e, y_e, z_e = SimRenderingMixin._euler_from_matrix(r_y)
        assert np.isclose(np.rad2deg(x_e), 0.0)
        assert np.isclose(np.rad2deg(y_e), 90.0)
        assert np.isclose(np.rad2deg(z_e), 0.0)


# ---------------------------------------------------------------------------
# UD #9492: live-kinematics overlay behaviour
# ---------------------------------------------------------------------------


class TestAddLiveKinematicsOverlays:
    """Guard and drawing behaviour of the live-kinematics overlay pass."""

    def test_rejects_missing_rgb(self, overlay_widget) -> None:
        with pytest.raises(ValueError):
            overlay_widget._add_live_kinematics_overlays(None)

    def test_returns_input_without_model(self, widget) -> None:
        widget.timer.stop()
        rgb = np.zeros((4, 4, 3), np.uint8)
        assert widget._add_live_kinematics_overlays(rgb) is rgb

    def test_returns_input_without_opencv(self, overlay_widget, monkeypatch) -> None:
        monkeypatch.setattr(sim_rendering_mixin, "get_cv2", lambda: None)
        rgb = np.zeros((4, 4, 3), np.uint8)
        assert overlay_widget._add_live_kinematics_overlays(rgb) is rgb

    def test_returns_input_without_selected_body(self, overlay_widget) -> None:
        overlay_widget.manipulator = None
        rgb = np.zeros((4, 4, 3), np.uint8)
        assert overlay_widget._add_live_kinematics_overlays(rgb) is rgb

    def test_returns_negative_body_id_unchanged(self, overlay_widget) -> None:
        overlay_widget.manipulator = SimpleNamespace(selected_body_id=-1)
        rgb = np.zeros((4, 4, 3), np.uint8)
        assert overlay_widget._add_live_kinematics_overlays(rgb) is rgb

    def test_returns_copy_when_body_is_off_screen(
        self, overlay_widget, monkeypatch
    ) -> None:
        monkeypatch.setattr(overlay_widget, "_world_to_screen", lambda pos: None)
        rgb = np.zeros((4, 4, 3), np.uint8)
        result = overlay_widget._add_live_kinematics_overlays(rgb)
        assert result is not rgb
        assert np.array_equal(result, rgb)

    def test_no_toggles_enabled_leaves_the_frame_unchanged(
        self, overlay_widget, fake_cv2
    ) -> None:
        rgb = np.zeros((8, 8, 3), np.uint8)
        result = overlay_widget._add_live_kinematics_overlays(rgb)
        assert result is not rgb
        assert np.array_equal(result, rgb)
        assert fake_cv2.calls == []

    def test_draws_euler_overlay_for_selected_body(
        self, overlay_widget, fake_cv2
    ) -> None:
        overlay_widget.show_live_euler = True
        overlay_widget._add_live_kinematics_overlays(np.zeros((8, 8, 3), np.uint8))
        texts = [call for call in fake_cv2.calls if call[0] == "text"]
        assert len(texts) == 1
        _, msg, org, color = texts[0]
        assert msg.startswith("Euler (xyz)")
        assert org == (20, 50)  # x + 10, y + 30
        assert color == (0, 255, 255)

    def test_quat_overlay_offsets_below_euler(self, overlay_widget, fake_cv2) -> None:
        overlay_widget.show_live_euler = True
        overlay_widget.show_live_quat = True
        overlay_widget._add_live_kinematics_overlays(np.zeros((8, 8, 3), np.uint8))
        texts = [call for call in fake_cv2.calls if call[0] == "text"]
        assert texts[0][1].startswith("Euler (xyz)")
        assert texts[0][2] == (20, 50)
        assert texts[1][1].startswith("Quat (w,x,y,z)")
        assert texts[1][2] == (20, 65)  # one 15 px row below the euler text
        assert texts[1][3] == (255, 255, 0)
        assert texts == [
            ("text", "Euler (xyz): [0.0, -0.0, 0.0] deg", (20, 50), (0, 255, 255)),
            (
                "text",
                "Quat (w,x,y,z): [1.00, 0.00, 0.00, 0.00]",
                (20, 65),
                (255, 255, 0),
            ),
        ]

    def test_screw_overlay_draws_isa_line_and_pitch(
        self, overlay_widget, fake_cv2, monkeypatch
    ) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
            screw_kinematics,
        )

        monkeypatch.setattr(
            screw_kinematics,
            "ScrewKinematicsAnalyzer",
            lambda model: FakeScrewAnalyzer(),
        )
        overlay_widget.show_live_screw = True
        overlay_widget._add_live_kinematics_overlays(np.zeros((8, 8, 3), np.uint8))
        texts = [call for call in fake_cv2.calls if call[0] == "text"]
        lines = [call for call in fake_cv2.calls if call[0] == "line"]
        circles = [call for call in fake_cv2.calls if call[0] == "circle"]
        # start (0, 0, 0) -> screen (10, 20); end (1, 1, 1) -> screen (11, 21).
        assert lines == [("line", (10, 20), (11, 21), (255, 0, 255))]
        assert circles == [("circle", (11, 21), (255, 0, 255))]
        assert texts == [("text", "ISA Pitch: 0.125 m/rad", (20, 50), (255, 0, 255))]
