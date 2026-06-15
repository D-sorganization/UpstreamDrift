"""Tests for the headless AnalysisOrchestrator (issue #7446).

Verifies that:
- the orchestrator computes structured, JSON-serializable PlotData that
  matches what the PyQt6 dashboard renders (golden-value checks),
- counterfactual computation works headlessly,
- importing the orchestrator pulls in no Qt or matplotlib modules.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.analysis.orchestrator import (
    AnalysisOrchestrator,
    get_plot_data,
)
from src.shared.python.analysis.plot_data import (
    CounterfactualResult,
    PlotData,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

N_FRAMES = 20
N_JOINTS = 3


class StubRecorder:
    """Deterministic Qt-free recorder stub mimicking GenericPhysicsRecorder."""

    def __init__(self, with_data: bool = True) -> None:
        self.post_hoc_calls = 0
        if not with_data:
            self.times = np.array([])
            self.data: dict[str, np.ndarray] = {}
            self.counterfactuals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            self.induced: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            return
        self.times = np.linspace(0.0, 1.0, N_FRAMES)
        base = np.arange(N_FRAMES, dtype=float)[:, None]
        cols = np.arange(1, N_JOINTS + 1, dtype=float)[None, :]
        self.data = {
            "joint_positions": 0.01 * base * cols,
            "joint_velocities": 0.10 * base * cols,
            "joint_torques": 0.50 * base * cols,
            "actuator_powers": 0.25 * base * cols,
            "kinetic_energy": 2.0 * base[:, 0],
            "potential_energy": 1.0 * base[:, 0],
            "total_energy": 3.0 * base[:, 0],
            "club_head_speed": 0.5 * base[:, 0],
            "club_head_position": 0.1 * base * np.array([[1.0, 2.0, 3.0]]),
            "angular_momentum": 0.2 * base * np.array([[1.0, 0.5, 0.25]]),
            "cop_position": 0.05 * base * np.array([[1.0, -1.0, 0.0]]),
            "com_position": 0.04 * base * np.array([[1.0, 1.0, 1.0]]),
            "ground_forces": 10.0 * base * np.array([[1.0, -0.5, 2.0]]),
            "joint_accelerations": 0.2 * base * cols,
        }
        cf = np.ones((N_FRAMES, N_JOINTS)) * 0.5
        self.counterfactuals = {"ztcf": (self.times, cf), "zvcf": (self.times, 2 * cf)}
        self.induced = {
            "gravity": (self.times, 3 * cf),
            "control": (self.times, 4 * cf),
        }

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        if field_name not in self.data:
            return np.array([]), np.array([])
        return self.times, self.data[field_name]

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.counterfactuals.get(cf_name, (np.array([]), np.array([])))

    def get_induced_acceleration_series(
        self, source_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.induced.get(source_name, (np.array([]), np.array([])))

    def compute_analysis_post_hoc(self) -> None:
        self.post_hoc_calls += 1
        cf = np.full((len(self.times), N_JOINTS), 9.0) if len(self.times) else None
        if cf is not None:
            self.counterfactuals["ztcf"] = (self.times, cf)


@pytest.fixture
def recorder() -> StubRecorder:
    return StubRecorder()


@pytest.fixture
def orchestrator(recorder: StubRecorder) -> AnalysisOrchestrator:
    return AnalysisOrchestrator(recorder, joint_names=["Hip", "Shoulder", "Wrist"])


# ----------------------------------------------------------------------
# Construction / DbC
# ----------------------------------------------------------------------


def test_requires_recorder() -> None:
    with pytest.raises(ValueError, match="recorder"):
        AnalysisOrchestrator(None)  # type: ignore[arg-type]


def test_rejects_recorder_without_interface() -> None:
    with pytest.raises(TypeError, match="get_time_series"):
        AnalysisOrchestrator(object())  # type: ignore[arg-type]


def test_unknown_plot_type_raises(orchestrator: AnalysisOrchestrator) -> None:
    with pytest.raises(ValueError, match="Unknown plot type"):
        orchestrator.get_plot_data("nope")


def test_non_string_plot_type_raises(orchestrator: AnalysisOrchestrator) -> None:
    with pytest.raises(TypeError, match="plot_type"):
        orchestrator.get_plot_data(42)  # type: ignore[arg-type]


def test_registry_and_labels_consistent() -> None:
    types = AnalysisOrchestrator.available_plot_types()
    assert types == sorted(AnalysisOrchestrator.PLOT_TYPES)
    # every dashboard label is routed through a registered headless plot type
    assert set(AnalysisOrchestrator.DASHBOARD_LABEL_TO_PLOT_TYPE) == set(
        AnalysisOrchestrator.DASHBOARD_PLOT_LABELS
    )
    for label, pt in AnalysisOrchestrator.DASHBOARD_LABEL_TO_PLOT_TYPE.items():
        assert label in AnalysisOrchestrator.DASHBOARD_PLOT_LABELS
        assert pt in AnalysisOrchestrator.PLOT_TYPES
    # dashboard catalogue unchanged (20 entries, GUI parity)
    assert len(AnalysisOrchestrator.DASHBOARD_PLOT_LABELS) == 20


# ----------------------------------------------------------------------
# Golden-value checks against recorded data
# ----------------------------------------------------------------------


def test_joint_angles_golden(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    data = orchestrator.get_plot_data("joint_angles")
    assert isinstance(data, PlotData)
    assert data.y_label == "Joint Angle (degrees)"
    assert [s.name for s in data.series] == ["Hip", "Shoulder", "Wrist"]
    expected = np.rad2deg(recorder.data["joint_positions"][:, 1])
    np.testing.assert_allclose(data.series[1].y, expected)
    np.testing.assert_allclose(data.series[1].x, recorder.times)
    assert data.series[1].units == "deg"


def test_joint_velocities_units(orchestrator: AnalysisOrchestrator) -> None:
    data = orchestrator.get_plot_data("joint_velocities")
    assert data.series[0].units == "deg/s"
    assert data.y_label == "Angular Velocity (deg/s)"


def test_joint_torques_raw_si(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    data = orchestrator.get_plot_data("joint_torques")
    np.testing.assert_allclose(data.series[2].y, recorder.data["joint_torques"][:, 2])
    assert data.series[2].units == "Nm"


def test_energies_three_series(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    data = orchestrator.get_plot_data("energies")
    assert [s.name for s in data.series] == [
        "Kinetic Energy",
        "Potential Energy",
        "Total Energy",
    ]
    np.testing.assert_allclose(data.series[2].y, recorder.data["total_energy"])


def test_club_head_speed_mph(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    data = orchestrator.get_plot_data("club_head_speed")
    expected = recorder.data["club_head_speed"] * 2.23694
    np.testing.assert_allclose(data.series[0].y, expected)
    assert data.metadata["peak_speed_mph"] == pytest.approx(expected[-1])
    assert data.metadata["peak_time_s"] == pytest.approx(1.0)


def test_angular_momentum_magnitude(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    data = orchestrator.get_plot_data("angular_momentum")
    assert [s.name for s in data.series] == ["Lx", "Ly", "Lz", "Magnitude"]
    am = recorder.data["angular_momentum"]
    np.testing.assert_allclose(data.series[3].y, np.linalg.norm(am, axis=1))


def test_joint_power_curves_is_tau_times_omega(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    data = orchestrator.get_plot_data("joint_power_curves")
    expected = (
        recorder.data["joint_torques"][:, 0] * recorder.data["joint_velocities"][:, 0]
    )
    np.testing.assert_allclose(data.series[0].y, expected)
    assert data.series[0].units == "W"


def test_impulse_accumulation_matches_trapezoid(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    data = orchestrator.get_plot_data("impulse_accumulation")
    tau = recorder.data["joint_torques"][:, 0]
    dt = float(np.mean(np.diff(recorder.times)))
    expected = np.concatenate(([0.0], np.cumsum((tau[1:] + tau[:-1]) * 0.5 * dt)))
    np.testing.assert_allclose(data.series[0].y, expected)
    assert data.series[0].y[0] == 0.0


def test_phase_diagram_joint0(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    data = orchestrator.get_plot_data("phase_diagram")
    np.testing.assert_allclose(
        data.series[0].x, np.rad2deg(recorder.data["joint_positions"][:, 0])
    )
    np.testing.assert_allclose(
        data.series[0].y, np.rad2deg(recorder.data["joint_velocities"][:, 0])
    )
    assert data.metadata["joint_idx"] == 0


def test_trajectory_plots(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    cop = orchestrator.get_plot_data("cop_trajectory")
    np.testing.assert_allclose(cop.series[0].x, recorder.data["cop_position"][:, 0])

    stab = orchestrator.get_plot_data("stability_diagram")
    assert [s.name for s in stab.series] == ["CoP", "CoM (Proj)"]

    club = orchestrator.get_plot_data("club_head_trajectory_3d")
    assert club.series[0].z is not None
    np.testing.assert_allclose(
        club.series[0].z, recorder.data["club_head_position"][:, 2]
    )


def test_remaining_dashboard_plot_types_are_structured(
    orchestrator: AnalysisOrchestrator,
) -> None:
    poincare = orchestrator.get_plot_data("poincare_map_3d")
    assert poincare.plot_type == "poincare_map_3d"
    assert poincare.metadata["z_label"] == "Acceleration (deg/s^2)"

    lyap = orchestrator.get_plot_data("lyapunov_exponent")
    assert lyap.plot_type == "lyapunov_exponent"
    assert "estimated_mle" in lyap.metadata or lyap.is_empty

    recurrence = orchestrator.get_plot_data("recurrence_plot")
    assert recurrence.metadata["chart"] == "heatmap"
    assert "matrix" in recurrence.series[0].metadata

    grf = orchestrator.get_plot_data("grf_butterfly_diagram")
    assert grf.series[0].name == "CoP Path"
    assert grf.metadata["vectors"]

    sequence = orchestrator.get_plot_data("kinematic_sequence_bars")
    assert sequence.metadata["indices"] == {
        "proximal": 0,
        "mid_proximal": 1,
        "mid_distal": 2,
    }

    summary = orchestrator.get_plot_data("summary_dashboard")
    assert summary.metadata["chart"] == "dashboard"
    assert len(summary.metadata["panels"]) == 6


def test_module_level_convenience(recorder: StubRecorder) -> None:
    data = get_plot_data("energies", recorder)
    assert isinstance(data, PlotData)
    assert data.plot_type == "energies"


# ----------------------------------------------------------------------
# JSON serializability and empty-data behavior
# ----------------------------------------------------------------------


def test_all_plot_types_json_serializable(
    orchestrator: AnalysisOrchestrator,
) -> None:
    for plot_type in AnalysisOrchestrator.available_plot_types():
        data = orchestrator.get_plot_data(plot_type)
        assert data.plot_type == plot_type
        payload = json.dumps(data.to_dict())
        decoded = json.loads(payload)
        assert decoded["plot_type"] == plot_type


def test_empty_recorder_returns_empty_plotdata() -> None:
    orch = AnalysisOrchestrator(StubRecorder(with_data=False))
    for plot_type in AnalysisOrchestrator.available_plot_types():
        data = orch.get_plot_data(plot_type)
        assert data.is_empty, plot_type
        assert "message" in data.metadata, plot_type
        json.dumps(data.to_dict())  # still serializable


# ----------------------------------------------------------------------
# Counterfactuals
# ----------------------------------------------------------------------


def test_counterfactual_kind_validation(
    orchestrator: AnalysisOrchestrator,
) -> None:
    with pytest.raises(ValueError, match="Unknown counterfactual kind"):
        orchestrator.compute_counterfactual("warp_drive")
    with pytest.raises(TypeError, match="kind"):
        orchestrator.compute_counterfactual(7)  # type: ignore[arg-type]


def test_counterfactual_ztcf_golden(
    orchestrator: AnalysisOrchestrator, recorder: StubRecorder
) -> None:
    result = orchestrator.compute_counterfactual("ztcf")
    assert isinstance(result, CounterfactualResult)
    assert result.metadata["n_frames"] == N_FRAMES
    np.testing.assert_allclose(result.values, np.full((N_FRAMES, N_JOINTS), 0.5))
    assert recorder.post_hoc_calls == 0  # data present -> no recompute
    json.dumps(result.to_dict())


def test_counterfactual_induced_sources(
    orchestrator: AnalysisOrchestrator,
) -> None:
    grav = orchestrator.compute_counterfactual("gravity", run_post_hoc=False)
    np.testing.assert_allclose(grav.values, np.full((N_FRAMES, N_JOINTS), 1.5))
    ctrl = orchestrator.compute_counterfactual("control", run_post_hoc=False)
    np.testing.assert_allclose(ctrl.values, np.full((N_FRAMES, N_JOINTS), 2.0))


def test_counterfactual_triggers_post_hoc_when_missing(
    recorder: StubRecorder,
) -> None:
    recorder.counterfactuals.pop("ztcf")
    orch = AnalysisOrchestrator(recorder)
    result = orch.compute_counterfactual("ztcf")
    assert recorder.post_hoc_calls == 1
    np.testing.assert_allclose(result.values, np.full((N_FRAMES, N_JOINTS), 9.0))


# ----------------------------------------------------------------------
# Helpers extracted from the PyQt6 window (DRY)
# ----------------------------------------------------------------------


def test_kinematic_sequence_indices(orchestrator: AnalysisOrchestrator) -> None:
    indices = orchestrator.derive_kinematic_sequence_indices()
    assert indices == {"proximal": 0, "mid_proximal": 1, "mid_distal": 2}


def test_kinematic_sequence_indices_empty() -> None:
    orch = AnalysisOrchestrator(StubRecorder(with_data=False))
    assert orch.derive_kinematic_sequence_indices() == {}


def test_recurrence_matrix_requires_data() -> None:
    orch = AnalysisOrchestrator(StubRecorder(with_data=False))
    with pytest.raises(ValueError, match="No data available"):
        orch.compute_recurrence_matrix()


def test_recurrence_matrix_shape(orchestrator: AnalysisOrchestrator) -> None:
    rm = orchestrator.compute_recurrence_matrix()
    assert rm.shape[0] == rm.shape[1] > 0


# ----------------------------------------------------------------------
# Headless import purity
# ----------------------------------------------------------------------


def test_orchestrator_import_is_qt_and_matplotlib_free() -> None:
    """Importing the orchestrator must not import Qt or matplotlib."""
    repo_root = Path(__file__).resolve().parents[3]
    code = (
        "import sys\n"
        "import src.shared.python.analysis.orchestrator\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('PyQt6', 'PySide6', 'matplotlib')]\n"
        "assert not bad, f'GUI modules imported: {bad}'\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "src")])
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
