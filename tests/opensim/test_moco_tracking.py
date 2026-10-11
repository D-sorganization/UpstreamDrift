"""Tests for OpenSim Moco dynamic tracking configuration and TRC sanitization (OS-4).

Exercises pure-Python contracts without requiring OpenSim bindings, plus
OpenSim-dependent tests when available.
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
    sanitize_trc_for_horizon,
    _is_finite_float,
    build_moco_study,
)


@pytest.mark.unit
def test_moco_tracking_config_defaults() -> None:
    """Config has sensible defaults for swing tracking."""
    cfg = MocoTrackingConfig()
    cfg.validate()
    assert cfg.horizon_s == 0.85
    assert cfg.t_start_s == 0.0
    assert cfg.duration_s == 0.85
    assert cfg.mesh_interval_s == 0.02
    assert cfg.num_mesh_intervals == 42 or cfg.num_mesh_intervals == 43
    assert cfg.effort_weight == 1e-4
    assert cfg.marker_weight == 1.0


@pytest.mark.unit
def test_moco_tracking_config_validation_errors() -> None:
    """Config enforces DbC parameter preconditions."""
    with pytest.raises(ValueError, match="t_start_s must be >= 0"):
        MocoTrackingConfig(t_start_s=-0.1).validate()

    with pytest.raises(ValueError, match="horizon_s .* must be > t_start_s"):
        MocoTrackingConfig(horizon_s=0.5, t_start_s=0.5).validate()

    with pytest.raises(ValueError, match="mesh_interval_s must be > 0"):
        MocoTrackingConfig(mesh_interval_s=0.0).validate()

    with pytest.raises(ValueError, match="Goal weights must be non-negative"):
        MocoTrackingConfig(effort_weight=-1.0).validate()

    with pytest.raises(ValueError, match="optim_max_iterations must be > 0"):
        MocoTrackingConfig(optim_max_iterations=0).validate()

    with pytest.raises(ValueError, match="Tolerances must be > 0"):
        MocoTrackingConfig(optim_convergence_tolerance=-1e-3).validate()


@pytest.mark.unit
def test_is_finite_float() -> None:
    """Test float finiteness checker."""
    assert _is_finite_float("1.234")
    assert _is_finite_float("-0.567")
    assert _is_finite_float("0.0")
    assert not _is_finite_float("nan")
    assert not _is_finite_float("NaN")
    assert not _is_finite_float("inf")
    assert not _is_finite_float("-inf")
    assert not _is_finite_float("abc")


@pytest.mark.unit
def test_sanitize_trc_for_horizon_prunes_nan_markers() -> None:
    """Test that markers with NaNs inside the observation window are pruned."""
    lines = [
        "PathFileType\t4\t(X/Y/Z)\ttest.trc\n",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        "100.0\t100.0\t3\t3\tm\t100.0\t1\t3\n",
        "Frame#\tTime\tMarkerGood\t\tMarkerDropout\t\tMarkerLateNaN\t\n",
        "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3\n",
        "\n",
        "1\t0.00\t0.1\t0.2\t0.3\tnan\tnan\tnan\t1.0\t1.1\t1.2\n",
        "2\t0.01\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\t1.0\t1.1\t1.2\n",
        "3\t0.02\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\tnan\t1.1\t1.2\n",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "in.trc"
        out_path = Path(tmpdir) / "out.trc"
        in_path.write_text("".join(lines), encoding="utf-8")

        retained = sanitize_trc_for_horizon(
            in_path, out_path, t_start=0.0, t_end=0.015, max_missing_ratio=0.0
        )
        assert retained == ["MarkerGood", "MarkerLateNaN"]
        assert out_path.exists()

        out_lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(out_lines) >= 8
        assert "MarkerGood" in out_lines[3]
        assert "MarkerLateNaN" in out_lines[3]
        assert "MarkerDropout" not in out_lines[3]


@pytest.mark.unit
@pytest.mark.parametrize("scheme", ["central", "forward"])
def test_finite_difference_scheme_accepts_typed_values(scheme: str) -> None:
    """The typed config permits only the finite-difference policy values."""
    cfg = MocoTrackingConfig(optim_finite_difference_scheme=scheme)
    cfg.validate()
    assert cfg.optim_finite_difference_scheme == scheme


@pytest.mark.unit
@pytest.mark.parametrize("scheme", ["__unsupported__", "", None, 1, True])
def test_finite_difference_scheme_rejects_invalid_values(scheme: object) -> None:
    """Invalid policy values fail validation instead of relying on SWIG."""
    cfg = MocoTrackingConfig(optim_finite_difference_scheme=scheme)
    with pytest.raises((TypeError, ValueError), match="optim_finite_difference_scheme"):
        cfg.validate()


@pytest.mark.unit
def test_invalid_finite_difference_is_rejected_before_study_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid solver policy must fail before creating a native study."""
    study_created = False

    def create_study() -> object:
        nonlocal study_created
        study_created = True
        raise AssertionError("study creation must not be reached")

    monkeypatch.setitem(sys.modules, "opensim", SimpleNamespace(MocoStudy=create_study))
    cfg = MocoTrackingConfig(optim_finite_difference_scheme="__unsupported__")
    with pytest.raises((TypeError, ValueError), match="optim_finite_difference_scheme"):
        build_moco_study("model.osim", "markers.trc", "guess.sto", cfg)
    assert not study_created


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reported_scheme", "should_raise"), [("forward", False), ("central", True)]
)
def test_forward_scheme_readback_is_checked_before_guess_creation(
    monkeypatch: pytest.MonkeyPatch,
    reported_scheme: str,
    should_raise: bool,
) -> None:
    """Native solver readback must match before its initial guess is created."""
    events: list[str] = []

    class FakeGuess:
        def insertStatesTrajectory(self, table: object, interpolate: bool) -> None:
            events.append("insert_guess")

        def getStateNames(self) -> list[str]:
            return []

        def getControlNames(self) -> list[str]:
            return []

    class FakeSolver:
        def set_num_mesh_intervals(self, value: int) -> None:
            events.append("mesh")

        def set_optim_max_iterations(self, value: int) -> None:
            events.append("iterations")

        def set_optim_convergence_tolerance(self, value: float) -> None:
            events.append("convergence_tolerance")

        def set_optim_constraint_tolerance(self, value: float) -> None:
            events.append("constraint_tolerance")

        def set_optim_finite_difference_scheme(self, value: str) -> None:
            events.append(f"finite_difference:{value}")

        def get_optim_finite_difference_scheme(self) -> str:
            events.append("finite_difference_readback")
            return reported_scheme

        def createGuess(self) -> FakeGuess:
            events.append("create_guess")
            return FakeGuess()

        def setGuess(self, guess: FakeGuess) -> None:
            events.append("set_guess")

    class FakeProblem:
        def setModelProcessor(self, processor: object) -> None:
            events.append("model_processor")

        def setTimeBounds(self, start: float, end: float) -> None:
            events.append("time_bounds")

        def addGoal(self, goal: object) -> None:
            events.append("goal")

    class FakeStudy:
        def setName(self, name: str) -> None:
            events.append("study_name")

        def updProblem(self) -> FakeProblem:
            return FakeProblem()

        def initCasADiSolver(self) -> FakeSolver:
            events.append("init_solver")
            return FakeSolver()

    class FakeGoal:
        def __init__(self, name: str) -> None:
            events.append(f"goal:{name}")

        def setWeight(self, weight: float) -> None:
            events.append("goal_weight")

        def setMarkersReference(self, reference: object) -> None:
            events.append("markers_reference")

        def setAllowUnusedReferences(self, allow: bool) -> None:
            events.append("unused_references")

    fake_opensim = SimpleNamespace(
        MocoStudy=FakeStudy,
        ModelProcessor=lambda path: object(),
        MocoControlGoal=FakeGoal,
        MocoMarkerTrackingGoal=FakeGoal,
        SetMarkerWeights=lambda: object(),
        MarkersReference=lambda path, weights: object(),
        MocoCasADiSolver=SimpleNamespace(safeDownCast=lambda solver: solver),
        TimeSeriesTable=lambda path: object(),
    )
    monkeypatch.setitem(sys.modules, "opensim", fake_opensim)

    if should_raise:
        with pytest.raises(RuntimeError, match="did not retain"):
            build_moco_study(
                "model.osim",
                "markers.trc",
                "guess.sto",
                MocoTrackingConfig(optim_finite_difference_scheme="forward"),
            )
        assert "create_guess" not in events
    else:
        build_moco_study(
            "model.osim",
            "markers.trc",
            "guess.sto",
            MocoTrackingConfig(optim_finite_difference_scheme="forward"),
        )
        assert events.index("finite_difference_readback") < events.index("create_guess")
    assert "finite_difference:forward" in events
