from pathlib import Path
from unittest.mock import MagicMock, patch

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples"


def run_example(name, monkeypatch):
    import runpy

    # Prevent sys.exit from killing tests
    monkeypatch.setattr("sys.exit", lambda code=0: None)
    runpy.run_path(str(EXAMPLES_DIR / name), run_name="__main__")


def test_injury_risk_tutorial(monkeypatch):
    run_example("03_injury_risk_tutorial.py", monkeypatch)


def test_aerodynamics_demo(monkeypatch):
    run_example("aerodynamics_demo.py", monkeypatch)


def test_basic_flight_simulation(monkeypatch):
    run_example("basic_flight_simulation.py", monkeypatch)


def test_topography_demo(monkeypatch):
    run_example("topography_demo.py", monkeypatch)


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.close")
def test_motion_training_demo(mock_close, mock_show, monkeypatch, tmp_path):
    import sys

    sys.modules["motion_training"] = MagicMock()
    sys.modules["motion_training.club_trajectory_parser"] = MagicMock()
    sys.modules["motion_training.dual_hand_ik_solver"] = MagicMock()
    sys.modules["motion_training.trajectory_exporter"] = MagicMock()
    sys.modules["motion_training.motion_visualizer"] = MagicMock()

    # Mock visualizers, UI, IK solvers, parse functions.
    mock_parser_class = MagicMock()
    mock_parser = mock_parser_class.return_value
    mock_trajectory = MagicMock()
    mock_trajectory.num_frames = 10
    mock_trajectory.duration = 1.0  # Actually float
    mock_trajectory.events.address = 0
    mock_trajectory.events.top = 4
    mock_trajectory.events.impact = 7
    mock_trajectory.events.finish = 9
    mock_trajectory.events.club_head_speed = 100.0
    mock_trajectory.frames = list(range(10))
    mock_parser.parse.return_value = mock_trajectory

    sys.modules[
        "motion_training.club_trajectory_parser"
    ].ClubTrajectoryParser = mock_parser_class

    mock_ik_result = MagicMock()
    mock_ik_result.convergence_rate = 1.0
    mock_ik_result.left_hand_errors = [0.0]
    mock_ik_result.right_hand_errors = [0.0]

    mock_solver_class = MagicMock()
    mock_solver = mock_solver_class.return_value
    mock_solver.model.nq = 10
    mock_solver.solve_trajectory.return_value = mock_ik_result

    sys.modules[
        "motion_training.dual_hand_ik_solver"
    ].create_ik_solver = mock_solver_class

    # Needs to mock before run_path execution via sys.modules or monkeypatch on import
    # But for a script run via runpy, monkeypatching the actual module objects works if they are imported!
    # Instead, let's just use run_example and mock out sys.argv.

    monkeypatch.setattr(
        "sys.argv",
        [
            "motion_training_demo.py",
            "--trajectory",
            "dummy.xlsx",
            "--sheet",
            "TW_wiffle",
            "--urdf",
            "dummy.urdf",
            "--output",
            str(tmp_path),
            "--plot-only",
        ],
    )

    # We will just run plot-only so it doesn't need IK
    monkeypatch.setattr("pathlib.Path.exists", lambda s: True)

    run_example("motion_training_demo.py", monkeypatch)
