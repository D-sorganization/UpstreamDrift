import os
import subprocess
import sys
from pathlib import Path


def test_golf_modeling_demo_import_path():
    """Verify the portfolio demo import path works without rusting out or failing."""
    cmd = [
        sys.executable,
        "-c",
        "from src.shared.python.physics.ball_flight_physics import BallFlightSimulator, LaunchConditions; print(BallFlightSimulator.__name__, LaunchConditions.__name__)",
    ]
    repo_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=repo_root,
        env=env,
    )
    assert result.returncode == 0, f"Portfolio demo import failed: {result.stderr}"
    assert "BallFlightSimulator LaunchConditions" in result.stdout
