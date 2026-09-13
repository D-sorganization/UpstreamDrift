"""Regression tests for the guarded native batch-driver generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[3]
    / "docs"
    / "development"
    / "native_parallel_performance"
    / "prepare_batched_solver_driver.py"
)


@pytest.mark.unit
def test_generated_driver_is_guarded_and_compiles(tmp_path: Path) -> None:
    """Spawn-safe output keeps all operational statements inside ``main``."""
    source = tmp_path / "run20.py"
    output = tmp_path / "generated.py"
    source.write_text(
        "import argparse\n"
        "from src.shared.python.motion_matching.multi_shooting_fit import MultipleShootingOptions,fit_multiple_shooting\n"
        "parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--audit-only',action='store_true');parser.add_argument('--horizon',type=float)\n"
        "def full(theta,clock):return replay_candidate(raw,candidate(theta),clock,rtol=1e-11,atol=1e-13,max_step=.00025).markers_m\n"
        "evaluation_count=0\n"
        "def checkpoint(theta,states,cost):\n"
        " global evaluation_count\n"
        "options=MultipleShootingOptions(solver='slsqp',)\n"
        "(out/'returned-nodes.json').write_text(json.dumps({str(k):v.tolist() for k,v in fit.intermediate_states.items()},indent=2)+'\\n')\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = output.read_text()
    assert "def main():" in generated
    assert "nonlocal evaluation_count" in generated
    assert "if __name__ == '__main__':\n main()" in generated
    assert "--batch-workers" in generated
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
