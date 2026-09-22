"""NM-01 (#10616): DbC that survives python -O."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SNIPPET = r"""
from src.shared.python.neural_motion.experiment import break_even_queries
from src.shared.python.neural_motion.tasks import ConditioningSpec

try:
    ConditioningSpec(
        geometry_id="g",
        q0=(0.0,),
        v0=(0.0,),
        horizon_s=0.5,
        time_step_s=0.01,
        constraint_profile="none",
        contact_profile="none",
        observation_mask=(),
    )
except ValueError as exc:
    assert "observation_mask" in str(exc).lower()
else:
    raise SystemExit("expected ValueError for empty observation mask")

assert break_even_queries(10.0, 0.0) is None
assert break_even_queries(10.0, -0.5) is None
print("OK")
"""


def test_nm01_contracts_reject_under_optimize() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-O", "-c", _SNIPPET],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
