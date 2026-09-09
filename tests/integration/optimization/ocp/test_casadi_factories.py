"""Real-process matrix factory compatibility for pinned Bioptim (#9842)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.shared.python.optimization.ocp import _compat

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_bioptim,
    pytest.mark.skipif(not _compat.bioptim_available(), reason="bioptim not installed"),
]

_FACTORY_PROBE = """
import casadi as ca
import numpy as np
from src.shared.python.optimization.ocp._compat import require_bioptim

names = ("MX_eye", "SX_eye", "DM_eye")
original = {name: getattr(ca, name, None) for name in names}
require_bioptim()
factories = {name: getattr(ca, name) for name in names}
for name, kind in zip(names, (ca.MX, ca.SX, ca.DM), strict=True):
    factory = factories[name]
    if original[name] is not None:
        assert factory is original[name], "existing SDK factory was replaced"
    for size in (0, 1, 3):
        matrix = factory(size)
        assert isinstance(matrix, kind)
        assert matrix.shape == (size, size)
        evaluated = matrix if kind is ca.DM else ca.Function("identity", [], [matrix])()["o0"]
        np.testing.assert_array_equal(np.asarray(evaluated), np.eye(size))
    if kind is ca.DM:
        continue  # A numerical DM matrix has no symbolic differentiation variable.
    state = kind.sym("state", 3)
    derivative = ca.jacobian(factory(3) @ state, state)
    evaluated = ca.Function("derivative", [state], [derivative])([1, 2, 3])
    np.testing.assert_array_equal(np.asarray(evaluated), np.eye(3))
require_bioptim()
assert all(getattr(ca, name) is factories[name] for name in names)
"""


def test_pinned_bioptim_uses_exact_idempotent_sdk_matrix_factories() -> None:
    """Exercise genuine imports/graphs, isolated from the unit tree's mocks."""
    root = Path(__file__).resolve().parents[4]
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
    }
    result = subprocess.run(
        [sys.executable, "-c", _FACTORY_PROBE],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
