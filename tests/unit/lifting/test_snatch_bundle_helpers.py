"""Snatch support in the LIFT-5 bundle pipeline (#11745).

``_bar_path_m`` routes the ``snatch`` bar convention to the vendored Tools
phase-correct bar (hanging through the pulls, overhead from the catch on);
``_bent_elbow_seed`` is the retry seed of the hang search that keeps the arm
solve at the snatch's 0.9 m grip out of a hyperextended-elbow minimum.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.lifting import bundles

pytestmark = pytest.mark.unit


class _Dyn:
    """Bar one unit below (``deadlift``) or above (``snatch``) the angle sum."""

    def bar_position(self, q: np.ndarray, convention: str) -> np.ndarray:
        sign = -1.0 if convention == "deadlift" else 1.0
        return np.array([float(np.sum(q)), sign])


def test_non_snatch_conventions_use_the_configuration_bar() -> None:
    angles = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.5]])
    path = bundles._bar_path_m(_Dyn(), angles, np.array([0.0, 1.0]), "deadlift")
    np.testing.assert_allclose(path, [[0.6, -1.0], [0.5, -1.0]])


def test_snatch_bar_hangs_in_the_pull_and_is_overhead_from_the_catch() -> None:
    snatch = pytest.importorskip("movement_optimizer.exercises.snatch")
    if not hasattr(snatch, "snatch_bar_path"):
        pytest.skip("vendored Tools predates the phase-correct snatch bar")
    time_s = np.linspace(1.0, 3.0, 21)  # starts away from zero on purpose
    angles = np.zeros((time_s.size, 3))
    path = bundles._bar_path_m(_Dyn(), angles, time_s, "snatch")
    frac = (time_s - time_s[0]) / (time_s[-1] - time_s[0])
    assert np.all(path[frac <= snatch.SNATCH_PHASES["turnover"], 1] == -1.0)
    assert np.all(path[frac >= snatch.SNATCH_PHASES["catch"], 1] == 1.0)


def test_bent_elbow_seed_flexes_only_the_elbows_within_their_bounds() -> None:
    order = ("LEInput", "LSInputX", "REInput", "pelvis_tilt")
    standing = SimpleNamespace(order=order)
    q = np.array([-0.35, 0.2, -0.35, 0.1])
    seed = bundles._bent_elbow_seed(standing, q)
    assert seed[0] == seed[2] == bundles._HANG_RETRY_ELBOW_RAD
    np.testing.assert_array_equal(seed[[1, 3]], q[[1, 3]])
    np.testing.assert_array_equal(q, [-0.35, 0.2, -0.35, 0.1])
    # Inside the cited elbow range (-150 deg, +5 deg) of the bounded solve.
    assert np.radians(-150.0) < bundles._HANG_RETRY_ELBOW_RAD < np.radians(5.0)
