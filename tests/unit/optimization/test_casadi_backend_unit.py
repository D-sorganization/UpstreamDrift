"""Unit tests for the CasADi backend's degradation path (B3/#8398).

This tree's conftest installs a spec-less ``casadi`` MagicMock, so the
mock-tolerant probe must report the SDK as unavailable and the backend
must raise with an install hint. Real solves live in
``tests/integration/optimization/test_casadi_swing_live.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
)
from src.shared.python.optimization.casadi_backend import (
    CasadiNotAvailableError,
    casadi_available,
    require_casadi,
    solve_swing_casadi,
)
from src.shared.python.optimization.model_provider import swing_joint_limits

pytestmark = pytest.mark.unit


def test_mocked_casadi_counts_as_unavailable() -> None:
    assert casadi_available() is False


def test_require_casadi_raises_with_install_hint() -> None:
    with pytest.raises(CasadiNotAvailableError, match="optimal-control"):
        require_casadi()


def test_solve_raises_when_casadi_unavailable() -> None:
    golfer, club = GolferModel(), ClubModel()
    config = OptimizationConfig(n_nodes=5)
    with pytest.raises(CasadiNotAvailableError):
        solve_swing_casadi(
            golfer,
            club,
            config,
            {},
            swing_joint_limits(golfer),
            np.zeros(2 * 7 * 5),
        )
