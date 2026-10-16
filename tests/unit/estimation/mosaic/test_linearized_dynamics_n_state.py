"""``LinearizedDynamics.n_state`` exposes the state dimension of ``A``."""

from __future__ import annotations

import numpy as np

from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics


def test_n_state_equals_a_second_axis() -> None:
    dyn = LinearizedDynamics(np.zeros((5, 4, 4)), np.zeros((5, 4, 2)))
    assert dyn.n_state == dyn.a.shape[1] == 4
    assert dyn.n_steps == 5
