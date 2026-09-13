"""Shooting seeds are guesses, never a replacement for the original initial state."""

import numpy as np
import pytest
from src.shared.python.motion_matching.shooting_state_seed import select_shooting_states

pytestmark = pytest.mark.unit


def document() -> dict:
    return {
        "model_sha256": "model",
        "times_s": [0.0, 0.1, 0.2],
        "coordinates": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        "rates": [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
    }


def test_selects_detached_exact_interior_states() -> None:
    source = document()
    result = select_shooting_states(source, [0.1], model_sha256="model", dimension=2)
    np.testing.assert_array_equal(result[0.1], [0.0, 1.0, -1.0, 0.0])
    result[0.1][0] = 9
    assert source["coordinates"][1][0] == 0


@pytest.mark.parametrize(
    "times,identity",
    [([0.0], "model"), ([0.1001], "model"), ([0.1, 0.1], "model"), ([0.1], "other")],
)
def test_rejects_initial_override_missing_clock_duplicate_and_model_mismatch(
    times: list, identity: str
) -> None:
    with pytest.raises(ValueError):
        select_shooting_states(document(), times, model_sha256=identity, dimension=2)


def test_rejects_nonfinite_rates() -> None:
    source = document()
    source["rates"][1][0] = float("nan")
    with pytest.raises(ValueError):
        select_shooting_states(source, [0.1], model_sha256="model", dimension=2)
