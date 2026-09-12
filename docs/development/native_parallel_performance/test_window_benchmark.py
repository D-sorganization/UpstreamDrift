"""The benchmark must reject even one changed returned derivative."""

import numpy as np
import pytest

from window_benchmark import compare_arrays


def test_all_arrays_are_checked() -> None:
    reference = {name: np.ones((2, 3)) for name in ("markers", "end", "mj", "ej")}
    assert all(value == 0 for value in compare_arrays(reference, reference).values())
    for name in reference:
        changed = {key: value.copy() for key, value in reference.items()}
        changed[name][0, 0] += 1e-8
        with pytest.raises(ValueError, match="differ"):
            compare_arrays(reference, changed)


def test_invalid_arrays_are_rejected() -> None:
    with pytest.raises(ValueError):
        compare_arrays({"x": np.ones(2)}, {"x": np.ones(3)})
    with pytest.raises(ValueError):
        compare_arrays({"x": np.array([np.nan])}, {"x": np.array([np.nan])})
    with pytest.raises(ValueError):
        compare_arrays({"x": np.ones(2)}, {})
