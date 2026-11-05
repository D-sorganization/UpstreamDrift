"""Registered lie modifier for dispersion tables (#14610)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.sg_optimizer.shot_model.dispersion_lie import (
    lie_modifier_for_table,
)

pytestmark = pytest.mark.unit


def test_lie_modifier_for_table_is_deterministic_and_non_identity_on_rough() -> None:
    rows = np.tile(np.array([100.0, 1.0, 2.0, -3.0, 4.0, 5.0]), (200, 1))
    mod = lie_modifier_for_table(12)
    fair = mod(rows, "fairway")
    rough = mod(rows, "rough")
    assert np.array_equal(fair, rows)
    assert rough[:, 0].mean() < 100.0
    mod2 = lie_modifier_for_table(12)
    np.testing.assert_allclose(rough, mod2(rows, "rough"))
