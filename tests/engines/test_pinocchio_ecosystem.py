"""Pinocchio ecosystem availability tests (Pinocchio, Pink, Crocoddyl).

Replaces a hollow ``unittest`` shell that advertised ecosystem coverage but
contained zero tests (``test_classes = []`` — epic #8390, A5/#8395). These
tests are import-gated: each skips when the optional package is absent, and
asserts real behavior when present. Deeper Crocoddyl solver coverage lands
with the DDP adapter (#8399).
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_PINOCCHIO = importlib.util.find_spec("pinocchio") is not None
_HAS_PINK = importlib.util.find_spec("pink") is not None
_HAS_CROCODDYL = importlib.util.find_spec("crocoddyl") is not None


@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
def test_pinocchio_builds_and_steps_a_sample_model() -> None:
    import numpy as np
    import pinocchio as pin

    model = pin.buildSampleModelHumanoid()
    data = model.createData()
    q = pin.neutral(model)
    v = np.zeros(model.nv)
    tau = pin.rnea(model, data, q, v, np.zeros(model.nv))
    assert tau.shape == (model.nv,)
    assert np.all(np.isfinite(tau))


@pytest.mark.skipif(not _HAS_PINK, reason="pin-pink not installed")
def test_pink_configuration_wraps_pinocchio_model() -> None:
    import pinocchio as pin
    from pink import Configuration

    model = pin.buildSampleModelHumanoid()
    configuration = Configuration(model, model.createData(), pin.neutral(model))
    assert configuration.q.shape[0] == model.nq


@pytest.mark.skipif(not _HAS_CROCODDYL, reason="crocoddyl not installed")
def test_crocoddyl_state_multibody_constructs() -> None:
    import crocoddyl
    import pinocchio as pin

    model = pin.buildSampleModelHumanoid()
    state = crocoddyl.StateMultibody(model)
    assert state.nx == model.nq + model.nv


def test_availability_flags_are_consistent() -> None:
    """PINK and Crocoddyl both require Pinocchio; flag combinations that
    claim otherwise indicate a broken environment probe."""
    if _HAS_PINK or _HAS_CROCODDYL:
        assert _HAS_PINOCCHIO, "pink/crocoddyl present without pinocchio"
