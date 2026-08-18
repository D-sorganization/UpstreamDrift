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


def _module_available(name: str) -> bool:
    """find_spec that tolerates mock modules other suites place in
    sys.modules without a __spec__ (find_spec raises ValueError there).

    A spec-less entry is a test mock, not a usable install, so it counts
    as unavailable — these tests exercise real engine behavior.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


_HAS_PINOCCHIO = _module_available("pinocchio")
_HAS_PINK = _module_available("pink")
_HAS_CROCODDYL = _module_available("crocoddyl")


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
