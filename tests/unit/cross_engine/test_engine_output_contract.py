from unittest.mock import Mock

import numpy as np
import pytest

from tests.cross_engine.engine_output_contract import require_engine_array

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "shape", "message"),
    [
        ([0.0] * 6, (6,), "must be a numpy array"),
        (np.zeros(5), (6,), "shape"),
        (np.array([0.0, np.nan, 0.0]), (3,), "finite"),
        (np.array([0.0, np.inf, 0.0]), (3,), "finite"),
    ],
)
def test_engine_output_contract_rejects_invalid_arrays(value, shape, message) -> None:
    with pytest.raises(AssertionError, match=message):
        require_engine_array(value, expected_shape=shape, name="bias")


def test_engine_output_contract_preserves_valid_array_identity() -> None:
    value = np.arange(6, dtype=np.float64)
    assert require_engine_array(value, expected_shape=(6,), name="bias") is value


def _fake_backend(bias: object) -> Mock:
    backend = Mock()
    backend._model.gravity = -9.81
    backend.compute_mass_matrix.return_value = np.eye(6)
    backend.compute_bias_forces.return_value = bias
    backend.compute_gravity_forces.return_value = np.zeros(6)
    backend.compute_coriolis_matrix.return_value = np.eye(6)
    return backend


def test_jaxsim_terms_performs_required_setup_before_reading_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.cross_engine import test_jaxsim_vs_pinocchio as parity

    q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    v = np.zeros(6)
    backend = _fake_backend(np.zeros(6))
    backend.compute_coriolis_matrix.return_value = np.zeros((6, 6))

    monkeypatch.setattr(parity.pytest, "importorskip", lambda _name: None)
    monkeypatch.setattr(parity, "JaxSimBackend", lambda: backend)

    parity._jaxsim_terms(q, v)

    assert [item[0] for item in backend.mock_calls] == [
        "load_from_string",
        "set_state",
        "forward",
        "compute_mass_matrix",
        "compute_bias_forces",
        "compute_gravity_forces",
        "compute_coriolis_matrix",
    ]
    assert backend.load_from_string.call_args.kwargs == {"extension": "sdf"}
    assert backend.set_state.call_args.args[0] is q
    assert backend.set_state.call_args.args[1] is v


def test_jaxsim_terms_propagates_unexpected_initializer_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.cross_engine import test_jaxsim_vs_pinocchio as parity

    def fail_initialization() -> None:
        raise RuntimeError("injected initializer failure")

    monkeypatch.setattr(parity.pytest, "importorskip", lambda _name: None)
    monkeypatch.setattr(parity, "JaxSimBackend", fail_initialization)

    with pytest.raises(RuntimeError, match="injected initializer failure"):
        parity._jaxsim_terms(np.zeros(7), np.zeros(6))


@pytest.mark.parametrize(
    ("bias", "message"),
    [
        (np.full(6, np.nan), "finite"),
        (np.zeros((1, 6)), "shape"),
        ([0.0] * 6, "numpy array"),
    ],
)
def test_jaxsim_terms_rejects_malformed_backend_results(
    monkeypatch: pytest.MonkeyPatch, bias: object, message: str
) -> None:
    from tests.cross_engine import test_jaxsim_vs_pinocchio as parity

    backend = _fake_backend(bias)
    monkeypatch.setattr(parity.pytest, "importorskip", lambda _name: None)
    monkeypatch.setattr(parity, "JaxSimBackend", lambda: backend)

    with pytest.raises(AssertionError, match=message):
        parity._jaxsim_terms(np.zeros(7), np.zeros(6))
