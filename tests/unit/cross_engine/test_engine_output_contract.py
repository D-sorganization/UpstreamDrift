from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.unit
_Q, _V = np.array([0, 0, 0, 1, 0, 0, 0]), np.zeros(6)


@pytest.fixture
def parity(monkeypatch: pytest.MonkeyPatch):
    from tests.cross_engine import test_jaxsim_vs_pinocchio

    monkeypatch.setattr(
        test_jaxsim_vs_pinocchio.pytest, "importorskip", lambda *_: None
    )
    return test_jaxsim_vs_pinocchio


@pytest.fixture
def backend():
    backend = Mock()
    backend._model.gravity = -9.81
    backend.compute_mass_matrix.return_value = np.eye(6)
    backend.compute_bias_forces.return_value = np.zeros(6)
    backend.compute_gravity_forces.return_value = np.zeros(6)
    backend.compute_coriolis_matrix.return_value = np.eye(6)
    return backend


@pytest.mark.parametrize(
    ("engine", "dependency", "module", "exception"),
    [
        ("jaxsim", "jax", None, pytest.skip.Exception),
        ("pinocchio", "pinocchio", None, pytest.skip.Exception),
        ("pinocchio", None, SimpleNamespace(Model=object), RuntimeError),
        ("pinocchio", None, Mock(), pytest.skip.Exception),
    ],
)
def test_missing_optional_engine_is_a_typed_skip(
    parity, monkeypatch: pytest.MonkeyPatch, engine, dependency, module, exception
) -> None:
    def import_optional(name: str):
        if dependency == name:
            pytest.skip(f"missing {name}")
        return module

    monkeypatch.setattr(parity.pytest, "importorskip", import_optional)
    message = dependency or ("mock" if isinstance(module, Mock) else "partial")
    with pytest.raises(exception, match=message):
        args = (_Q, _V, np.eye(3)) if engine == "pinocchio" else (_Q, _V)
        getattr(parity, f"_{engine}_terms")(*args)


def test_pinocchio_malformed_mass_fails_after_required_model_setup(
    parity, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, data = Mock(), object()
    model.gravity.linear = None
    model.createData.return_value = data

    def malformed_mass(loaded, loaded_data, _q):
        assert loaded is model and loaded_data is data
        assert model.appendBodyToJoint.call_count == model.createData.call_count == 1
        np.testing.assert_array_equal(model.gravity.linear, [0, 0, -9.81])
        return np.zeros((5, 5))

    pin = SimpleNamespace(
        Model=lambda: model,
        JointModelFreeFlyer=object,
        SE3=SimpleNamespace(Identity=object),
        Inertia=lambda *_: object(),
        crba=Mock(side_effect=malformed_mass),
        rnea=Mock(),
        computeCoriolisMatrix=Mock(),
    )
    monkeypatch.setattr(parity.pytest, "importorskip", lambda *_: pin)
    with pytest.raises(AssertionError, match="shape"):
        parity._pinocchio_terms(_Q, _V, np.eye(3))
    model.addJoint.assert_called_once()
    pin.crba.assert_called_once()


def test_jaxsim_terms_performs_required_setup_before_reading_outputs(
    parity, monkeypatch: pytest.MonkeyPatch, backend
) -> None:
    q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    backend.compute_coriolis_matrix.return_value = np.zeros((6, 6))
    monkeypatch.setattr(parity, "JaxSimBackend", lambda: backend)

    parity._jaxsim_terms(q, _V)

    assert " ".join(item[0] for item in backend.mock_calls) == (
        "load_from_string set_state forward compute_mass_matrix compute_bias_forces "
        "compute_gravity_forces compute_coriolis_matrix"
    )
    assert backend.load_from_string.call_args.kwargs == {"extension": "sdf"}
    assert backend.set_state.call_args.args == (q, _V)
    monkeypatch.setattr(
        parity,
        "JaxSimBackend",
        Mock(side_effect=RuntimeError("injected initializer failure")),
    )
    with pytest.raises(RuntimeError, match="injected initializer failure"):
        parity._jaxsim_terms(_Q, _V)


@pytest.mark.parametrize(
    ("bias", "message"),
    [
        (np.full(6, np.nan), "finite"),
        (np.full(6, np.inf), "finite"),
        (np.ones(6, dtype=complex), "real numeric"),
        (np.zeros((1, 6)), "shape"),
        ([0.0] * 6, "numpy array"),
    ],
)
def test_jaxsim_terms_rejects_malformed_backend_results(
    parity, monkeypatch: pytest.MonkeyPatch, backend, bias, message
) -> None:
    backend.compute_bias_forces.return_value = bias
    monkeypatch.setattr(parity, "JaxSimBackend", lambda: backend)
    with pytest.raises(AssertionError, match=message):
        parity._jaxsim_terms(_Q, _V)
