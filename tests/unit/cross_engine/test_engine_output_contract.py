from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.unit
_Q, _V = np.array([0, 0, 0, 1, 0, 0, 0]), np.zeros(6)


def _fake_backend(bias: object) -> Mock:
    backend = Mock()
    backend._model.gravity = -9.81
    backend.compute_mass_matrix.return_value = np.eye(6)
    backend.compute_bias_forces.return_value = bias
    backend.compute_gravity_forces.return_value = np.zeros(6)
    backend.compute_coriolis_matrix.return_value = np.eye(6)
    return backend


def _parity(monkeypatch: pytest.MonkeyPatch, backend: Mock | None = None):
    from tests.cross_engine import test_jaxsim_vs_pinocchio as parity

    monkeypatch.setattr(parity.pytest, "importorskip", lambda _name: None)
    if backend is not None:
        monkeypatch.setattr(parity, "JaxSimBackend", lambda: backend)
    return parity


@pytest.mark.parametrize(
    ("engine", "missing"),
    [
        ("jaxsim", "jax"),
        ("pinocchio", "pinocchio"),
        ("pinocchio", "partial"),
        ("pinocchio", "mock"),
    ],
)
def test_missing_optional_engine_is_a_typed_skip(
    monkeypatch: pytest.MonkeyPatch, engine: str, missing: str
) -> None:
    parity = _parity(monkeypatch)
    stub = (
        Mock()
        if missing == "mock"
        else Mock(spec=["Model"])
        if missing == "partial"
        else None
    )
    monkeypatch.setattr(
        parity.pytest,
        "importorskip",
        lambda name: (
            stub
            if stub is not None
            else pytest.skip(f"missing {name}")
            if name == missing
            else None
        ),
    )
    reason = "explicit Pinocchio mock" if stub else f"missing {missing}"
    with pytest.raises(pytest.skip.Exception, match=reason):
        getattr(parity, f"_{engine}_terms")(_Q, _V)


def test_partial_pinocchio_install_is_not_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parity = _parity(monkeypatch)
    monkeypatch.setattr(
        parity.pytest,
        "importorskip",
        lambda _name: SimpleNamespace(Model=object),
    )
    with pytest.raises(RuntimeError, match="partial Pinocchio"):
        try:
            parity._pinocchio_terms(_Q, _V)
        except pytest.skip.Exception as exc:
            pytest.fail(f"partial Pinocchio install was incorrectly skipped: {exc}")


def test_pinocchio_malformed_mass_fails_after_required_model_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parity = _parity(monkeypatch)
    model = Mock()
    model.gravity = Mock(linear=None)
    model.createData.return_value = object()
    pin = SimpleNamespace(
        Model=lambda: model,
        JointModelFreeFlyer=object,
        SE3=SimpleNamespace(Identity=object),
        Inertia=lambda *_args: object(),
        crba=Mock(),
        rnea=Mock(),
        computeCoriolisMatrix=Mock(),
    )

    def malformed_mass(loaded, data, _q):
        assert loaded is model and data is model.createData.return_value
        assert model.appendBodyToJoint.call_count == model.createData.call_count == 1
        np.testing.assert_array_equal(model.gravity.linear, [0, 0, -9.81])
        return np.zeros((5, 5))

    pin.crba.side_effect = malformed_mass
    monkeypatch.setattr(parity.pytest, "importorskip", lambda _name: pin)
    with pytest.raises(AssertionError, match="shape"):
        parity._pinocchio_terms(_Q, _V)
    model.addJoint.assert_called_once()
    pin.crba.assert_called_once()


def test_jaxsim_terms_performs_required_setup_before_reading_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    backend = _fake_backend(np.zeros(6))
    backend.compute_coriolis_matrix.return_value = np.zeros((6, 6))
    parity = _parity(monkeypatch, backend)

    parity._jaxsim_terms(q, _V)

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
    assert backend.set_state.call_args.args[1] is _V
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
    monkeypatch: pytest.MonkeyPatch, bias: object, message: str
) -> None:
    backend = _fake_backend(bias)
    parity = _parity(monkeypatch, backend)

    with pytest.raises(AssertionError, match=message):
        parity._jaxsim_terms(_Q, _V)
