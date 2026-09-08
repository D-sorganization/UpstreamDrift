"""Phase 1.1: ``SymbolicSwingModel`` against the Pinocchio oracle."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


CASADI = _available("casadi")
PIN = _available("pinocchio")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_casadi,
    pytest.mark.skipif(not CASADI, reason="casadi not installed"),
]

if PIN:
    import pinocchio as pin

from src.shared.python.optimization._swing_kinematics import JOINTS  # noqa: E402
from src.shared.python.optimization._swing_models import (  # noqa: E402
    ClubModel,
    GolferModel,
)
from src.shared.python.optimization.ocp.symbolic_model import (  # noqa: E402
    MARKER_NAMES,
    SymbolicSwingModel,
)

_EMPTY = np.zeros(0)


@pytest.fixture(scope="module")
def model() -> SymbolicSwingModel:
    return SymbolicSwingModel(GolferModel(), ClubModel())


@pytest.mark.skipif(not PIN, reason="pinocchio not installed")
def test_dynamics_kernels_match_pinocchio(model: SymbolicSwingModel) -> None:
    from src.shared.python.optimization.model_provider import build_pinocchio_model

    pin_model = build_pinocchio_model(model.golfer, model.club)
    data = pin_model.createData()
    rng = np.random.default_rng(11)
    for _ in range(50):
        q = rng.uniform(-1.0, 1.0, 7)
        v = rng.uniform(-5.0, 5.0, 7)
        a = rng.uniform(-20.0, 20.0, 7)
        tau = rng.uniform(-50.0, 50.0, 7)
        np.testing.assert_allclose(
            np.asarray(model.rnea(q, v, a, _EMPTY)).ravel(),
            pin.rnea(pin_model, data, q, v, a),
            atol=1e-9,
        )
        crba = pin.crba(pin_model, data, q)
        crba = np.triu(crba) + np.triu(crba, 1).T
        np.testing.assert_allclose(
            np.asarray(model.mass_matrix(q, _EMPTY)), crba, atol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(model.nonlinear_effects(q, v, _EMPTY)).ravel(),
            pin.nonLinearEffects(pin_model, data, q, v),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.asarray(model.forward_dynamics(q, v, tau, _EMPTY)).ravel(),
            pin.aba(pin_model, data, q, v, tau),
            atol=1e-8,
        )


@pytest.mark.skipif(not PIN, reason="pinocchio not installed")
def test_markers_and_com_match_pinocchio_placements(model: SymbolicSwingModel) -> None:
    from src.shared.python.optimization.model_provider import build_pinocchio_model

    pin_model = build_pinocchio_model(model.golfer, model.club)
    data = pin_model.createData()
    rng = np.random.default_rng(5)
    for _ in range(20):
        q = rng.uniform(-1.0, 1.0, 7)
        v = rng.uniform(-5.0, 5.0, 7)
        pin.forwardKinematics(pin_model, data, q, v)
        # Joint i of the URDF chain is Pinocchio joint i + 1 (0 is the universe).
        expected = np.column_stack([data.oMi[i + 1].translation for i in range(7)])
        np.testing.assert_allclose(
            np.asarray(model.joint_positions(q, _EMPTY)), expected, atol=1e-12
        )
        markers = np.asarray(model.markers(q, _EMPTY))
        assert markers.shape == (3, len(MARKER_NAMES))
        np.testing.assert_allclose(
            markers[:, model.marker_index("clubhead")], expected[:, 6]
        )
        np.testing.assert_allclose(
            markers[:, model.marker_index("shoulder")], expected[:, 2]
        )
        com = pin.centerOfMass(pin_model, data, q, v)
        np.testing.assert_allclose(
            np.asarray(model.center_of_mass(q, _EMPTY)).ravel(), com, atol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(model.center_of_mass_velocity(q, v, _EMPTY)).ravel(),
            data.vcom[0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            float(model.total_mass(_EMPTY)), pin.computeTotalMass(pin_model)
        )


def test_clubhead_matches_legacy_kernel(model: SymbolicSwingModel) -> None:
    import casadi as ca

    from src.shared.python.optimization.casadi_backend import build_clubhead_position

    legacy = build_clubhead_position(model.golfer, model.club)
    q = np.array([0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7])
    v = np.array([1.0, -2.0, 0.5, 0.25, -1.5, 3.0, -0.75])
    np.testing.assert_allclose(
        np.asarray(model.clubhead_position(q, _EMPTY)).ravel(),
        np.asarray(legacy(q)).ravel(),
        atol=1e-12,
    )
    qs = ca.SX.sym("q", 7)
    vs = ca.SX.sym("v", 7)
    speed = ca.Function("s", [qs, vs], [ca.jtimes(legacy(qs), qs, vs)])
    np.testing.assert_allclose(
        np.asarray(model.clubhead_velocity(q, v, _EMPTY)).ravel(),
        np.asarray(speed(q, v)).ravel(),
        atol=1e-12,
    )


def test_parameterised_model_reproduces_numeric_model_at_nominal() -> None:
    golfer, club = GolferModel(arm_length=0.62), ClubModel(total_length=1.2)
    numeric = SymbolicSwingModel(golfer, club)
    symbolic = SymbolicSwingModel(
        golfer, club, parameters=("arm_length", "club_length", "mass")
    )
    assert symbolic.parameter_names == ("arm_length", "club_length", "mass")
    nominal = symbolic.nominal_parameters()
    np.testing.assert_allclose(nominal, [0.62, 1.2, golfer.mass])
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    v = np.ones(7)
    tau = np.linspace(-10, 10, 7)
    np.testing.assert_allclose(
        np.asarray(symbolic.forward_dynamics(q, v, tau, nominal)),
        np.asarray(numeric.forward_dynamics(q, v, tau, _EMPTY)),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(symbolic.markers(q, nominal)), np.asarray(numeric.markers(q, _EMPTY))
    )
    # A longer arm moves the wrist marker: the parameter is live.
    longer = nominal.copy()
    longer[0] *= 1.1
    wrist = symbolic.marker_index("wrist")
    assert not np.allclose(
        np.asarray(symbolic.markers(q, longer))[:, wrist],
        np.asarray(symbolic.markers(q, nominal))[:, wrist],
    )
    assert float(symbolic.total_mass(longer)) == pytest.approx(
        float(numeric.total_mass(_EMPTY))
    )


def test_parameter_names_validated() -> None:
    with pytest.raises(ValueError, match="unknown parameters"):
        SymbolicSwingModel(parameters=("shoe_size",))
    with pytest.raises(ValueError, match="unique"):
        SymbolicSwingModel(parameters=("mass", "mass"))
    model = SymbolicSwingModel()
    assert model.dof_names == tuple(JOINTS)
    assert model.torque_limits().shape == (7,)
    with pytest.raises(KeyError):
        model.marker_index("nose")
