"""JaxSim vs Pinocchio dynamics parity gate for issue #6654.

The gate compares the same single free rigid body in both engines and checks
the dynamics terms named in the JaxSim 2.1 acceptance criteria:

- ``M(q)`` free-floating mass matrix
- ``h(q, qd)`` bias forces
- ``g(q)`` gravity forces
- ``C(q, qd) v`` Coriolis action

The model is intentionally small because the purpose of this gate is the
cross-engine convention contract, not full humanoid coverage. Optional engine
imports are guarded so core installs skip cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.engines.physics_engines.jaxsim import JaxSimBackend
from src.shared.python.engine_core.velocity_conventions import (
    CANONICAL_GRAVITY_INERTIAL,
)

pytestmark = [
    pytest.mark.gate,
    pytest.mark.requires_jaxsim,
    pytest.mark.requires_pinocchio,
]

MATRIX_RTOL = 0.20
VECTOR_RTOL = 0.20
ABS_TOL = 1e-7

_PIN_TO_CANONICAL = np.array([3, 4, 5, 0, 1, 2], dtype=int)
_CANONICAL_TO_PIN = np.array([3, 4, 5, 0, 1, 2], dtype=int)

_SINGLE_BODY_SDF = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="inline_model">
    <link name="base_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>2.0</iyy>
          <iyz>0.0</iyz>
          <izz>3.0</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
"""


@dataclass(frozen=True)
class DynamicsTerms:
    """Dynamics terms in canonical ``[angular; linear]`` velocity order."""

    mass: np.ndarray
    bias: np.ndarray
    gravity: np.ndarray
    coriolis: np.ndarray


def _pin_to_canonical_vector(value: np.ndarray) -> np.ndarray:
    """Convert Pinocchio ``[linear; angular]`` vectors to canonical order."""

    vector = np.asarray(value, dtype=np.float64).reshape(6)
    return vector[_PIN_TO_CANONICAL]


def _pin_to_canonical_matrix(value: np.ndarray) -> np.ndarray:
    """Convert Pinocchio ``[linear; angular]`` matrices to canonical order."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (6, 6):
        raise ValueError(f"Pinocchio matrix must have shape (6, 6), got {matrix.shape}")
    return matrix[np.ix_(_PIN_TO_CANONICAL, _PIN_TO_CANONICAL)]


def _canonical_to_pin_velocity(value: np.ndarray) -> np.ndarray:
    """Convert canonical ``[angular; linear]`` vectors to Pinocchio order."""

    vector = np.asarray(value, dtype=np.float64).reshape(6)
    return vector[_CANONICAL_TO_PIN]


def _relative_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return normalized RMSE used for parity-spec cross-engine tolerances."""

    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    numerator = float(np.sqrt(np.mean((actual - expected) ** 2)))
    denominator = float(np.sqrt(np.mean(expected**2)))
    return numerator / max(denominator, ABS_TOL)


def _pin_array(value: object, name: str) -> np.ndarray:
    """Return a real Pinocchio ndarray or skip local mocked Pinocchio modules."""

    if not isinstance(value, np.ndarray):
        pytest.skip(f"Pinocchio {name} did not return an ndarray")
    return value


def _require_pinocchio_dynamics_api(pin: object) -> None:
    """Skip partial Pinocchio installs that lack model or dynamics APIs."""

    required = (
        "Model",
        "JointModelFreeFlyer",
        "SE3",
        "Inertia",
        "crba",
        "rnea",
        "computeCoriolisMatrix",
    )
    missing = [name for name in required if not hasattr(pin, name)]
    if missing:
        pytest.skip(f"Pinocchio install lacks required dynamics APIs: {missing}")


def _jaxsim_terms(q: np.ndarray, v: np.ndarray) -> DynamicsTerms:
    pytest.importorskip("jax")
    pytest.importorskip("jaxlib")
    pytest.importorskip("jaxsim")

    backend = JaxSimBackend()
    backend.load_from_string(_SINGLE_BODY_SDF, extension="sdf")
    backend.set_state(q, v)
    backend.forward()
    return DynamicsTerms(
        mass=backend.compute_mass_matrix(),
        bias=backend.compute_bias_forces(),
        gravity=backend.compute_gravity_forces(),
        coriolis=backend.compute_coriolis_matrix(),
    )


def _pinocchio_terms(q: np.ndarray, v: np.ndarray) -> DynamicsTerms:
    pin = pytest.importorskip("pinocchio")
    _require_pinocchio_dynamics_api(pin)

    model = pin.Model()
    model.gravity.linear = np.asarray(CANONICAL_GRAVITY_INERTIAL, dtype=np.float64)
    joint_id = model.addJoint(
        0,
        pin.JointModelFreeFlyer(),
        pin.SE3.Identity(),
        "root_joint",
    )
    inertia = pin.Inertia(
        1.0,
        np.zeros(3, dtype=np.float64),
        np.diag([1.0, 2.0, 3.0]),
    )
    model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
    data = model.createData()

    q_pin = np.concatenate([q[:3], q[[4, 5, 6, 3]]])
    v_pin = _canonical_to_pin_velocity(v)
    zero_acc = np.zeros(6, dtype=np.float64)

    mass = _pin_array(pin.crba(model, data, q_pin), "crba")
    mass = (mass + mass.T) / 2.0
    bias = _pin_array(pin.rnea(model, data, q_pin, v_pin, zero_acc), "rnea")
    gravity = _pin_array(
        pin.rnea(model, data, q_pin, np.zeros(6, dtype=np.float64), zero_acc),
        "gravity rnea",
    )
    coriolis = _pin_array(
        pin.computeCoriolisMatrix(model, data, q_pin, v_pin),
        "computeCoriolisMatrix",
    )

    return DynamicsTerms(
        mass=_pin_to_canonical_matrix(mass),
        bias=_pin_to_canonical_vector(bias),
        gravity=_pin_to_canonical_vector(gravity),
        coriolis=_pin_to_canonical_matrix(coriolis),
    )


@pytest.mark.parametrize(
    ("q", "v"),
    [
        (
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.20, -0.15, 0.10, 0.0, 0.0, 0.0]),
        ),
        (
            np.array([0.10, -0.20, 0.30, 1.0, 0.0, 0.0, 0.0]),
            np.array([-0.30, 0.25, 0.40, 0.10, -0.20, 0.30]),
        ),
    ],
)
def test_jaxsim_pinocchio_free_body_dynamics_terms_match(
    q: np.ndarray,
    v: np.ndarray,
) -> None:
    """JaxSim and Pinocchio agree on ``M,h,g,C`` within parity tolerances."""

    jaxsim_terms = _jaxsim_terms(q, v)
    pinocchio_terms = _pinocchio_terms(q, v)

    assert _relative_rmse(jaxsim_terms.mass, pinocchio_terms.mass) < MATRIX_RTOL
    assert _relative_rmse(jaxsim_terms.bias, pinocchio_terms.bias) < VECTOR_RTOL
    assert _relative_rmse(jaxsim_terms.gravity, pinocchio_terms.gravity) < VECTOR_RTOL
    np.testing.assert_allclose(
        jaxsim_terms.coriolis @ v,
        pinocchio_terms.coriolis @ v,
        rtol=VECTOR_RTOL,
        atol=ABS_TOL,
    )


def test_parity_metric_reports_zero_for_identical_terms() -> None:
    """The RMSE helper is dependency-free and exact for identical arrays."""

    terms = np.eye(3, dtype=np.float64)
    assert _relative_rmse(terms, terms.copy()) == 0.0


def test_pinocchio_order_conversion_maps_linear_angular_to_canonical() -> None:
    """Pinocchio vectors are normalized before cross-engine comparison."""

    pin_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    canonical = _pin_to_canonical_vector(pin_vector)

    np.testing.assert_allclose(canonical, np.array([4.0, 5.0, 6.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(_canonical_to_pin_velocity(canonical), pin_vector)
