"""JaxSim vs Pinocchio dynamics parity gate for issue #14290.

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
from src.shared.python.pendulum_simulator.tolerance_bands import (
    compute_cross_engine_tolerance_band,
)
from tests.cross_engine.engine_output_contract import require_engine_array

pytestmark = [
    pytest.mark.gate,
    pytest.mark.requires_jaxsim,
    pytest.mark.requires_pinocchio,
]

_PARITY_BAND = compute_cross_engine_tolerance_band(
    is_pointwise=True,
    epsilon_model=0.20,
    floor=1e-6,
)
MATRIX_RTOL = _PARITY_BAND.tolerance
VECTOR_RTOL = _PARITY_BAND.tolerance
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


def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotate_pin_spatial(value: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Map canonical local [angular; linear] vectors to world axes at p=0."""
    return np.concatenate([rotation @ value[:3], rotation @ value[3:]])


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
    """Require a finite Pinocchio vector; malformed outputs fail the gate."""

    return require_engine_array(value, expected_shape=(6,), name=f"Pinocchio {name}")


def _pin_matrix(value: object, name: str) -> np.ndarray:
    """Require a finite Pinocchio matrix; malformed outputs fail the gate."""

    return require_engine_array(value, expected_shape=(6, 6), name=f"Pinocchio {name}")


def _require_pinocchio_dynamics_api(pin: object) -> None:
    """Skip explicit mocks but reject broken real engine installations."""

    if type(pin).__module__ == "unittest.mock":
        pytest.skip("explicit Pinocchio mock")
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
        raise RuntimeError(
            f"partial Pinocchio install lacks required dynamics APIs: {missing}"
        )


def _jaxsim_terms(q: np.ndarray, v: np.ndarray) -> DynamicsTerms:
    pytest.importorskip("jax")
    pytest.importorskip("jaxlib")
    pytest.importorskip("jaxsim")

    backend = JaxSimBackend()
    backend.load_from_string(_SINGLE_BODY_SDF, extension="sdf")
    assert float(np.asarray(backend._model.gravity)) == pytest.approx(-9.81)
    backend.set_state(q, v)
    backend.forward()
    return DynamicsTerms(
        mass=require_engine_array(
            backend.compute_mass_matrix(), expected_shape=(6, 6), name="JaxSim mass"
        ),
        bias=require_engine_array(
            backend.compute_bias_forces(), expected_shape=(6,), name="JaxSim bias"
        ),
        gravity=require_engine_array(
            backend.compute_gravity_forces(), expected_shape=(6,), name="JaxSim gravity"
        ),
        coriolis=require_engine_array(
            backend.compute_coriolis_matrix(),
            expected_shape=(6, 6),
            name="JaxSim Coriolis matrix",
        ),
    )


def _pinocchio_terms(q: np.ndarray, v: np.ndarray) -> DynamicsTerms:
    pin = pytest.importorskip("pinocchio")
    _require_pinocchio_dynamics_api(pin)

    model = pin.Model()
    model.gravity.linear = np.array([0.0, 0.0, -9.81], dtype=np.float64)
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
    rotation = _rotation_from_wxyz(q[3:7])
    v_local = np.concatenate([rotation.T @ v[:3], rotation.T @ v[3:6]])
    v_pin = _canonical_to_pin_velocity(v_local)
    zero_acc = np.zeros(6, dtype=np.float64)

    mass = _pin_matrix(pin.crba(model, data, q_pin), "crba")
    mass = (mass + mass.T) / 2.0
    bias = _pin_array(pin.rnea(model, data, q_pin, v_pin, zero_acc), "rnea")
    gravity = _pin_array(
        pin.rnea(model, data, q_pin, np.zeros(6, dtype=np.float64), zero_acc),
        "gravity rnea",
    )
    coriolis = _pin_matrix(
        pin.computeCoriolisMatrix(model, data, q_pin, v_pin),
        "computeCoriolisMatrix",
    )

    axes = np.block([[rotation, np.zeros((3, 3))], [np.zeros((3, 3)), rotation]])
    return DynamicsTerms(
        mass=axes @ _pin_to_canonical_matrix(mass) @ axes.T,
        bias=_rotate_pin_spatial(_pin_to_canonical_vector(bias), rotation),
        gravity=_rotate_pin_spatial(_pin_to_canonical_vector(gravity), rotation),
        coriolis=axes @ _pin_to_canonical_matrix(coriolis) @ axes.T,
    )


@pytest.mark.parametrize(
    ("case_id", "q", "v"),
    [
        (
            "identity-angular-only",
            # Identity pose: tests gyroscopic (ω×Iω), gravity, angular-only velocity.
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.20, -0.15, 0.10, 0.0, 0.0, 0.0]),
        ),
        (
            "identity-mixed-velocity",
            # Zero position, full mixed velocity: tests Coriolis (ω×mv) and
            # linear velocity contributions. Position must be zero because JaxSim
            # (INERTIAL representation, angular momentum about world origin) and
            # Pinocchio (LOCAL representation, CoM-relative) produce equivalent
            # results only when the body origin coincides with the world origin.
            # Non-zero position shifts the angular-momentum reference point and
            # causes a physically-correct but representation-dependent divergence
            # in M and h that exceeds the parity tolerance (issue #7097).
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([-0.30, 0.25, 0.40, 0.10, -0.20, 0.30]),
        ),
        (
            "rotated-world-origin",
            # R=Rz(90)Rx(45), p=0; JaxSim INERTIAL world-origin twist,
            # Pinocchio LOCAL CoM twist. v and omega are world-axis quantities.
            np.array(
                [0.0, 0.0, 0.0, 0.6532814824, 0.2705980501, 0.2705980501, 0.6532814824]
            ),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
    ],
    ids=["identity-angular-only", "identity-mixed-velocity", "rotated-world-origin"],
)
def test_jaxsim_pinocchio_free_body_dynamics_terms_match(
    case_id: str,
    q: np.ndarray,
    v: np.ndarray,
) -> None:
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
    if case_id == "rotated-world-origin":
        expected_mass = np.diag([2.5, 1.0, 2.5, 1.0, 1.0, 1.0])
        expected_mass[0, 2] = expected_mass[2, 0] = 0.5
        np.testing.assert_allclose(jaxsim_terms.mass, expected_mass, atol=ABS_TOL)
        np.testing.assert_allclose(
            jaxsim_terms.bias, [10.0, 4.0, -6.0, -3.0, 6.0, 6.81], atol=ABS_TOL
        )
        np.testing.assert_allclose(
            jaxsim_terms.gravity, [0, 0, 0, 0, 0, 9.81], atol=ABS_TOL
        )
        np.testing.assert_allclose(
            jaxsim_terms.coriolis @ v, [10.0, 4.0, -6.0, -3.0, 6.0, -3.0], atol=ABS_TOL
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
