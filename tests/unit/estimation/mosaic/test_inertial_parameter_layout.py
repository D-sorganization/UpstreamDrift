"""Guard the inertial parameter layout at every MOSAIC boundary (issue #12675).

Two orderings of the ten rigid-body parameters exist in the codebase:

* MOSAIC (``inertial.py``, ``mujoco_regressor.py``, ``humanoid_transfer.py``):
  ``[m, h_x, h_y, h_z, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]``;
* Pinocchio (``Inertia.toDynamicParameters`` and the columns of
  ``computeJointTorqueRegressor``, hence ``PinocchioRegressorModel``):
  ``[m, h_x, h_y, h_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]``.

Both express the inertia about the body-frame origin; only entries 6 and 7
differ.  The audit for #12675 found no live path that hands a Pinocchio-layout
vector to ``inertial.py``: ``PinocchioRegressorModel`` is consumed only in
planar mode, whose columns ``(0, 1, 2, 9)`` are identical in both layouts.
These tests pin each module's layout and fail when a new consumer couples the
Pinocchio provider to MOSAIC spatial algebra without a layout converter.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.humanoid_transfer import (
    skew3,
    spatial_payload_regressor,
)
from src.shared.python.estimation.mosaic.inertial import (
    is_spatial_consistent,
    parameters_from_pseudo_inertia,
    pseudo_inertia,
)
from src.shared.python.estimation.mosaic.mujoco_regressor import (
    MujocoRegressorModel,
    is_mujoco_available,
    mujoco_to_spatial_parameters,
)
from src.shared.python.estimation.mosaic.pinocchio_provider import (
    is_pinocchio_available,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
# Index permutation between the two layouts (it is its own inverse).
PINOCCHIO_TO_MOSAIC = (0, 1, 2, 3, 4, 5, 7, 6, 8, 9)
# Distinct entries so that any transposition of Ixz and Iyy is detectable.
INERTIA = np.array(
    [[0.30, -0.02, 0.05], [-0.02, 0.25, 0.03], [0.05, 0.03, 0.12]], dtype=np.float64
)
MASS = 2.0
FIRST_MOMENT = MASS * np.array([0.01, -0.02, 0.15], dtype=np.float64)


def _mosaic_vector(mass: float, moment: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    upper = [inertia[0, 0], inertia[0, 1], inertia[0, 2]]
    lower = [inertia[1, 1], inertia[1, 2], inertia[2, 2]]
    return np.array([mass, *moment, *upper, *lower], dtype=np.float64)


def _forearm_box_pinocchio_parameters() -> np.ndarray:
    """Pinocchio-layout parameters of a 1.2 kg, 0.45 m forearm-like box."""
    import pinocchio as pin

    box = pin.Inertia.FromBox(1.2, 0.10, 0.08, 0.45)
    lever = np.array([0.0, 0.0, 0.2], dtype=np.float64)
    return np.asarray(pin.Inertia(box.mass, lever, box.inertia).toDynamicParameters())


def test_inertial_pseudo_inertia_reads_mosaic_layout() -> None:
    """``pseudo_inertia`` places entry 6 at Ixz and entry 7 at Iyy."""
    pi = _mosaic_vector(MASS, FIRST_MOMENT, INERTIA)
    sigma = 0.5 * np.trace(INERTIA) * np.eye(3) - INERTIA

    pseudo = pseudo_inertia(pi[None])[0]

    np.testing.assert_allclose(pseudo[:3, :3], sigma, atol=1e-15)
    np.testing.assert_allclose(pseudo[:3, 3], FIRST_MOMENT, atol=1e-15)
    np.testing.assert_allclose(parameters_from_pseudo_inertia(pseudo[None])[0], pi)


def test_mujoco_spatial_parameters_use_mosaic_layout() -> None:
    """``mujoco_to_spatial_parameters`` emits the layout ``inertial.py`` reads."""
    angle = 0.7
    iquat = np.array([np.cos(angle / 2), 0.3, 0.5, 0.2], dtype=np.float64)
    iquat /= np.linalg.norm(iquat)
    com = np.array([0.02, -0.01, 0.18], dtype=np.float64)
    principal = np.array([0.04, 0.03, 0.01], dtype=np.float64)

    pi = mujoco_to_spatial_parameters(MASS, com, principal, iquat)
    sigma = pseudo_inertia(pi[None])[0, :3, :3]
    inertia_origin = np.trace(sigma) * np.eye(3) - sigma

    expected = _mosaic_vector(MASS, MASS * com, inertia_origin)
    np.testing.assert_allclose(pi, expected, atol=1e-14)
    assert inertia_origin[0, 2] != pytest.approx(inertia_origin[1, 1])


def test_mujoco_regressor_columns_use_mosaic_layout() -> None:
    """``MujocoRegressorModel`` columns pair with MOSAIC-layout parameters."""
    if not is_mujoco_available():
        pytest.skip("MuJoCo is not installed (unqualified report)")
    xml = """
<mujoco model="tilted_ball_body">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="segment">
      <joint name="ball" type="ball"/>
      <inertial pos="0.02 -0.01 0.18" mass="2.0"
                fullinertia="0.06 0.05 0.02 -0.004 0.006 0.003"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = MujocoRegressorModel.from_xml_string(xml)
    q = np.array([0.9, 0.2, -0.3, 0.25], dtype=np.float64)
    q /= np.linalg.norm(q)
    v = np.array([1.1, -0.7, 0.9], dtype=np.float64)
    a = np.array([-2.0, 1.5, 0.8], dtype=np.float64)

    regressor = model.compute_regressor(q, v, a)
    pi = model.spatial_parameters
    tau = model.inverse_dynamics(q, v, a)

    np.testing.assert_allclose(regressor @ pi, tau, atol=1e-6)
    swapped_error = np.abs(regressor @ pi[list(PINOCCHIO_TO_MOSAIC)] - tau).max()
    assert swapped_error > 1e-3
    assert bool(is_spatial_consistent(pi[None])[0])


def test_humanoid_payload_regressor_uses_mosaic_layout() -> None:
    """The payload regressor's torque rows read Ixz at column 6, Iyy at 7."""
    pi = _mosaic_vector(MASS, FIRST_MOMENT, INERTIA)
    accel = np.array([[0.4, -1.2, 2.0]], dtype=np.float64)
    omega = np.array([[1.3, -0.6, 0.9]], dtype=np.float64)
    alpha = np.array([[-0.5, 2.2, 0.7]], dtype=np.float64)
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    wrench = spatial_payload_regressor(accel, omega, alpha, gravity)[0] @ pi

    w, net = omega[0], accel[0] - gravity
    torque = -skew3(net) @ FIRST_MOMENT + INERTIA @ alpha[0]
    torque += np.cross(w, INERTIA @ w)
    np.testing.assert_allclose(wrench[3:], torque, atol=1e-12)


def test_pinocchio_regressor_columns_use_pinocchio_layout() -> None:
    """``PinocchioRegressorModel`` columns pair with ``toDynamicParameters``."""
    if not is_pinocchio_available():
        pytest.skip("Pinocchio is not installed (unqualified report)")
    import pinocchio as pin

    from src.shared.python.estimation.mosaic.pinocchio_provider import (
        PinocchioRegressorModel,
    )

    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelSpherical(), pin.SE3.Identity(), "ball")
    lever = FIRST_MOMENT / MASS
    about_com = INERTIA - MASS * pin.skew(lever).T @ pin.skew(lever)
    model.appendBodyToJoint(
        joint, pin.Inertia(MASS, lever, about_com), pin.SE3.Identity()
    )
    provider = PinocchioRegressorModel(model)

    q = pin.normalize(model, np.array([0.2, -0.3, 0.25, 0.9]))[None]
    v = np.array([[1.1, -0.7, 0.9]])
    a = np.array([[-2.0, 1.5, 0.8]])
    tau = pin.rnea(model, model.createData(), q[0], v[0], a[0])
    regressor = provider.regressor(q, v, a)[0]
    native = np.asarray(model.inertias[joint].toDynamicParameters())

    np.testing.assert_allclose(regressor @ native, tau, atol=1e-10)
    mosaic = _mosaic_vector(MASS, FIRST_MOMENT, INERTIA)
    np.testing.assert_allclose(native[list(PINOCCHIO_TO_MOSAIC)], mosaic, atol=1e-12)


def test_pinocchio_layout_flips_psd_verdict_of_a_real_body() -> None:
    """A consistent box reads as inconsistent when its layout is not converted."""
    if not is_pinocchio_available():
        pytest.skip("Pinocchio is not installed (unqualified report)")
    native = _forearm_box_pinocchio_parameters()

    assert not bool(is_spatial_consistent(native[None])[0])
    assert bool(is_spatial_consistent(native[list(PINOCCHIO_TO_MOSAIC)][None])[0])


_PINOCCHIO_SPATIAL_SOURCES = re.compile(
    r"PinocchioRegressorModel|computeJointTorqueRegressor|toDynamicParameters"
)
_MOSAIC_SPATIAL_ALGEBRA = re.compile(
    r"pseudo_inertia|is_spatial_consistent|project_spatial_consistent"
    r"|log_cholesky|SpatialLogCholeskyParameterization|check_physical_consistency"
)
_LAYOUT_CONVERTER = "swap_inertia_layout"


def test_no_module_mixes_pinocchio_parameters_with_mosaic_algebra() -> None:
    """A module that both produces Pinocchio-layout parameters and runs MOSAIC
    spatial algebra must convert through the shared ``swap_inertia_layout``.
    """
    offenders = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        mixes = _PINOCCHIO_SPATIAL_SOURCES.search(text) and (
            _MOSAIC_SPATIAL_ALGEBRA.search(text)
        )
        if mixes and _LAYOUT_CONVERTER not in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == [], (
        "Pinocchio orders inertia as Ixx, Ixy, Iyy, Ixz, Iyz, Izz but MOSAIC "
        "inertial.py expects Ixx, Ixy, Ixz, Iyy, Iyz, Izz; convert with "
        f"{_LAYOUT_CONVERTER} in: {offenders}"
    )
