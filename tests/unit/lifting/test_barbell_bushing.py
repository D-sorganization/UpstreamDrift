"""Barbell held at both hands through OSV-7 bushings (LIFT-4, #11744).

Tolerances are fixed before measuring and are not loosened:

* Static hold (issue #11744 tests): the hand forces carry the bar weight
  within 1 %, a symmetric grip splits evenly within 1e-6 of the bar weight,
  bushing deflection at most 3 mm.
* Newton's third law: the bar load reaching the unactuated root equals the
  negative sum of the hand forces within 1e-6 N.
* Finite-difference hand Jacobian against MuJoCo's analytic ``mj_jac``:
  1e-6 (m/rad, rad/rad).
* Cross-engine parity: acceleration-parity level L1 of epic #11605,
  ``|dX| <= 1e-6 + 1e-8 * max|X|`` per quantity, as in GCV-8 and #12341.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.lifting.barbell_bushing import (
    BAR_COORDINATES,
    Barbell,
    BarbellBushingPlant,
    _n_matrix_rate,
    barbell_dynamics,
    barbell_interface,
)
from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE
from src.shared.python.motion_matching.same_input import InputBundle, VectorPlant
from tests.support.real_engines import real_module

pytestmark = pytest.mark.unit

pytest.importorskip("mujoco")

HOLD_FRACTION = 0.01
SPLIT_FRACTION = 1e-6
MAX_DEFLECTION_M = 3e-3
THIRD_LAW_N = 1e-6
JACOBIAN_TOL = 1e-6
ROOT_SLIDES = ("TranslationInputX", "TranslationInputY", "TranslationInputZ")


def l1_tolerance(scale: float) -> float:
    return 1e-6 + 1e-8 * scale


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return InputBundle.load(BACK_SQUAT_BUNDLE)


@pytest.fixture(scope="module")
def loaded(bundle: InputBundle) -> BarbellBushingPlant:
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    return BarbellBushingPlant(plant, bundle.spec_bytes, Barbell(100.0), bundle.q0)


def _gravity(bundle: InputBundle) -> float:
    return float(np.linalg.norm(json.loads(bundle.spec_bytes)["gravity_m_s2"]))


# -- Barbell value object ----------------------------------------------------
def test_barbell_inertia_follows_iwf_geometry() -> None:
    bar = Barbell(100.0)
    assert bar.plate_mass_kg == pytest.approx(80.0)
    assert bar.axial_inertia_kg_m2 == pytest.approx(0.5 * 20.0 * 0.014**2)
    expected = 20.0 * 2.2**2 / 12.0 + 80.0 * (0.70**2 + 0.25 * 0.225**2)
    assert bar.transverse_inertia_kg_m2 == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"load_kg": -1.0}, ValueError),
        ({"load_kg": 10.0}, ValueError),
        ({"load_kg": float("nan")}, TypeError),
        ({"load_kg": "100"}, TypeError),
        ({"load_kg": 100.0, "plate_offset_m": 0.0}, ValueError),
    ],
)
def test_barbell_rejects_bad_inputs(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        Barbell(**kwargs)


def test_barbell_body_sits_midway_on_the_grip_axis(bundle: InputBundle) -> None:
    spec = json.loads(bundle.spec_bytes)
    bar = Barbell(100.0)
    interface = barbell_interface(spec, bar)
    dynamics = barbell_dynamics(bar, interface)
    left = np.asarray(interface.left.position_m)
    right = np.asarray(interface.right.position_m)
    np.testing.assert_allclose(dynamics.com_m, 0.5 * (left + right), atol=1e-15)
    axis = np.asarray(interface.left.rotation)[:, 0]
    assert axis @ dynamics.inertia_com_kg_m2 @ axis == pytest.approx(
        bar.axial_inertia_kg_m2
    )
    assert dynamics.mass_kg == bar.load_kg


def test_n_matrix_rate_matches_finite_difference() -> None:
    from src.shared.python.grip_contact.bushing_law import body_xyz_n_matrix

    angles = np.array([0.3, -0.4, 0.7])
    rates = np.array([1.1, -0.6, 0.9])
    h = 1e-6
    numeric = (
        body_xyz_n_matrix(angles + h * rates) - body_xyz_n_matrix(angles - h * rates)
    ) / (2 * h)
    np.testing.assert_allclose(_n_matrix_rate(angles, rates), numeric, atol=1e-8)


# -- Coupled plant -------------------------------------------------------------
def test_coordinates_append_the_bar(loaded: BarbellBushingPlant) -> None:
    assert loaded.coordinate_order[-6:] == BAR_COORDINATES
    q = np.concatenate([np.zeros(loaded.nv_body), loaded.attached_bar_q])
    rows = loaded.closure_rate_matrix(q)
    assert rows.shape == (6, loaded.nv_body + 6)
    assert not rows[:, -6:].any()


def test_attached_bar_is_unloaded_and_falls_freely(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    q, v = loaded.attached_state(bundle.q0, np.zeros_like(bundle.q0), hanging=False)
    sample = loaded.sample(q, v)
    for side in ("L", "R"):
        assert np.abs(sample.wrench[side].force_n).max() <= 1e-6
    tau = np.concatenate([bundle.efforts[0], np.zeros(6)])
    acceleration = loaded.acceleration(q, v, tau)
    g = np.asarray(json.loads(bundle.spec_bytes)["gravity_m_s2"])
    np.testing.assert_allclose(acceleration[-6:-3], g, atol=1e-9)
    plain = VectorPlant("mujoco", bundle.spec_bytes).acceleration(
        bundle.q0, np.zeros_like(bundle.q0), bundle.efforts[0]
    )
    scale = float(np.abs(plain).max())
    assert np.abs(acceleration[:-6] - plain).max() <= l1_tolerance(scale)


def test_bar_efforts_must_be_zero(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    q, v = loaded.attached_state(bundle.q0, np.zeros_like(bundle.q0))
    tau = np.zeros(loaded.nv_body + 6)
    tau[-1] = 1.0
    with pytest.raises(ValueError, match="unactuated"):
        loaded.acceleration(q, v, tau)


def _static_state(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> tuple[np.ndarray, np.ndarray]:
    return loaded.attached_state(bundle.q0, np.zeros_like(bundle.q0))


def test_hanging_offset_is_the_static_deflection(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    q, _ = _static_state(loaded, bundle)
    stiffness = loaded.interface.bushing.translational_stiffness_n_m[0]
    drop = loaded.barbell.load_kg * _gravity(bundle) / (2.0 * stiffness)
    offset = q[-6:-3] - loaded.attached_bar_q[:3]
    np.testing.assert_allclose(offset, [0.0, 0.0, -drop], atol=1e-15)


def test_attached_bar_moves_rigidly_with_the_hands(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    q, v = loaded.attached_state(bundle.q0, bundle.v0, hanging=False)
    sample = loaded.sample(q, v)
    for side in ("L", "R"):
        assert np.abs(sample.wrench[side].translation_rate_m_s).max() <= 1e-9
        assert np.abs(sample.wrench[side].angle_rates_rad_s).max() <= 1e-9


def test_static_hold_hands_carry_the_bar_weight(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    q, v = _static_state(loaded, bundle)
    sample = loaded.sample(q, v)
    weight = loaded.barbell.load_kg * _gravity(bundle)
    left, right = (sample.wrench[s].force_n for s in ("L", "R"))
    assert abs((left + right)[2] - weight) <= HOLD_FRACTION * weight
    assert abs(left[2] - right[2]) <= SPLIT_FRACTION * weight
    assert sample.max_deflection_m() <= MAX_DEFLECTION_M
    tau = np.concatenate([bundle.efforts[0], np.zeros(6)])
    assert np.abs(loaded.acceleration(q, v, tau)[-6:]).max() <= 1e-6
    analysis = sample.grip_analysis()
    assert analysis.split_method == "bushing"
    assert analysis.net_force_n is not None
    assert analysis.net_force_n[2] == pytest.approx((left + right)[2], abs=1e-9)


def test_bar_load_reaches_the_root(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    """Newton's third law: the root slides feel minus the hand forces."""
    q, v = _static_state(loaded, bundle)
    load = loaded.hand_generalized_force(q, v)
    sample = loaded.sample(q, v)
    on_body = -(sample.wrench["L"].force_n + sample.wrench["R"].force_n)
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    order = list(loaded.body_order)
    axes = []
    for name in ROOT_SLIDES:
        step = np.zeros(loaded.nv_body)
        step[order.index(name)] = 1e-3
        plus = plant.kinematic_frames(bundle.q0 + step)["Hub"][:3, 3]
        minus = plant.kinematic_frames(bundle.q0 - step)["Hub"][:3, 3]
        axes.append((plus - minus) / 2e-3)
    expected = np.array(axes) @ on_body
    actual = np.array([load[order.index(n)] for n in ROOT_SLIDES])
    assert np.abs(actual - expected).max() <= THIRD_LAW_N


def test_weight_feedforward_equals_the_static_hand_load(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    q, v = _static_state(loaded, bundle)
    feedforward = loaded.weight_generalized_force(q[: loaded.nv_body])
    measured = loaded.hand_generalized_force(q, v)
    assert np.abs(feedforward - measured).max() <= THIRD_LAW_N


def test_hand_jacobian_matches_mujoco_analytic(
    loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    q = bundle.reference_q[100]
    numeric = loaded._hand_jacobian(q)  # noqa: SLF001
    pose = loaded._hand_pose(q)  # noqa: SLF001
    adapter = NativeMujocoFullBodyModel(bundle.spec_bytes)
    names = list(adapter.coordinate_order)
    adapter.frame_poses(dict(zip(names, map(float, q), strict=True)))
    model, data = adapter.model, adapter.data
    mujoco.mj_comPos(model, data)
    site = model.site(adapter.metadata["frame_sites"]["Grip"])
    body = int(site.bodyid[0])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, pose[:3, 3], body)
    dofs = [int(model.jnt_dofadr[model.joint(n).id]) for n in names]
    analytic = np.vstack([jacp[:, dofs], jacr[:, dofs]])
    assert np.abs(numeric - analytic).max() <= JACOBIAN_TOL


@pytest.mark.parametrize("engine", ["drake", "pinocchio"])
def test_cross_engine_parity_at_a_loaded_state(
    engine: str, loaded: BarbellBushingPlant, bundle: InputBundle
) -> None:
    real_module("pydrake" if engine == "drake" else "pinocchio")
    other = BarbellBushingPlant(
        VectorPlant(engine, bundle.spec_bytes),
        bundle.spec_bytes,
        Barbell(100.0),
        bundle.q0,
    )
    q, v = _static_state(loaded, bundle)
    q[: loaded.nv_body] = bundle.reference_q[60]
    v[: loaded.nv_body] = bundle.reference_v[60]
    tau = np.concatenate([bundle.efforts[60], np.zeros(6)])
    a, b = loaded.acceleration(q, v, tau), other.acceleration(q, v, tau)
    for part in (slice(0, loaded.nv_body), slice(loaded.nv_body, None)):
        scale = float(np.abs(a[part]).max())
        assert np.abs(a[part] - b[part]).max() <= l1_tolerance(scale)
    sa, sb = loaded.sample(q, v), other.sample(q, v)
    for side in ("L", "R"):
        for field in ("force_n", "moment_nm"):
            x = getattr(sa.wrench[side], field)
            y = getattr(sb.wrench[side], field)
            assert np.abs(x - y).max() <= l1_tolerance(float(np.abs(x).max()))
