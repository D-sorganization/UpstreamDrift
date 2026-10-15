"""Spinal inverse dynamics on the invented fixture (SPN-10a, #12760).

All expected numbers come from an independent numpy/scipy rigid-body
composition of the fixture bodies, never from OpenSim.  Every fixture number
is invented.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
    ContactPoint,
    write_external_loads,
)
from src.engines.physics_engines.opensim.python.spine.drive import (
    FIXTURE_PELVIS,
    export_mot,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_base_offset,
    fixture_body_names,
    fixture_level_names,
    fixture_neutral_transforms,
    fixture_terminal_offset,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    ResidualBudget,
    SpineIDOptions,
    euler_rate_columns,
    prepare_model_for_id,
    run_spinal_inverse_dynamics,
)
from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.loads import REQUIRED_PROVENANCE_KEYS, LoadObservable
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.requires_opensim, pytest.mark.integration]

_THORAX_LENGTH = 0.25  # fixture constant (inertia/COM only)
_G = np.array([0.0, -GRAVITY_FLOAT, 0.0])


@pytest.fixture(scope="module", autouse=True)
def _osim():
    return require_opensim()


@pytest.fixture(scope="module")
def spec() -> FixtureSpec:
    return FixtureSpec()


@pytest.fixture(scope="module")
def model_path(spec, tmp_path_factory) -> Path:
    return write_fixture_osim(spec, tmp_path_factory.mktemp("id") / "fixture.osim")


# --------------------------------------------------- independent rigid bodies


def _rigid(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def _body_poses(spec, theta, pelvis=np.eye(4)) -> dict[str, np.ndarray]:
    """Ground poses of all fixture bodies for ISB angles ``theta[level]``."""
    neutral = fixture_neutral_transforms(spec)
    poses = {"pelvis": pelvis}
    poses[fixture_level_names(spec)[0].inferior_body] = pelvis @ fixture_base_offset(
        spec
    )
    for names in fixture_level_names(spec):
        rot = Rotation.from_euler("ZXY", theta[names.level]).as_matrix()
        poses[names.superior_body] = (
            poses[names.inferior_body] @ neutral[names.level] @ _rigid(rot, np.zeros(3))
        )
    last = fixture_level_names(spec)[-1].superior_body
    poses["thorax_proxy"] = poses[last] @ fixture_terminal_offset(spec)
    return poses


def _mass_length(spec, body: str) -> tuple[float, float]:
    if body == "thorax_proxy":
        return spec.thorax_proxy_mass_kg, _THORAX_LENGTH
    return spec.segment_mass_kg[body], spec.segment_length_m[body]


def _com(spec, poses, body: str) -> tuple[float, np.ndarray]:
    mass, length = _mass_length(spec, body)
    pose = poses[body]
    return mass, pose[:3, :3] @ np.array([0.0, length / 2.0, 0.0]) + pose[:3, 3]


def _support_moment(spec, poses, bodies, about: np.ndarray) -> np.ndarray:
    """Moment (ground frame) that supports gravity on ``bodies`` about a point."""
    total = np.zeros(3)
    for body in bodies:
        mass, com = _com(spec, poses, body)
        total -= np.cross(com - about, mass * _G)
    return total


def _theta(spec) -> dict:
    return {
        names.level: np.array([0.12 + 0.03 * k, 0.07 - 0.02 * k, -0.09 + 0.04 * k])
        for k, names in enumerate(fixture_level_names(spec))
    }


def _columns(spec, theta) -> dict[str, float]:
    cols = {}
    for names in fixture_level_names(spec):
        for coord, value in zip(names.coordinates, theta[names.level], strict=True):
            cols[coord] = float(value)
    for name in (*FIXTURE_PELVIS.rotation, *FIXTURE_PELVIS.translation):
        cols[name] = 0.0
    return cols


def _write_mot(path: Path, times: np.ndarray, cols: dict[str, np.ndarray]) -> Path:
    osim = import_opensim()
    table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    for name in cols:
        labels.append(name)
    table.setColumnLabels(labels)
    matrix = np.column_stack(list(cols.values()))
    for t, row in zip(times, matrix, strict=True):
        table.appendRow(float(t), osim.RowVector([float(v) for v in row]))
    table.addTableMetaDataString("inDegrees", "no")
    osim.STOFileAdapter.write(table, str(path))
    return path


def _static_mot(spec, theta, tmp_path, n: int = 8) -> Path:
    times = np.linspace(0.0, 0.7, n)
    base = _columns(spec, theta)
    cols = {k: np.full(n, v) for k, v in base.items()}
    return _write_mot(tmp_path / "static.mot", times, cols)


def _run(model_path, mot, **kw):
    return run_spinal_inverse_dynamics(
        model_path,
        mot,
        "fixture",
        options=SpineIDOptions(pelvis=FIXTURE_PELVIS),
        **kw,
    )


# ------------------------------------------------------------ static analytic


def test_static_level_efforts_are_analytic_gravity_moments(
    spec, model_path, tmp_path
) -> None:
    theta = _theta(spec)
    series, budget = _run(model_path, _static_mot(spec, theta, tmp_path))
    poses = _body_poses(spec, theta)
    bodies = fixture_body_names(spec)
    assert series.observable is LoadObservable.GENERALIZED_EFFORT
    assert series.levels == tuple(n.level for n in fixture_level_names(spec))
    assert np.all(series.force_n == 0.0)
    for k, names in enumerate(fixture_level_names(spec)):
        above = bodies[bodies.index(names.superior_body) :]
        centre = poses[names.superior_body][:3, 3]
        expected_ground = _support_moment(spec, poses, above, centre)
        expected = poses[names.inferior_body][:3, :3].T @ expected_ground
        np.testing.assert_allclose(series.moment_nm[:, k, :], expected, atol=1e-6)
    assert budget.generalized.shape == (8, 6)


def test_static_pelvis_residual_is_total_weight(spec, model_path, tmp_path) -> None:
    theta = _theta(spec)
    _series, budget = _run(model_path, _static_mot(spec, theta, tmp_path))
    poses = _body_poses(spec, theta)
    bodies = fixture_body_names(spec)
    total = sum(_mass_length(spec, b)[0] for b in bodies)
    weight = total * GRAVITY_FLOAT
    np.testing.assert_allclose(budget.force_n[:, 1], weight, atol=1e-6)
    np.testing.assert_allclose(budget.force_n[:, [0, 2]], 0.0, atol=1e-6)
    expected_m = _support_moment(spec, poses, bodies, np.zeros(3))
    np.testing.assert_allclose(budget.moment_nm, np.tile(expected_m, (8, 1)), atol=1e-6)
    assert budget.body_weight_n == pytest.approx(weight, abs=1e-9)
    assert budget.peak_force_n == pytest.approx(weight, abs=1e-6)
    assert budget.rms_force_n == pytest.approx(weight, abs=1e-6)
    assert budget.force_ratio == pytest.approx(1.0, abs=1e-9)
    assert budget.passed is False  # gravity only: nothing carries the weight


def test_external_load_cancels_pelvis_residual_not_level_efforts(
    spec, model_path, tmp_path
) -> None:
    theta = _theta(spec)
    mot = _static_mot(spec, theta, tmp_path)
    plain, _ = _run(model_path, mot)
    poses = _body_poses(spec, theta)
    bodies = fixture_body_names(spec)
    total = sum(_mass_length(spec, b)[0] for b in bodies)
    com = sum(_com(spec, poses, b)[0] * _com(spec, poses, b)[1] for b in bodies) / total
    times = np.linspace(-0.1, 0.9, 11)
    forces = np.zeros((11, 1, 3))
    forces[:, 0, 1] = total * GRAVITY_FLOAT
    point = ContactPoint("support", "pelvis", tuple(float(v) for v in com))
    xml = write_external_loads(tmp_path / "ext", times, forces, (point,))
    series, budget = _run(model_path, mot, external_loads=xml)
    np.testing.assert_allclose(series.moment_nm, plain.moment_nm, atol=1e-6)
    np.testing.assert_allclose(budget.generalized, 0.0, atol=1e-6)
    assert budget.passed is True
    text = series.provenance["external_loads"]
    assert "support" in text and "included" in text and xml.name in text
    assert "excluded" not in text


# --------------------------------------------------------- dynamic vs manual


def _dyn_curves(spec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    times = np.linspace(0.0, 0.8, 41)
    names = [c for n in fixture_level_names(spec) for c in n.coordinates]
    names += [*FIXTURE_PELVIS.rotation, *FIXTURE_PELVIS.translation]
    q = np.zeros((times.size, len(names)))
    for j in range(len(names)):
        amp = 0.15 if j < 15 else (0.2 if j < 18 else 0.05)
        q[:, j] = amp * np.sin(2.0 * math.pi * (0.9 + 0.13 * j) * times + 0.3 * j)
    return times, q, names


def _manual_tau(model_path, spline, names, t, pelvis_u) -> np.ndarray:
    """M(q) udot + (C + G) with M from the matter subsystem."""
    osim = import_opensim()
    model = osim.Model(str(model_path))
    state = model.initSystem()
    coords = model.getCoordinateSet()
    q, qd, qdd = (spline(t, k) for k in (0, 1, 2))
    state.setTime(t)
    for j, name in enumerate(names):
        coords.get(name).setValue(state, float(q[j]), False)
    nrot = len(FIXTURE_PELVIS.rotation)
    omega, alpha = pelvis_u
    for j, name in enumerate(names):
        speed = qd[j]
        if name in FIXTURE_PELVIS.rotation:
            speed = omega[FIXTURE_PELVIS.rotation.index(name)]
        coords.get(name).setSpeedValue(state, float(speed))
    assert nrot == 3
    model.realizeDynamics(state)
    udot = np.zeros(state.getNU())
    for j, name in enumerate(names):
        acc = qdd[j]
        if name in FIXTURE_PELVIS.rotation:
            acc = alpha[FIXTURE_PELVIS.rotation.index(name)]
        coords.get(name).setSpeedValue(state, coords.get(name).getSpeedValue(state))
        udot[_u_index(model, state, name)] = acc
    solver = osim.InverseDynamicsSolver(model)
    tau0 = solver.solve(state, osim.Vector(state.getNU(), 0.0))
    mass = osim.Matrix()
    model.getMatterSubsystem().calcM(state, mass)
    m_np = np.array(
        [[mass.get(i, j) for j in range(state.getNU())] for i in range(state.getNU())]
    )
    base = np.array([tau0.get(i) for i in range(state.getNU())])
    return m_np @ udot + base


def _u_index(model, state, name: str) -> int:
    coords = model.getCoordinateSet()
    saved = np.array([state.getU().get(i) for i in range(state.getNU())])
    probe = state.updU()
    for i in range(state.getNU()):
        probe.set(i, 0.0)
    coords.get(name).setSpeedValue(state, 1.0)
    index = next(i for i in range(state.getNU()) if state.getU().get(i) == 1.0)
    for i, value in enumerate(saved):
        state.updU().set(i, float(value))
    return index


def _omega_alpha(spline, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Ground-frame angular velocity/acceleration by 4th-order differences."""
    h = 2e-3

    def rot(tt: float) -> np.ndarray:
        return Rotation.from_euler("XYZ", spline(tt)[15:18]).as_matrix()

    def omega(tt: float) -> np.ndarray:
        dr = (
            8.0 * (rot(tt + h) - rot(tt - h)) - (rot(tt + 2 * h) - rot(tt - 2 * h))
        ) / (12.0 * h)
        w = dr @ rot(tt).T
        return np.array([w[2, 1], w[0, 2], w[1, 0]])

    alpha = (
        8.0 * (omega(t + h) - omega(t - h)) - (omega(t + 2 * h) - omega(t - 2 * h))
    ) / (12.0 * h)
    return omega(t), alpha


def test_dynamic_motion_matches_manual_mass_matrix_balance(
    spec, model_path, tmp_path
) -> None:
    times, q, names = _dyn_curves(spec)
    mot = _write_mot(
        tmp_path / "dyn.mot", times, {n: q[:, j] for j, n in enumerate(names)}
    )
    series, budget = _run(model_path, mot)

    class _S:
        def __init__(self) -> None:
            self.cs = CubicSpline(times, q, axis=0)

        def __call__(self, t, order=0):
            return self.cs(t, order)

    spline = _S()
    for i in (3, 17, 28, 36):
        t = float(times[i])
        tau = _manual_tau(model_path, spline, names, t, _omega_alpha(spline, t))
        # Levels: tau_theta = J^T M_P with M_P = R_neutral^T M_inferior.
        neutral = fixture_neutral_transforms(spec)
        for k, lv in enumerate(fixture_level_names(spec)):
            th = spline(t)[3 * k : 3 * k + 3]
            ez = np.array([0.0, 0.0, 1.0])
            rz = Rotation.from_euler("Z", th[0]).as_matrix()
            rzx = rz @ Rotation.from_euler("X", th[1]).as_matrix()
            jac = np.column_stack(
                [ez, rz @ np.array([1.0, 0, 0]), rzx @ np.array([0, 1.0, 0])]
            )
            m_p = neutral[lv.level][:3, :3].T @ series.moment_nm[i, k]
            idx = [_u_index_cache(model_path, n) for n in lv.coordinates]
            np.testing.assert_allclose(jac.T @ m_p, tau[idx], atol=1e-6)
        # Pelvis: u = (omega, v) -> wrench is the pelvis generalized force.
        np.testing.assert_allclose(budget.moment_nm[i], tau[0:3], atol=1e-6)
        np.testing.assert_allclose(budget.force_n[i], tau[3:6], atol=1e-6)


_U_CACHE: dict = {}


def _u_index_cache(model_path, name: str) -> int:
    if model_path not in _U_CACHE:
        osim = import_opensim()
        model = osim.Model(str(model_path))
        state = model.initSystem()
        _U_CACHE[model_path] = {
            c.getName(): _u_index(model, state, c.getName())
            for c in (
                model.getCoordinateSet().get(i)
                for i in range(model.getCoordinateSet().getSize())
            )
        }
    return _U_CACHE[model_path][name]


def test_euler_rate_columns_match_rotation_differences() -> None:
    def q(t: float) -> np.ndarray:
        return np.array([0.3 * math.sin(2 * t), 0.2 * math.cos(3 * t), 0.4 * t])

    t0, h = 0.37, 1e-4
    qd = np.array([0.6 * math.cos(2 * t0), -0.6 * math.sin(3 * t0), 0.4])
    qdd = np.array([-1.2 * math.sin(2 * t0), -1.8 * math.cos(3 * t0), 0.0])
    omega, alpha = euler_rate_columns(q(t0), qd, qdd, "XYZ")

    def rot(t):
        return Rotation.from_euler("XYZ", q(t)).as_matrix()

    def w(t):
        dr = (rot(t + h) - rot(t - h)) / (2 * h)
        m = dr @ rot(t).T
        return np.array([m[2, 1], m[0, 2], m[1, 0]])

    np.testing.assert_allclose(omega, w(t0), atol=1e-7)
    np.testing.assert_allclose(alpha, (w(t0 + h) - w(t0 - h)) / (2 * h), atol=1e-6)


# ----------------------------------------------------- couplers & provenance


def test_couplers_are_disabled_in_the_run_model(spec, tmp_path) -> None:
    from src.shared.python.spine.coupling import LinearCouplingLaw
    from src.shared.python.spine.levels import SpinalLevel, levels_between

    phi = ("trunk_flex_ext", "trunk_lat_bend", "trunk_axial_rot")
    mats = {
        lv: np.diag([0.2, 0.2, 0.1]) for lv in levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
    }
    law = LinearCouplingLaw("invented", "1", mats, phi, "invented, SPN-10a test")
    path = write_fixture_osim(spec, tmp_path / "coupled.osim", coupling=law)
    kept = prepare_model_for_id(path, disable_couplers=False)
    assert kept.disabled_constraints == ()
    assert kept.n_constraints == 15
    assert all(c for c in kept.enforced_flags())
    prepared = prepare_model_for_id(path, disable_couplers=True)
    assert len(prepared.disabled_constraints) == 15
    assert prepared.n_constraints == 15
    assert not any(prepared.enforced_flags())
    theta = _theta(spec)
    series, _ = run_spinal_inverse_dynamics(
        path,
        _static_mot(spec, theta, tmp_path),
        "fixture",
        options=SpineIDOptions(pelvis=FIXTURE_PELVIS),
    )
    assert "15" in series.provenance["couplers"]
    assert "disabled" in series.provenance["couplers"]
    poses = _body_poses(spec, theta)
    k = 0
    bodies = fixture_body_names(spec)
    names = fixture_level_names(spec)[k]
    expected = poses[names.inferior_body][:3, :3].T @ _support_moment(
        spec,
        poses,
        bodies[bodies.index(names.superior_body) :],
        poses[names.superior_body][:3, 3],
    )
    np.testing.assert_allclose(series.moment_nm[0, k], expected, atol=1e-6)


def test_provenance_is_complete_and_explicit(spec, model_path, tmp_path) -> None:
    theta = _theta(spec)
    series, _ = _run(model_path, _static_mot(spec, theta, tmp_path), filter_hz=6.0)
    prov = series.provenance
    for key in REQUIRED_PROVENANCE_KEYS:
        assert prov[key].strip()
    assert prov["external_loads"] == "none (excluded): gravity only"
    assert prov["model_id"] == "fixture"
    assert len(prov["model_sha256"]) == 64
    assert prov["filter_hz"] == "6.0"
    assert "CubicSpline" in prov["derivatives"]
    assert prov["units"] == "N*m"
    plain, _ = _run(model_path, _static_mot(spec, theta, tmp_path))
    assert plain.provenance["filter_hz"] == "none"


def test_budget_thresholds_are_declared_and_applied(spec, model_path, tmp_path) -> None:
    theta = _theta(spec)
    loose = ResidualBudget(force_fraction_bw=2.0, moment_fraction_bw_height=2.0)
    _s, budget = run_spinal_inverse_dynamics(
        model_path,
        _static_mot(spec, theta, tmp_path),
        "fixture",
        options=SpineIDOptions(pelvis=FIXTURE_PELVIS, budget=loose),
    )
    assert budget.passed is True
    assert budget.budget is loose


@pytest.mark.parametrize(
    "bad",
    [
        {"filter_hz": -1.0},
        {"filter_hz": 0.0},
        {"external_loads": "does_not_exist.xml"},
    ],
)
def test_invalid_arguments_raise(spec, model_path, tmp_path, bad) -> None:
    mot = _static_mot(spec, _theta(spec), tmp_path)
    with pytest.raises(ValueError):
        _run(model_path, mot, **bad)


def test_missing_coordinate_column_raises(spec, model_path, tmp_path) -> None:
    times = np.linspace(0.0, 0.5, 8)
    base = _columns(spec, _theta(spec))
    base.pop("l5_s1_flex_ext")
    mot = _write_mot(
        tmp_path / "missing.mot", times, {k: np.full(8, v) for k, v in base.items()}
    )
    with pytest.raises(ValueError, match="l5_s1_flex_ext"):
        _run(model_path, mot)


def test_wrong_types_raise(model_path, tmp_path) -> None:
    with pytest.raises(TypeError):
        run_spinal_inverse_dynamics(model_path, 3.0, "fixture")
    with pytest.raises(TypeError):
        run_spinal_inverse_dynamics(model_path, tmp_path / "x.mot", "fixture", options=1)
