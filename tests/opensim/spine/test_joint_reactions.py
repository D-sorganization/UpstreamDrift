"""Per-level joint reactions with muscle forces (SPN-10c, #12762).

Fixture design (stated honestly).  Every number is INVENTED.  The closed-form
tests use bare (muscle-free) fixtures and a zero-tilt ``FixtureSpec`` (no
lordosis, no sacral slope, no thorax tilt) so the spine is a vertical stack;
the tilted tests compare against a rigid-body sum built from the model's own
body poses and mass centres (independent of the joint-reaction code path).
Muscle tests use the ONE-LEVEL fixture (``L3_L4``) whose lumped muscles run
pelvis -> L3 -> thorax_proxy; tendon slack lengths are retuned test-locally so
no muscle is slack (see ``test_static_optimization``).  The expected change of
the reaction under a known muscle force is computed from the muscle path
points in ground (force ``F (u_next - u_prev)`` per interior point), so the
hand calculation does not use the code under test.

Reaction definition under test: the wrench on the SUPERIOR body BY the
INFERIOR body through the joint, including the reserve (residual) torque that
makes the accelerations equal the measured ones, excluding the muscles
(external forces on the bodies), in the inferior ISB frame about the joint
centre.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.musculoskeletal_swing import write_sto
from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    transform_to_matrix,
)
from src.engines.physics_engines.opensim.python.spine.drive import FIXTURE_PELVIS
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_body_names,
    fixture_level_names,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    load_level_maps,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    SpinalGeneralizedForces,
    SpineIDOptions,
    run_spinal_inverse_dynamics,
    spinal_generalized_forces,
)
from src.engines.physics_engines.opensim.python.spine.joint_reactions import (
    JointReactionOptions,
    compute_joint_reactions,
)
from src.engines.physics_engines.opensim.python.spine.static_optimization import (
    SpineMuscleBasis,
    SpineSOOptions,
    SpineStaticOptimizationResult,
    run_spine_static_optimization,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.loads import REQUIRED_PROVENANCE_KEYS, LoadObservable
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.requires_opensim, pytest.mark.integration]

_TIMES = np.linspace(0.0, 0.7, 8)
_FRAMES = (0, 3, 7)
_G = 9.80665
_ZERO_TILT = {
    "sacral_slope_deg": 0.0,
    "thorax_tilt_deg": 0.0,
}


def _zero_tilt_spec(levels=None) -> FixtureSpec:
    """Vertical stack: no lordosis, sacral slope or thorax tilt."""
    kwargs = {} if levels is None else {"levels": levels}
    probe = FixtureSpec(**kwargs)
    zero = dict.fromkeys(probe.levels, 0.0)
    return FixtureSpec(neutral_lordosis_deg=zero, **_ZERO_TILT, **kwargs)


# ------------------------------------------------------------------ helpers


def _retune(path):
    """Set tendon slack = neutral path length - optimal fibre length."""
    osim = require_opensim()
    model = osim.Model(str(path))
    state = model.initSystem()
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        slack = muscle.getLength(state) - muscle.get_optimal_fiber_length()
        muscle.set_tendon_slack_length(slack)
    model.printToXML(str(path))
    return path


def _columns(spec, angles, pelvis_ry=0.0) -> dict:
    cols = {n: np.zeros(_TIMES.size) for n in FIXTURE_PELVIS.rotation}
    cols.update({n: np.zeros(_TIMES.size) for n in FIXTURE_PELVIS.translation})
    cols[FIXTURE_PELVIS.rotation[1]] = np.full(_TIMES.size, pelvis_ry)
    for names in fixture_level_names(spec):
        values = angles.get(names.level, (0.0, 0.0, 0.0))
        for coord, v in zip(names.coordinates, values, strict=True):
            cols[coord] = np.broadcast_to(np.asarray(v, dtype=float), _TIMES.shape)
    return cols


def _static(spec, angle_by_level, tmp, pelvis_ry=0.0):
    cols = _columns(spec, angle_by_level, pelvis_ry)
    return write_sto(tmp / "static.mot", _TIMES, cols)


def _level_map(spec):
    full = load_level_maps()["fixture"]
    return replace(full, levels={lv: full.levels[lv] for lv in spec.levels})


def _efforts(path, spec, mot) -> SpinalGeneralizedForces:
    options = SpineIDOptions(pelvis=FIXTURE_PELVIS, level_map=_level_map(spec))
    return spinal_generalized_forces(path, mot, "fixture", options=options)


def _stub(path, spec, efforts, force_row) -> SpineStaticOptimizationResult:
    """SO-shaped result with prescribed muscle forces and consistent reserves."""
    frames = np.array(_FRAMES)
    basis_names: tuple[str, ...] = ()
    arms = np.zeros((0, 3 * len(spec.levels)))
    n_mus = 0
    model = require_opensim().Model(str(path))
    if model.getMuscles().getSize():
        basis = SpineMuscleBasis(path, "fixture", level_map=_level_map(spec))
        basis_names = basis.muscle_names
        n_mus = len(basis_names)
    force = np.zeros((frames.size, n_mus))
    muscle = np.zeros((frames.size, 3 * len(spec.levels)))
    for k, frame in enumerate(frames):
        if n_mus:
            force[k] = force_row
            _a, _p, arms = basis.evaluate(
                efforts.model_coordinates, efforts.q[frame], efforts.qd[frame]
            )
            muscle[k] = force[k] @ arms
    required = efforts.level_tau()[frames]
    provenance = {
        "model_id": "fixture",
        "model_sha256": "stub",
        "solver": "test stub",
        "cost_exponent": "3",
        "distribution_source": efforts.provenance["distribution_source"],
        "external_loads": efforts.provenance["external_loads"],
    }
    k = frames.size
    return SpineStaticOptimizationResult(
        time_s=efforts.time_s[frames],
        frame_index=frames,
        frame_labels=("stride",) * k,
        levels=efforts.levels,
        muscle_names=basis_names,
        coordinate_names=efforts.level_coordinates,
        activation=np.zeros((k, n_mus)),
        force_n=force,
        required_nm=required,
        muscle_moment_nm=muscle,
        reserve_nm=required - muscle,
        reserve_share=np.zeros(k),
        flagged=np.zeros(k, bool),
        flag_reasons=((),) * k,
        success=np.ones(k, bool),
        reserve_limits_nm=np.full(3 * len(spec.levels), 1e3),
        cost_exponent=3,
        provenance=provenance,
    )


def _reactions(path, spec, mot, force_row=None, *, efforts=None, so=None, **opts):
    efforts = efforts or _efforts(path, spec, mot)
    so = so or _stub(path, spec, efforts, force_row)
    options = JointReactionOptions(
        pelvis=FIXTURE_PELVIS, level_map=_level_map(spec), **opts
    )
    return compute_joint_reactions(path, mot, so, "fixture", options=options)


def _pose(path, names_values):
    model = require_opensim().Model(str(path))
    state = model.initSystem()
    coords = model.getCoordinateSet()
    for name, value in names_values.items():
        coords.get(name).setValue(state, float(value), False)
    model.realizeDynamics(state)
    return model, state


def _frame_pose(path, spec, mot_cols, index):
    values = {n: float(np.asarray(v)[index]) for n, v in mot_cols.items()}
    return _pose(path, values)


def _ground_rotation(model, state, body):
    return transform_to_matrix(
        model.getBodySet().get(body).getTransformInGround(state)
    )[:3, :3]


def _origin(model, state, body):
    return transform_to_matrix(
        model.getBodySet().get(body).getTransformInGround(state)
    )[:3, 3]


def _bodies_above(spec, names) -> list[str]:
    bodies = fixture_body_names(spec)
    return list(bodies[bodies.index(names.superior_body) :])


def _gravity_wrench(model, state, spec, names):
    """Wrench of the lower chain on the upper chain in ground (static)."""
    centre = _origin(model, state, names.superior_body)
    force, moment = np.zeros(3), np.zeros(3)
    g_vec = np.array([0.0, -_G, 0.0])
    for name in _bodies_above(spec, names):
        body = model.getBodySet().get(name)
        mat = transform_to_matrix(body.getTransformInGround(state))
        com = (
            mat[:3, :3] @ np.array([body.getMassCenter().get(i) for i in range(3)])
            + mat[:3, 3]
        )
        weight = body.getMass() * g_vec
        force -= weight
        moment -= np.cross(com - centre, weight)
    return force, moment


def _muscle_point_forces(model, state, muscle_name, tension):
    """``[(body, point_ground, force_on_point)]`` of one muscle path."""
    path = model.getMuscles().get(muscle_name).getGeometryPath()
    points = path.getCurrentPath(state)
    pos = [
        np.array([points.get(i).getLocationInGround(state).get(k) for k in range(3)])
        for i in range(points.getSize())
    ]
    bodies = [str(points.get(i).getBody().getName()) for i in range(points.getSize())]
    out = []
    for i, p in enumerate(pos):
        f = np.zeros(3)
        if i + 1 < len(pos):
            f += tension * (pos[i + 1] - p) / np.linalg.norm(pos[i + 1] - p)
        if i > 0:
            f += tension * (pos[i - 1] - p) / np.linalg.norm(pos[i - 1] - p)
        out.append((bodies[i], p, f))
    return out


# ------------------------------------------------------------- native setup


@pytest.fixture(scope="module")
def five_level_bare(tmp_path_factory):
    require_opensim()
    tmp = tmp_path_factory.mktemp("jr_five")
    return tmp


@pytest.fixture(scope="module")
def one_level(tmp_path_factory):
    require_opensim()
    spec = FixtureSpec(levels=(SpinalLevel.L3_L4,))
    tmp = tmp_path_factory.mktemp("jr_one")
    muscled = _retune(write_fixture_osim(spec, tmp / "muscled.osim", muscles=True))
    return spec, muscled, tmp


# ----------------------------------------------- (1) static gravity, closed form


def test_zero_tilt_compression_is_the_weight_above(five_level_bare) -> None:
    spec = _zero_tilt_spec((SpinalLevel.L5_S1, SpinalLevel.L4_L5, SpinalLevel.L3_L4))
    path = write_fixture_osim(spec, five_level_bare / "zt.osim")
    series = _reactions(path, spec, _static(spec, {}, five_level_bare), None)
    assert series.observable is LoadObservable.JOINT_REACTION
    masses = dict(spec.segment_mass_kg)
    masses["thorax_proxy"] = spec.thorax_proxy_mass_kg
    bodies = list(fixture_body_names(spec))
    for k, names in enumerate(fixture_level_names(spec)):
        above = sum(masses[b] for b in bodies[bodies.index(names.superior_body) :])
        np.testing.assert_allclose(series.compression_n()[:, k], above * _G, atol=1e-6)
        np.testing.assert_allclose(series.ap_shear_n()[:, k], 0.0, atol=1e-6)
        np.testing.assert_allclose(series.ml_shear_n()[:, k], 0.0, atol=1e-6)
        np.testing.assert_allclose(series.moment_nm[:, k, :], 0.0, atol=1e-6)


def test_tilted_static_matches_rigid_body_sum(five_level_bare) -> None:
    spec = FixtureSpec(levels=(SpinalLevel.L5_S1, SpinalLevel.L4_L5, SpinalLevel.L3_L4))
    path = write_fixture_osim(spec, five_level_bare / "tilt.osim")
    angles = {
        names.level: (0.10 + 0.02 * k, 0.05 - 0.01 * k, -0.08)
        for k, names in enumerate(fixture_level_names(spec))
    }
    mot = _static(spec, angles, five_level_bare)
    series = _reactions(path, spec, mot, None)
    cols = _columns(spec, angles)
    model, state = _frame_pose(path, spec, cols, 0)
    assert np.max(np.abs(series.moment_nm)) > 1e-3  # tilt makes it non-trivial
    assert np.max(np.abs(series.ap_shear_n())) > 1e-3
    for k, names in enumerate(fixture_level_names(spec)):
        force_g, moment_g = _gravity_wrench(model, state, spec, names)
        rot = _ground_rotation(model, state, names.inferior_body)
        for i in range(series.time_s.size):
            np.testing.assert_allclose(series.force_n[i, k], rot.T @ force_g, atol=1e-6)
            np.testing.assert_allclose(
                series.moment_nm[i, k], rot.T @ moment_g, atol=1e-6
            )


def test_rotating_the_model_about_ground_y_leaves_loads_unchanged(
    five_level_bare,
) -> None:
    spec = FixtureSpec(levels=(SpinalLevel.L5_S1, SpinalLevel.L4_L5))
    path = write_fixture_osim(spec, five_level_bare / "inv.osim")
    angles = {names.level: (0.12, 0.04, -0.06) for names in fixture_level_names(spec)}
    base = _reactions(path, spec, _static(spec, angles, five_level_bare), None)
    turned = _reactions(
        path, spec, _static(spec, angles, five_level_bare, pelvis_ry=0.7), None
    )
    np.testing.assert_allclose(turned.force_n, base.force_n, atol=1e-6)
    np.testing.assert_allclose(turned.moment_nm, base.moment_nm, atol=1e-6)


# --------------------------------------------- (2) a single muscle of known force


def test_single_muscle_changes_the_reaction_by_its_line_of_action(one_level) -> None:
    spec, path, tmp = one_level
    names = fixture_level_names(spec)[0]
    angles = {names.level: (0.10, 0.05, -0.08)}
    mot = _static(spec, angles, tmp)
    efforts = _efforts(path, spec, mot)
    basis = SpineMuscleBasis(path, "fixture", level_map=_level_map(spec))
    target = "erector_spinae_l"
    force = np.zeros(len(basis.muscle_names))
    force[basis.muscle_names.index(target)] = 100.0
    off = _reactions(path, spec, mot, np.zeros_like(force), efforts=efforts)
    on = _reactions(path, spec, mot, force, efforts=efforts)

    model, state = _frame_pose(path, spec, _columns(spec, angles), 0)
    above = set(_bodies_above(spec, names))
    centre = _origin(model, state, names.superior_body)
    f_above, m_above = np.zeros(3), np.zeros(3)
    for body, point, f in _muscle_point_forces(model, state, target, 100.0):
        if body in above:
            f_above += f
            m_above += np.cross(point - centre, f)
    assert np.linalg.norm(f_above) > 1.0  # the muscle really loads the joint
    rot = _ground_rotation(model, state, names.inferior_body)
    expect_force = rot.T @ -f_above
    expect_moment = rot.T @ -m_above
    for i in range(on.time_s.size):
        np.testing.assert_allclose(
            on.force_n[i, 0] - off.force_n[i, 0], expect_force, atol=1e-6
        )
        np.testing.assert_allclose(
            on.moment_nm[i, 0] - off.moment_nm[i, 0], expect_moment, atol=1e-6
        )
    # compression is +F_y: a muscle pulling the upper body down adds compression
    delta = on.compression_n() - off.compression_n()
    np.testing.assert_allclose(delta[:, 0], expect_force[1], atol=1e-6)
    assert expect_force[1] > 0.0


# ------------------------------------------------ (3) decomposition consistency


def test_effort_equals_reaction_plus_muscle_moment(one_level) -> None:
    spec, path, tmp = one_level
    names = fixture_level_names(spec)[0]
    ramp = lambda lo, hi: np.linspace(lo, hi, _TIMES.size)  # noqa: E731
    angles = {names.level: (ramp(0.10, 0.16), ramp(0.05, 0.02), ramp(-0.08, 0.0))}
    mot = write_sto(tmp / "dyn.mot", _TIMES, _columns(spec, angles))
    efforts = _efforts(path, spec, mot)
    options = SpineSOOptions(
        exponent=3,
        events={"address": 0.0, "top": 0.4, "impact": 0.7},
        level_map=_level_map(spec),
    )
    so = run_spine_static_optimization(path, efforts, "fixture", options=options)
    series = _reactions(path, spec, mot, efforts=efforts, so=so)
    effort, _ = run_spinal_inverse_dynamics(
        path,
        mot,
        "fixture",
        options=SpineIDOptions(pelvis=FIXTURE_PELVIS, level_map=_level_map(spec)),
    )
    cols = _columns(spec, angles)
    above = set(_bodies_above(spec, names))
    for row, frame in enumerate(so.frame_index):
        model, state = _frame_pose(path, spec, cols, int(frame))
        centre = _origin(model, state, names.superior_body)
        rot = _ground_rotation(model, state, names.inferior_body)
        moment = np.zeros(3)
        for m, name in enumerate(so.muscle_names):
            for body, point, f in _muscle_point_forces(
                model, state, name, so.force_n[row, m]
            ):
                if body in above:
                    moment += np.cross(point - centre, f)
        np.testing.assert_allclose(
            effort.moment_nm[frame, 0],
            series.moment_nm[row, 0] + rot.T @ moment,
            atol=1e-6,
        )
    assert np.max(np.abs(effort.moment_nm)) > 1.0


def test_inconsistent_reserves_are_refused(one_level) -> None:
    spec, path, tmp = one_level
    mot = _static(spec, {SpinalLevel.L3_L4: (0.1, 0.05, -0.08)}, tmp)
    efforts = _efforts(path, spec, mot)
    force = np.full(8, 50.0)
    so = _stub(path, spec, efforts, force)
    bad = replace(so, reserve_nm=so.reserve_nm + 0.5)
    with pytest.raises(ValueError, match="reserve"):
        _reactions(path, spec, mot, efforts=efforts, so=bad)


# ------------------------------------------- (4) provenance, contract, couplers


def test_provenance_and_contract(one_level) -> None:
    spec, path, tmp = one_level
    mot = _static(spec, {SpinalLevel.L3_L4: (0.1, 0.0, 0.0)}, tmp)
    efforts = _efforts(path, spec, mot)
    so = _stub(path, spec, efforts, np.zeros(8))
    series = _reactions(path, spec, mot, efforts=efforts, so=so)
    for key in REQUIRED_PROVENANCE_KEYS:
        assert series.provenance[key]
    assert series.provenance["cost_exponent"] == "3"
    assert "JointReaction" in series.provenance["solver"]
    assert series.provenance["external_loads"] == so.provenance["external_loads"]
    assert series.provenance["distribution_source"]
    assert series.provenance["couplers"].startswith(("disabled", "none present"))
    assert series.levels == (SpinalLevel.L3_L4,)
    np.testing.assert_array_equal(series.time_s, so.time_s)
    assert not series.force_n.flags.writeable
    with pytest.raises(TypeError, match="SpineStaticOptimizationResult"):
        compute_joint_reactions(path, mot, object(), "fixture")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JointReactionOptions"):
        compute_joint_reactions(path, mot, so, "fixture", options=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="muscle"):
        _reactions(
            path, spec, mot, efforts=efforts, so=replace(so, muscle_names=("x",))
        )


def test_couplers_are_disabled_for_the_reaction(tmp_path) -> None:
    from src.shared.python.spine.coupling import LinearCouplingLaw
    from src.shared.python.spine.levels import levels_between

    spec = _zero_tilt_spec()
    phi = ("trunk_flex_ext", "trunk_lat_bend", "trunk_axial_rot")
    mats = {
        lv: np.diag([0.2, 0.2, 0.1])
        for lv in levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
    }
    law = LinearCouplingLaw("invented", "1", mats, phi, "invented, SPN-10c test")
    path = write_fixture_osim(spec, tmp_path / "coupled.osim", coupling=law)
    series = _reactions(path, spec, _static(spec, {}, tmp_path), None)
    assert "disabled" in series.provenance["couplers"]
    assert "15" in series.provenance["couplers"]
    masses = dict(spec.segment_mass_kg)
    masses["thorax_proxy"] = spec.thorax_proxy_mass_kg
    bodies = list(fixture_body_names(spec))
    first = fixture_level_names(spec)[0]
    above = sum(masses[b] for b in bodies[bodies.index(first.superior_body) :])
    np.testing.assert_allclose(series.compression_n()[:, 0], above * _G, atol=1e-6)
