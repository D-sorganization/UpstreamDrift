"""Trunk static optimisation on the invented fixture (SPN-10b, #12761).

Fixture design (stated honestly).  The eight lumped SPN-7b muscles span all
15 rotational DOFs of the default five-level fixture, so with 8 actuators the
ID moments are in general NOT reachable there.  The tests therefore use

* a ONE-LEVEL variant (``FixtureSpec(levels=(L3_L4,))``): the muscles run
  pelvis -> L3 -> thorax_proxy and only the three ``L3_L4`` DOFs lie between
  pelvis and L3 (the other segments are welds), so 8 muscles act on 3 DOFs and
  the gravity ID moments are reachable with redundancy 5; and
* the default FIVE-LEVEL fixture, where reserves must carry exactly the
  weighted least-squares remainder (KKT-checked) and the frame is flagged.

The SPN-7b tendon slack lengths leave several muscles slack (zero active
force) on these geometries, so ``_retune`` sets each tendon slack length to
``path length - optimal fibre length`` at the neutral pose (invented values,
test-local; the SPN-7b generator is unchanged).  Every number is invented.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.musculoskeletal_swing import write_sto
from src.engines.physics_engines.opensim.python.spine.drive import FIXTURE_PELVIS
from src.engines.physics_engines.opensim.python.spine.fixture_coupling import (
    fixture_muscle_names,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_level_names,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    load_level_maps,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    SpinalGeneralizedForces,
    SpineIDOptions,
    spinal_generalized_forces,
)
from src.engines.physics_engines.opensim.python.spine.static_optimization import (
    ReserveSpec,
    SpineMuscleBasis,
    SpineSOOptions,
    SpineStaticOptimizationResult,
    flag_frame,
    reserve_share,
    run_spine_static_optimization,
    select_frames,
)
from src.engines.physics_engines.opensim.python.static_opt_core import (
    participation_ratio,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.loads import REQUIRED_PROVENANCE_KEYS
from tests.helpers.opensim_support import require_opensim

_TIMES = np.linspace(0.0, 0.7, 8)
_EVENTS = {"address": 0.0, "top": 0.4, "impact": 0.7}


# ----------------------------------------------------------- pure helpers


@pytest.mark.unit
def test_select_frames_events_and_stride() -> None:
    sel = select_frames(_TIMES, events={"top": 0.41, "address": 0.0}, stride=3)
    assert sel.indices.tolist() == [0, 3, 4, 6]
    assert sel.labels == ("address", "stride", "top", "stride")
    sel = select_frames(_TIMES, events=_EVENTS)
    assert sel.indices.tolist() == [0, 4, 7]
    assert sel.labels == ("address", "top", "impact")
    assert select_frames(_TIMES).indices.tolist() == list(range(8))


@pytest.mark.unit
def test_select_frames_contract() -> None:
    with pytest.raises(ValueError, match="max_frames"):
        select_frames(_TIMES, max_frames=5)
    with pytest.raises(ValueError, match="outside"):
        select_frames(_TIMES, events={"impact": 2.0})
    with pytest.raises(ValueError, match="stride"):
        select_frames(_TIMES, stride=0)
    with pytest.raises(TypeError, match="events"):
        select_frames(_TIMES, events=[0.1])  # type: ignore[arg-type]


@pytest.mark.unit
def test_reserve_share_and_flags() -> None:
    tau = np.array([10.0, -5.0, 5.0])
    assert reserve_share(tau, np.zeros(3)) == 0.0
    assert reserve_share(tau, np.array([1.0, -1.0, 0.0])) == pytest.approx(0.1)
    assert reserve_share(np.zeros(3), np.zeros(3)) == 0.0
    assert reserve_share(np.zeros(3), np.array([1e-3, 0, 0])) == float("inf")
    limits = np.array([2.0, 2.0, 2.0])
    assert flag_frame(tau, np.zeros(3), limits, 0.05, success=True) == ()
    reasons = flag_frame(tau, np.array([3.0, 0, 0]), limits, 0.5, success=True)
    assert len(reasons) == 1 and "limit" in reasons[0]
    reasons = flag_frame(tau, np.array([1.0, 1.0, 0]), limits, 0.05, success=False)
    assert any("share" in r for r in reasons)
    assert any("solver" in r for r in reasons)


@pytest.mark.unit
def test_options_contract() -> None:
    with pytest.raises(ValueError, match="exponent"):
        SpineSOOptions(exponent=4)
    with pytest.raises(ValueError, match="max_share"):
        ReserveSpec(max_share=0.0)
    with pytest.raises(ValueError, match="limit_nm"):
        ReserveSpec(limit_nm=-1.0)
    spec = ReserveSpec(limit_nm=2.0, per_coordinate={"b": 5.0})
    np.testing.assert_array_equal(spec.limits(("a", "b")), [2.0, 5.0])
    with pytest.raises(TypeError, match="SpinalGeneralizedForces"):
        run_spine_static_optimization("x.osim", object(), "fixture")  # type: ignore[arg-type]


# ----------------------------------------------------------- native fixtures


def _retune(path: Path) -> Path:
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


def _columns(spec: FixtureSpec, angles: dict[SpinalLevel, np.ndarray]) -> dict:
    cols = {n: np.zeros(_TIMES.size) for n in FIXTURE_PELVIS.rotation}
    cols.update({n: np.zeros(_TIMES.size) for n in FIXTURE_PELVIS.translation})
    for names in fixture_level_names(spec):
        for coord, values in zip(names.coordinates, angles[names.level], strict=True):
            cols[coord] = np.asarray(values, dtype=float)
    return cols


def _efforts(path, spec, angles, tmp, level_map, **kw) -> SpinalGeneralizedForces:
    mot = write_sto(tmp / "motion.mot", _TIMES, _columns(spec, angles))
    return spinal_generalized_forces(
        path,
        mot,
        "fixture",
        options=SpineIDOptions(pelvis=FIXTURE_PELVIS, level_map=level_map),
        **kw,
    )


@pytest.fixture(scope="module")
def one_level(tmp_path_factory):
    require_opensim()
    spec = FixtureSpec(levels=(SpinalLevel.L3_L4,))
    tmp = tmp_path_factory.mktemp("so_one")
    muscled = _retune(write_fixture_osim(spec, tmp / "muscled.osim", muscles=True))
    bare = write_fixture_osim(spec, tmp / "bare.osim")
    full = load_level_maps()["fixture"]
    level_map = replace(
        full, levels={SpinalLevel.L3_L4: full.levels[SpinalLevel.L3_L4]}
    )
    return spec, muscled, bare, level_map, tmp


def _ramp(lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, _TIMES.size)


def _asym(spec) -> dict:
    return {
        SpinalLevel.L3_L4: (_ramp(0.10, 0.16), _ramp(0.05, 0.02), _ramp(-0.08, 0.0))
    }


def _sym(spec) -> dict:
    zero = np.zeros(_TIMES.size)
    return {SpinalLevel.L3_L4: (_ramp(0.12, 0.18), zero, zero)}


def _solve(one_level, angles, exponent, **opts) -> SpineStaticOptimizationResult:
    spec, muscled, _bare, level_map, tmp = one_level
    efforts = _efforts(muscled, spec, angles, tmp, level_map)
    options = SpineSOOptions(
        exponent=exponent, events=_EVENTS, level_map=level_map, **opts
    )
    return run_spine_static_optimization(muscled, efforts, "fixture", options=options)


# ---------------------------------------------------------- native: accessor


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_generalized_forces_exclude_muscles(one_level) -> None:
    spec, muscled, bare, level_map, tmp = one_level
    angles = _asym(spec)
    ref = _efforts(bare, spec, angles, tmp, level_map)
    excl = _efforts(muscled, spec, angles, tmp, level_map)
    incl = _efforts(muscled, spec, angles, tmp, level_map, exclude_muscles=False)
    assert excl.level_coordinates == tuple(fixture_level_names(spec)[0].coordinates)
    assert excl.level_tau().shape == (_TIMES.size, 3)
    np.testing.assert_allclose(excl.level_tau(), ref.level_tau(), atol=1e-9)
    assert np.max(np.abs(incl.level_tau() - ref.level_tau())) > 1e-3
    assert "excluded" in excl.provenance["muscles"]
    assert not excl.tau.flags.writeable


# --------------------------------------------------------- native: the basis


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_basis_force_is_affine_in_activation(one_level) -> None:
    spec, muscled, _bare, level_map, tmp = one_level
    efforts = _efforts(muscled, spec, _asym(spec), tmp, level_map)
    basis = SpineMuscleBasis(muscled, "fixture", level_map=level_map)
    assert basis.muscle_names == fixture_muscle_names()
    q, qd = efforts.q[3], efforts.qd[3]
    active, passive, moment = basis.evaluate(efforts.model_coordinates, q, qd)
    assert moment.shape == (8, 3)
    assert np.all(active > 0.0)
    for a in (0.2, 0.37, 0.9):
        native = basis.tendon_force(efforts.model_coordinates, q, qd, a)
        np.testing.assert_allclose(native, passive + a * active, rtol=1e-9, atol=1e-9)


# -------------------------------------------------------- native: acceptance


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("exponent", [2, 3])
def test_reachable_id_moments_met_with_zero_reserves(one_level, exponent) -> None:
    result = _solve(one_level, _asym(one_level[0]), exponent)
    assert result.frame_labels == ("address", "top", "impact")
    assert result.activation.shape == (3, 8)
    assert result.cost_exponent == exponent
    assert np.all(result.success)
    np.testing.assert_allclose(result.muscle_moment_nm, result.required_nm, atol=1e-6)
    np.testing.assert_allclose(result.reserve_nm, 0.0, atol=1e-6)
    assert not np.any(result.flagged)
    assert np.all(result.activation >= 0.0) and np.all(result.activation <= 1.0)
    assert np.max(np.abs(result.required_nm)) > 1.0  # non-trivial demand
    for key in REQUIRED_PROVENANCE_KEYS:
        assert result.provenance[key]
    assert result.provenance["cost_exponent"] == str(exponent)
    assert not result.force_n.flags.writeable


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("exponent", [2, 3])
def test_symmetric_pose_gives_bilateral_activations(one_level, exponent) -> None:
    result = _solve(one_level, _sym(one_level[0]), exponent)
    names = result.muscle_names
    left = [i for i, n in enumerate(names) if n.endswith("_l")]
    right = [names.index(names[i][:-2] + "_r") for i in left]
    np.testing.assert_allclose(result.required_nm[:, 1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        result.activation[:, left], result.activation[:, right], atol=1e-6
    )
    assert np.max(result.activation) > 1e-3


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_p3_spreads_load_across_more_muscles_than_p2(one_level) -> None:
    two = _solve(one_level, _asym(one_level[0]), 2)
    three = _solve(one_level, _asym(one_level[0]), 3)
    for k in range(two.time_s.size):
        assert participation_ratio(three.activation[k]) > participation_ratio(
            two.activation[k]
        )
    np.testing.assert_allclose(
        three.participation, [participation_ratio(a) for a in three.activation]
    )


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_five_level_reserves_carry_the_remainder_and_are_flagged(
    tmp_path_factory,
) -> None:
    require_opensim()
    spec = FixtureSpec()
    tmp = tmp_path_factory.mktemp("so_five")
    path = _retune(write_fixture_osim(spec, tmp / "five.osim", muscles=True))
    angles = {
        names.level: (
            _ramp(0.05 + 0.02 * k, 0.1),
            _ramp(0.03, -0.01 * k),
            _ramp(-0.02 * k, 0.04),
        )
        for k, names in enumerate(fixture_level_names(spec))
    }
    efforts = _efforts(path, spec, angles, tmp, None)
    reserves = ReserveSpec(limit_nm=1.0, max_share=0.05)
    options = SpineSOOptions(
        exponent=3, events=_EVENTS, stride=4, max_frames=5, reserves=reserves
    )
    result = run_spine_static_optimization(path, efforts, "fixture", options=options)
    assert result.time_s.size <= 5
    assert result.reserve_nm.shape == (result.time_s.size, 15)
    np.testing.assert_allclose(
        result.muscle_moment_nm + result.reserve_nm, result.required_nm, atol=1e-9
    )
    assert np.all(result.flagged)
    assert all(any("share" in r for r in rs) for rs in result.flag_reasons)
    assert np.all(result.reserve_share > 0.05)
    basis = SpineMuscleBasis(path, "fixture")
    for k, frame in enumerate(result.frame_index):
        active, _p, moment = basis.evaluate(
            efforts.model_coordinates, efforts.q[frame], efforts.qd[frame]
        )
        slope = (moment * active[:, None]) @ result.reserve_nm[k]  # limits 1 N*m
        for a, g in zip(result.activation[k], slope, strict=True):
            if a < 1.0 - 1e-7:
                assert g <= 1e-5 * np.max(np.abs(slope))
            if a > 1e-7:
                assert g >= -1e-5 * np.max(np.abs(slope))


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_runtime_contract_refuses_too_many_frames(one_level) -> None:
    with pytest.raises(ValueError, match="max_frames"):
        _solve(one_level, _asym(one_level[0]), 2, stride=1, max_frames=5)
