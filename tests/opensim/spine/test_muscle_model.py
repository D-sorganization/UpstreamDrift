"""Thelen to Millard muscle swap for the rigid-tendon spine solves (#12761).

The licensed Bruno 2015 model carries only ``Thelen2003Muscle`` actuators and
OpenSim 4.6 does not implement ``ignore_tendon_compliance`` for them, so the
rigid-tendon static optimisation and joint-reaction solves failed in
``initSystem``.  Every number here is INVENTED (the five-level fixture with
its lumped muscles, retagged as Thelen in a test-local copy).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import defusedxml.ElementTree as ET

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    load_level_maps,
)
from src.engines.physics_engines.opensim.python.spine.muscle_model import (
    millard_equivalent_model,
)
from src.engines.physics_engines.opensim.python.spine.static_optimization import (
    SpineMuscleBasis,
)
from src.shared.python.spine.levels import SpinalLevel
from tests.helpers.opensim_support import require_opensim


def _one_level_map():
    full = load_level_maps()["fixture"]
    return replace(full, levels={SpinalLevel.L3_L4: full.levels[SpinalLevel.L3_L4]})


def _retag_thelen(src: Path, dst: Path) -> Path:
    tree = ET.parse(src)
    for node in tree.iter("Millard2012EquilibriumMuscle"):
        node.tag = "Thelen2003Muscle"
        for tag, text in (("FmaxTendonStrain", "0.04"), ("KshapeActive", "0.5")):
            child = node.makeelement(tag, {})
            child.text = text
            node.append(child)
    tree.write(dst)
    return dst


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    require_opensim()
    tmp = tmp_path_factory.mktemp("thelen")
    spec = FixtureSpec(levels=(SpinalLevel.L3_L4,))
    millard = write_fixture_osim(spec, tmp / "millard.osim", muscles=True)
    return millard, _retag_thelen(millard, tmp / "thelen.osim"), tmp


@pytest.mark.unit
def test_swap_keeps_core_parameters_and_drops_thelen_only(models) -> None:
    millard, thelen, tmp = models
    out = millard_equivalent_model(thelen, tmp / "swapped.osim")
    assert out.is_file()
    tags = {n.tag for n in ET.parse(out).iter() if n.tag.endswith("Muscle")}
    assert tags == {"Millard2012EquilibriumMuscle"}
    swapped = ET.parse(out).getroot()
    assert swapped.find(".//FmaxTendonStrain") is None
    assert swapped.find(".//KshapeActive") is None
    for a, b in zip(
        ET.parse(millard).iter("Millard2012EquilibriumMuscle"),
        swapped.iter("Millard2012EquilibriumMuscle"),
        strict=True,
    ):
        for key in ("max_isometric_force", "optimal_fiber_length"):
            assert a.findtext(key) == b.findtext(key)
        assert a.get("name") == b.get("name")


@pytest.mark.unit
def test_swap_is_a_noop_without_thelen(models) -> None:
    millard, _thelen, tmp = models
    assert millard_equivalent_model(millard, tmp / "same.osim") is None


@pytest.mark.unit
def test_swap_rejects_bad_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="not found"):
        millard_equivalent_model(tmp_path / "missing.osim", tmp_path / "o.osim")


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_basis_builds_on_thelen_model_and_matches_millard(models) -> None:
    millard, thelen, _tmp = models
    full = load_level_maps()["fixture"]
    level_map = replace(
        full, levels={SpinalLevel.L3_L4: full.levels[SpinalLevel.L3_L4]}
    )
    a = SpineMuscleBasis(millard, "fixture", level_map=level_map)
    b = SpineMuscleBasis(
        thelen, "fixture", level_map=level_map, muscle_model_policy="millard_equivalent"
    )
    assert a.muscle_names == b.muscle_names
    names = a.model.getCoordinateSet()
    q = np.zeros(names.getSize())
    qd = np.zeros(names.getSize())
    all_names = tuple(str(names.get(i).getName()) for i in range(names.getSize()))
    for x, y in zip(
        a.evaluate(all_names, q, qd), b.evaluate(all_names, q, qd), strict=True
    ):
        np.testing.assert_allclose(x, y, atol=1e-9)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_moment_arm_screen_never_drops_a_nonzero_native_arm() -> None:
    """The path-length screen must keep every pair with a native moment arm.

    Native ``computeMomentArm`` costs tens of milliseconds per pair on a
    full-body model (446 muscles x 51 coordinates did not finish one frame in
    45 minutes), so the basis evaluates only the pairs a path-length screen
    cannot rule out.  The lumped fixture muscles span every level, so here the
    check is that no nonzero native arm is dropped; the saving is measured on the
    full-body model.
    """
    from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
        muscle_moment_arm,
    )

    require_opensim()
    with_tmp = pytest.importorskip("tempfile").TemporaryDirectory()
    with with_tmp as tmp:
        path = write_fixture_osim(FixtureSpec(), Path(tmp) / "m.osim", muscles=True)
        basis = SpineMuscleBasis(path, "fixture")
        coords = basis.model.getCoordinateSet()
        names = tuple(str(coords.get(i).getName()) for i in range(coords.getSize()))
        q = np.linspace(0.02, 0.09, len(names))
        qd = np.zeros(len(names))
        mask = basis.active_pair_mask(names, q, qd)
        assert mask.shape == (len(basis.muscle_names), len(basis.coordinate_names))
        assert mask.dtype == bool
        basis._pose(names, q, qd)
        for i, muscle in enumerate(basis.muscle_names):
            for j, coord in enumerate(basis.coordinate_names):
                native = muscle_moment_arm(basis.model, basis.state, muscle, coord)
                if abs(native) > 1e-9:
                    assert mask[i, j], (muscle, coord, native)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_native_policy_refuses_thelen_naming_the_option(models) -> None:
    _millard, thelen, _tmp = models
    with pytest.raises(ValueError, match="muscle_model_policy"):
        SpineMuscleBasis(thelen, "fixture", level_map=_one_level_map())


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_native_policy_accepts_millard_and_reports_no_swap(models) -> None:
    millard, _thelen, _tmp = models
    basis = SpineMuscleBasis(millard, "fixture", level_map=_one_level_map())
    assert basis.swapped_muscles == 0
    swapped = SpineMuscleBasis(
        _thelen,
        "fixture",
        level_map=_one_level_map(),
        muscle_model_policy="millard_equivalent",
    )
    assert swapped.swapped_muscles == len(swapped.muscle_names) > 0


@pytest.mark.unit
def test_policy_validation() -> None:
    from src.engines.physics_engines.opensim.python.spine.muscle_model import (
        check_muscle_model_policy,
    )

    assert check_muscle_model_policy("native") == "native"
    with pytest.raises(ValueError, match="muscle_model_policy"):
        check_muscle_model_policy("thelen")
    with pytest.raises(TypeError):
        check_muscle_model_policy(1)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_screened_basis_equals_unscreened_basis(tmp_path) -> None:
    """The screened moment-arm matrix equals the full native matrix."""
    from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
        muscle_moment_arm,
    )

    require_opensim()
    path = write_fixture_osim(FixtureSpec(), tmp_path / "m.osim", muscles=True)
    basis = SpineMuscleBasis(path, "fixture")
    coords = basis.model.getCoordinateSet()
    names = tuple(str(coords.get(i).getName()) for i in range(coords.getSize()))
    q = np.linspace(0.02, 0.09, len(names))
    qd = np.zeros(len(names))
    _active, _passive, screened = basis.evaluate(names, q, qd)
    basis._pose(names, q, qd)
    full = np.array(
        [
            [
                muscle_moment_arm(basis.model, basis.state, m, c)
                for c in basis.coordinate_names
            ]
            for m in basis.muscle_names
        ]
    )
    np.testing.assert_allclose(screened, full, atol=1e-12, rtol=0.0)


def _locked_fixture(tmp_path: Path) -> tuple[Path, str]:
    osim = require_opensim()
    spec = FixtureSpec(levels=(SpinalLevel.L3_L4,))
    path = write_fixture_osim(spec, tmp_path / "locked.osim", muscles=True)
    model = osim.Model(str(path))
    coords = model.getCoordinateSet()
    name = str(coords.get(coords.getSize() - 1).getName())
    coords.get(name).set_locked(True)
    other = str(coords.get(0).getName())
    coords.get(other).set_locked(True)
    model.printToXML(str(path))
    return path, name


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_prepare_unlocks_only_the_named_coordinates(tmp_path) -> None:
    from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
        prepare_model_for_id,
    )

    path, name = _locked_fixture(tmp_path)

    def locked(prepared) -> set[str]:
        coords = prepared.model.getCoordinateSet()
        return {
            str(coords.get(i).getName())
            for i in range(coords.getSize())
            if coords.get(i).get_locked()
        }

    default = prepare_model_for_id(path)
    assert name in locked(default) and default.unlocked_coordinates == ()
    chosen = prepare_model_for_id(path, unlock_coordinates=(name,))
    assert name not in locked(chosen)
    assert len(locked(chosen)) == len(locked(default)) - 1
    assert chosen.unlocked_coordinates == (name,)
    with pytest.raises(ValueError, match="unknown coordinates"):
        prepare_model_for_id(path, unlock_coordinates=("nope",))
