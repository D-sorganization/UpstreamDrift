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
import xml.etree.ElementTree as ET

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


def _retag_thelen(src: Path, dst: Path) -> Path:
    tree = ET.parse(src)
    for node in tree.iter("Millard2012EquilibriumMuscle"):
        node.tag = "Thelen2003Muscle"
        ET.SubElement(node, "FmaxTendonStrain").text = "0.04"
        ET.SubElement(node, "KshapeActive").text = "0.5"
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
    b = SpineMuscleBasis(thelen, "fixture", level_map=level_map)
    assert a.muscle_names == b.muscle_names
    names = a.model.getCoordinateSet()
    q = np.zeros(names.getSize())
    qd = np.zeros(names.getSize())
    all_names = tuple(str(names.get(i).getName()) for i in range(names.getSize()))
    for x, y in zip(
        a.evaluate(all_names, q, qd), b.evaluate(all_names, q, qd), strict=True
    ):
        np.testing.assert_allclose(x, y, atol=1e-9)
