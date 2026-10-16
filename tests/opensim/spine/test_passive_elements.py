"""Passive spine elements (SPN-8c, #12755): translations and bushing extraction.

Every stiffness, damping and expression below is INVENTED for the synthetic
fixture; none is transcribed from a licensed model.  The licensed Senteler
30-DOF bushing matrices are read at runtime on the owner workstation only.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep  # test-only parse of generated XML
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    build_fixture_osim,
    fixture_level_names,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.fixture_passive import (
    FixtureBushing,
    default_fixture_bushings,
    fixture_level_map,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    LevelMapEntry,
    SpineInventory,
    inventory_spine,
)
from src.engines.physics_engines.opensim.python.spine.passive_elements import (
    PassiveElementSpec,
    extract_passive_elements,
)
from src.shared.python.spine.levels import SpinalLevel
from tests.helpers.opensim_support import require_opensim

LEVELS = FixtureSpec().levels
TRANS_SPEC = FixtureSpec(translational_dofs=True)

# --------------------------------------------------------------------- pure --


@pytest.mark.unit
class TestFixtureBushingPure:
    def test_defaults_cover_every_level_with_invented_distinct_values(self) -> None:
        table = default_fixture_bushings(TRANS_SPEC)
        assert tuple(table) == LEVELS
        stiff = {b.translational_stiffness for b in table.values()}
        assert len(stiff) == len(LEVELS)

    def test_negative_stiffness_rejected(self) -> None:
        with pytest.raises(ValueError, match="stiffness"):
            FixtureBushing(translational_stiffness=(-1.0, 1.0, 1.0))

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            FixtureBushing(kind="Spring")

    def test_expression_kind_needs_all_six_expressions(self) -> None:
        with pytest.raises(ValueError, match="expressions"):
            FixtureBushing(kind="ExpressionBasedBushingForce", expressions={"Fx": "1"})

    def test_wrong_vector_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="3"):
            FixtureBushing(rotational_damping=(1.0, 2.0))

    def test_unknown_level_rejected_in_build(self) -> None:
        with pytest.raises(ValueError, match="level"):
            build_fixture_osim(
                FixtureSpec(), bushings={SpinalLevel.T12_L1: FixtureBushing()}
            )

    def test_build_without_new_options_has_no_bushings_or_translations(self) -> None:
        text = build_fixture_osim(FixtureSpec())
        assert "BushingForce" not in text
        assert "ant_post" not in text

    def test_translational_dofs_names(self) -> None:
        names = fixture_level_names(TRANS_SPEC)[0]
        assert names.translations == (
            "l5_s1_ant_post",
            "l5_s1_cran_caud",
            "l5_s1_med_lat",
        )
        assert fixture_level_names(FixtureSpec())[0].translations is None

    def test_level_map_carries_translations(self) -> None:
        lmap = fixture_level_map(TRANS_SPEC)
        assert lmap.verified
        assert lmap.levels[SpinalLevel.L5_S1].translations is not None
        assert fixture_level_map(FixtureSpec()).levels[
            SpinalLevel.L5_S1
        ].translations is None


@pytest.mark.unit
class TestPassiveElementSpecPure:
    def _kw(self, **over):
        base = {
            "level": SpinalLevel.L5_S1,
            "name": "b",
            "bushing_type": "BushingForce",
            "frame1_body": "sacrum",
            "frame2_body": "L5",
            "frame1_on_inferior": True,
            "translational_stiffness": np.array([1.0, 2.0, 3.0]),
            "rotational_stiffness": np.array([1.0, 2.0, 3.0]),
            "translational_damping": np.zeros(3),
            "rotational_damping": np.zeros(3),
        }
        base.update(over)
        return base

    def test_arrays_made_read_only_copies(self) -> None:
        src = np.array([1.0, 2.0, 3.0])
        spec = PassiveElementSpec(**self._kw(translational_stiffness=src))
        assert not spec.translational_stiffness.flags.writeable
        src[0] = 99.0
        assert spec.translational_stiffness[0] == 1.0

    def test_bad_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            PassiveElementSpec(**self._kw(rotational_stiffness=np.zeros(4)))

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            PassiveElementSpec(**self._kw(translational_damping=-np.ones(3)))

    def test_bad_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="bushing_type"):
            PassiveElementSpec(**self._kw(bushing_type="Weld"))

    def test_expression_based_has_no_stiffness_vectors(self) -> None:
        spec = PassiveElementSpec(
            **self._kw(
                bushing_type="ExpressionBasedBushingForce",
                translational_stiffness=None,
                rotational_stiffness=None,
                expressions={"Fx": "1*dx"},
            )
        )
        assert spec.translational_stiffness is None
        with pytest.raises(ValueError, match="expressions"):
            PassiveElementSpec(
                **self._kw(
                    translational_stiffness=None, rotational_stiffness=None
                )
            )


# ------------------------------------------------------------------- native --


def _paths(tmp_path_factory, bushings):
    return write_fixture_osim(
        TRANS_SPEC,
        tmp_path_factory.mktemp("passive") / "fixture.osim",
        bushings=bushings,
    )


@pytest.fixture(scope="module")
def bushed_path(tmp_path_factory) -> Path:
    return _paths(tmp_path_factory, default_fixture_bushings(TRANS_SPEC))


@pytest.fixture(scope="module")
def inventory(bushed_path: Path) -> SpineInventory:
    require_opensim()
    return inventory_spine(
        bushed_path, "fixture", level_map=fixture_level_map(TRANS_SPEC), strict=True
    )


@pytest.mark.requires_opensim
@pytest.mark.integration
class TestTranslationInventory:
    def test_translation_names_and_alignment(self, inventory: SpineInventory) -> None:
        for names in fixture_level_names(TRANS_SPEC):
            assert inventory.translation_names[names.level] == names.translations
            deg = inventory.translation_alignment_deg[names.level]
            assert deg.shape == (3,)
            assert np.all(deg <= 1.0)
            assert not deg.flags.writeable
        assert inventory.axis_ok

    def test_no_translations_for_rotation_only_map(self, tmp_path: Path) -> None:
        require_opensim()
        path = write_fixture_osim(FixtureSpec(), tmp_path / "f.osim")
        inv = inventory_spine(path, "fixture")
        assert dict(inv.translation_names) == {}
        assert dict(inv.translation_alignment_deg) == {}

    def test_missing_translation_coordinate_raises(self, tmp_path: Path) -> None:
        require_opensim()
        path = write_fixture_osim(FixtureSpec(), tmp_path / "f.osim")
        lmap = fixture_level_map(TRANS_SPEC)  # names a translation the model lacks
        with pytest.raises(ValueError, match="l5_s1_ant_post"):
            inventory_spine(path, "fixture", level_map=lmap)

    def test_misaligned_translation_axis_listed(self, tmp_path: Path) -> None:
        require_opensim()
        path = write_fixture_osim(TRANS_SPEC, tmp_path / "f.osim")
        tree = ET.parse(path)  # noqa: S314  # nosemgrep  # generated, trusted
        for joint in tree.getroot().iter("CustomJoint"):
            if joint.get("name") != "L4_L5_jnt":
                continue
            for axis in joint.iter("TransformAxis"):
                if axis.get("name") == "translation1":
                    axis.find("axis").text = "1 1 0"
        bad = tmp_path / "bad.osim"
        tree.write(bad)
        inv = inventory_spine(bad, "fixture", level_map=fixture_level_map(TRANS_SPEC))
        assert not inv.axis_ok
        assert any("l4_l5_ant_post" in f for f in inv.axis_failures)

    def test_partial_translations_rejected(self) -> None:
        with pytest.raises(ValueError, match="translations"):
            LevelMapEntry("j", ("a", "b", "c"), translations=("x", "y"))  # type: ignore[arg-type]


@pytest.mark.requires_opensim
@pytest.mark.integration
class TestExtractPassiveElements:
    def test_reads_every_level_exactly(
        self, bushed_path: Path, inventory: SpineInventory
    ) -> None:
        expected = default_fixture_bushings(TRANS_SPEC)
        found = extract_passive_elements(bushed_path, inventory)
        assert tuple(found) == LEVELS
        for names in fixture_level_names(TRANS_SPEC):
            spec = found[names.level]
            ref = expected[names.level]
            assert spec.level is names.level
            assert spec.bushing_type == "BushingForce"
            assert spec.frame1_body == names.inferior_body
            assert spec.frame2_body == names.superior_body
            assert spec.frame1_on_inferior is True
            np.testing.assert_allclose(
                spec.translational_stiffness, ref.translational_stiffness, rtol=1e-14
            )
            np.testing.assert_allclose(
                spec.rotational_stiffness, ref.rotational_stiffness, rtol=1e-14
            )
            np.testing.assert_allclose(
                spec.translational_damping, ref.translational_damping, rtol=1e-14
            )
            np.testing.assert_allclose(
                spec.rotational_damping, ref.rotational_damping, rtol=1e-14
            )
            assert not spec.translational_stiffness.flags.writeable

    def test_reversed_frame_order_is_flagged_not_swapped(
        self, tmp_path: Path
    ) -> None:
        require_opensim()
        path = write_fixture_osim(
            TRANS_SPEC,
            tmp_path / "f.osim",
            bushings={SpinalLevel.L4_L5: FixtureBushing(frames_reversed=True)},
        )
        inv = inventory_spine(path, "fixture", level_map=fixture_level_map(TRANS_SPEC))
        spec = extract_passive_elements(path, inv)[SpinalLevel.L4_L5]
        assert spec.frame1_on_inferior is False
        assert (spec.frame1_body, spec.frame2_body) == ("L4", "L5")

    def test_accepts_loaded_model(self, bushed_path: Path, inventory) -> None:
        osim = require_opensim()
        found = extract_passive_elements(osim.Model(str(bushed_path)), inventory)
        assert len(found) == 5

    def test_expression_based_bushing(self, tmp_path: Path) -> None:
        require_opensim()
        expr = {k: f"{i + 1}*{k}" for i, k in enumerate(("Fx", "Fy", "Fz"))}
        expr.update({k: "0.5*theta" for k in ("Mx", "My", "Mz")})
        path = write_fixture_osim(
            TRANS_SPEC,
            tmp_path / "f.osim",
            bushings={
                SpinalLevel.L3_L4: FixtureBushing(
                    kind="ExpressionBasedBushingForce", expressions=expr
                )
            },
        )
        inv = inventory_spine(path, "fixture", level_map=fixture_level_map(TRANS_SPEC))
        found = extract_passive_elements(path, inv)
        assert tuple(found) == (SpinalLevel.L3_L4,)
        spec = found[SpinalLevel.L3_L4]
        assert spec.bushing_type == "ExpressionBasedBushingForce"
        assert spec.translational_stiffness is None
        assert spec.expressions == expr
        assert spec.translational_damping.shape == (3,)

    def test_level_without_bushing_omitted_unless_required(
        self, tmp_path: Path, inventory
    ) -> None:
        require_opensim()
        path = write_fixture_osim(
            TRANS_SPEC,
            tmp_path / "f.osim",
            bushings={SpinalLevel.L5_S1: FixtureBushing()},
        )
        assert tuple(extract_passive_elements(path, inventory)) == (
            SpinalLevel.L5_S1,
        )
        with pytest.raises(ValueError, match="L4_L5"):
            extract_passive_elements(path, inventory, require_all=True)

    def test_model_without_bushings_is_empty(self, tmp_path: Path, inventory) -> None:
        require_opensim()
        path = write_fixture_osim(TRANS_SPEC, tmp_path / "f.osim")
        assert dict(extract_passive_elements(path, inventory)) == {}

    def test_two_bushings_on_one_level_is_ambiguous(
        self, tmp_path: Path, inventory
    ) -> None:
        osim = require_opensim()
        path = write_fixture_osim(
            TRANS_SPEC,
            tmp_path / "f.osim",
            bushings={SpinalLevel.L5_S1: FixtureBushing()},
        )
        model = osim.Model(str(path))
        extra = osim.BushingForce(
            "dup_l5s1",
            "/bodyset/sacrum",
            "/bodyset/L5",
            osim.Vec3(1),
            osim.Vec3(1),
            osim.Vec3(0),
            osim.Vec3(0),
        )
        model.addForce(extra)
        model.finalizeConnections()
        with pytest.raises(ValueError, match="ambiguous.*L5_S1"):
            extract_passive_elements(model, inventory)

    def test_bushing_between_non_adjacent_spine_bodies_raises(
        self, tmp_path: Path, inventory
    ) -> None:
        osim = require_opensim()
        path = write_fixture_osim(TRANS_SPEC, tmp_path / "f.osim")
        model = osim.Model(str(path))
        model.addForce(
            osim.BushingForce(
                "skip",
                "/bodyset/sacrum",
                "/bodyset/L4",
                osim.Vec3(1),
                osim.Vec3(1),
                osim.Vec3(0),
                osim.Vec3(0),
            )
        )
        model.finalizeConnections()
        with pytest.raises(ValueError, match="skip"):
            extract_passive_elements(model, inventory)

    def test_bushing_outside_spine_is_ignored(self, tmp_path: Path, inventory) -> None:
        osim = require_opensim()
        path = write_fixture_osim(TRANS_SPEC, tmp_path / "f.osim")
        model = osim.Model(str(path))
        model.addForce(
            osim.BushingForce(
                "pelvis_ground",
                "/ground",
                "/bodyset/pelvis",
                osim.Vec3(1),
                osim.Vec3(1),
                osim.Vec3(0),
                osim.Vec3(0),
            )
        )
        model.finalizeConnections()
        assert dict(extract_passive_elements(model, inventory)) == {}

    def test_function_based_bushing_between_levels_is_unsupported(
        self, tmp_path: Path, inventory
    ) -> None:
        osim = require_opensim()
        path = write_fixture_osim(TRANS_SPEC, tmp_path / "f.osim")
        model = osim.Model(str(path))
        model.addForce(
            osim.FunctionBasedBushingForce("fb", "/bodyset/L5", "/bodyset/L4")
        )
        model.finalizeConnections()
        with pytest.raises(ValueError, match="FunctionBasedBushingForce"):
            extract_passive_elements(model, inventory)

    def test_rejects_wrong_types_and_missing_file(self, tmp_path: Path, inventory) -> None:
        require_opensim()
        with pytest.raises(TypeError, match="SpineInventory"):
            extract_passive_elements(tmp_path / "x.osim", object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="not found"):
            extract_passive_elements(tmp_path / "absent.osim", inventory)

    def test_levels_missing_from_model_bodies_raise(self, tmp_path: Path, inventory) -> None:
        require_opensim()
        small = FixtureSpec(levels=(SpinalLevel.L5_S1, SpinalLevel.L4_L5))
        path = write_fixture_osim(small, tmp_path / "s.osim")
        with pytest.raises(ValueError, match="joint"):
            extract_passive_elements(path, inventory)
