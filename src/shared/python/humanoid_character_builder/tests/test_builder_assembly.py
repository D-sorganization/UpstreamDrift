# ruff: noqa: E501
"""Value-asserting builder-assembly and URDF/segment-output tests.

Covers issue #3186: thin coverage of builder-assembly and segment/URDF output
paths. These tests perform a full character build from parameters and assert
the assembled structure (segment set, mass conservation, inertia), then parse
the emitted URDF and assert the link/joint/segment entities for a fixed input.
A preset-driven build and a preset file round-trip exercise the loader path.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from humanoid_character_builder.interfaces.api import (
    CharacterBuildResult,
    ExportOptions,
    quick_build,
)
from humanoid_character_builder.presets.loader import (
    list_available_presets,
    load_body_preset,
    load_preset_from_file,
    save_preset_to_file,
)

from humanoid_character_builder import (
    BodyParameters,
    CharacterBuilder,
)

# Fixed input for the golden assembly assertions.
_FIXED_HEIGHT_M = 1.80
_FIXED_MASS_KG = 80.0


@pytest.fixture
def builder() -> CharacterBuilder:
    return CharacterBuilder()


@pytest.fixture
def fixed_params() -> BodyParameters:
    return BodyParameters(height_m=_FIXED_HEIGHT_M, mass_kg=_FIXED_MASS_KG)


@pytest.fixture
def built(
    builder: CharacterBuilder, fixed_params: BodyParameters
) -> CharacterBuildResult:
    """A full character build (no mesh files) from the fixed parameters."""
    return builder.build(fixed_params, generate_meshes=False)


# ---------------------------------------------------------------------------
# Builder assembly — assert the assembled structure as a whole.
# ---------------------------------------------------------------------------


class TestBuilderAssembly:
    def test_build_succeeds_and_populates_segments(
        self, built: CharacterBuildResult
    ) -> None:
        assert built.success is True
        assert built.urdf_xml is not None
        # Full skeletal segment set is assembled.
        assert len(built.segments) == len(CharacterBuilder.list_segments())
        assert len(built.segments) > 15

    def test_assembled_segments_match_canonical_names(
        self, built: CharacterBuildResult
    ) -> None:
        assembled = set(built.get_all_segments())
        canonical = set(CharacterBuilder.list_segments())
        assert assembled == canonical
        # Known segments are present in the assembly.
        assert "pelvis" in assembled or "torso" in assembled

    def test_mass_is_conserved_across_assembly(
        self, built: CharacterBuildResult
    ) -> None:
        """Sum of segment masses equals the requested body mass."""
        assert built.get_total_mass() == pytest.approx(_FIXED_MASS_KG, rel=1e-3)

    def test_each_segment_has_positive_mass_and_inertia(
        self, built: CharacterBuildResult
    ) -> None:
        for name, seg in built.segments.items():
            assert seg.mass_kg > 0.0, f"{name} has non-positive mass"
            assert seg.segment_name == name
            # Diagonal inertia terms are physically positive.
            inertia = seg.inertia
            assert inertia.ixx > 0.0
            assert inertia.iyy > 0.0
            assert inertia.izz > 0.0

    def test_get_segment_lookup(self, built: CharacterBuildResult) -> None:
        any_name = built.get_all_segments()[0]
        seg = built.get_segment(any_name)
        assert seg is not None
        assert seg.segment_name == any_name
        assert built.get_segment("not_a_segment") is None

    def test_to_dict_reports_assembly_summary(
        self, built: CharacterBuildResult
    ) -> None:
        summary = built.to_dict()
        assert summary["success"] is True
        assert summary["segment_count"] == len(built.segments)
        assert summary["total_mass"] == pytest.approx(_FIXED_MASS_KG, rel=1e-3)
        assert summary["error_message"] is None

    def test_taller_character_increases_segment_lengths(
        self, builder: CharacterBuilder
    ) -> None:
        """Assembly responds to parameters: taller -> longer thigh segment."""
        short = builder.build(
            BodyParameters(height_m=1.50, mass_kg=_FIXED_MASS_KG),
            generate_meshes=False,
        )
        tall = builder.build(
            BodyParameters(height_m=2.00, mass_kg=_FIXED_MASS_KG),
            generate_meshes=False,
        )
        # Pick a common segment present in both assemblies.
        common = set(short.get_all_segments()) & set(tall.get_all_segments())
        sample = next(iter(common))
        short_len = short.segments[sample].dimensions.get("length", 0.0)
        tall_len = tall.segments[sample].dimensions.get("length", 0.0)
        assert tall_len > short_len


# ---------------------------------------------------------------------------
# URDF / segment output — assert emitted entities for a fixed input.
# ---------------------------------------------------------------------------


class TestURDFOutputStructure:
    @pytest.fixture
    def urdf_root(self, built: CharacterBuildResult) -> ET.Element:
        assert built.urdf_xml is not None
        # nosec B314 - parsing URDF XML generated in-process by the builder under
        # test (trusted, not external/untrusted input).
        return ET.fromstring(built.urdf_xml)  # nosec B314

    def test_root_is_named_robot(self, urdf_root: ET.Element) -> None:
        assert urdf_root.tag == "robot"
        assert urdf_root.get("name")

    def test_link_and_joint_topology(self, urdf_root: ET.Element) -> None:
        """A tree of N links is connected by N-1 joints."""
        links = urdf_root.findall("link")
        joints = urdf_root.findall("joint")
        assert len(links) > 10
        # Kinematic tree: joints connect links into a single tree.
        assert len(joints) == len(links) - 1

    def test_every_link_has_inertial_block(self, urdf_root: ET.Element) -> None:
        for link in urdf_root.findall("link"):
            inertial = link.find("inertial")
            assert inertial is not None, f"link {link.get('name')} missing inertial"
            mass_el = inertial.find("mass")
            assert mass_el is not None
            assert float(mass_el.get("value", "0")) > 0.0

    def test_joints_reference_existing_links(self, urdf_root: ET.Element) -> None:
        link_names = {link.get("name") for link in urdf_root.findall("link")}
        for joint in urdf_root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            assert parent is not None and child is not None
            assert parent.get("link") in link_names
            assert child.get("link") in link_names

    def test_links_carry_visual_geometry(self, urdf_root: ET.Element) -> None:
        geom_links = [
            link
            for link in urdf_root.findall("link")
            if link.find("visual/geometry") is not None
        ]
        assert geom_links, "no link emitted visual geometry"

    def test_export_urdf_writes_package(
        self, built: CharacterBuildResult, tmp_path: Path
    ) -> None:
        out = tmp_path / "char_pkg"
        options = ExportOptions(generate_meshes=False, save_config=True)
        urdf_path = built.export_urdf(out, options=options)
        assert urdf_path.exists()
        assert urdf_path.suffix == ".urdf"
        # Config written and parseable as the same URDF robot.
        config_dir = out / options.config_subdirectory
        assert config_dir.exists()
        # nosec B314 - parsing the URDF file just written by export_urdf in this
        # test (trusted, not external/untrusted input).
        root = ET.fromstring(urdf_path.read_text())  # nosec B314
        assert root.tag == "robot"


# ---------------------------------------------------------------------------
# Preset-driven assembly + loader file round-trip.
# ---------------------------------------------------------------------------


class TestPresetDrivenAssembly:
    def test_quick_build_from_preset(self) -> None:
        result = quick_build(preset="athletic")
        assert result.success is True
        assert result.urdf_xml is not None
        assert result.get_total_mass() > 0.0

    def test_all_listed_presets_load(self) -> None:
        presets = list_available_presets()
        assert presets
        for name in presets:
            params = load_body_preset(name)
            assert isinstance(params, BodyParameters)
            assert params.height_m > 0.0
            assert params.mass_kg > 0.0

    def test_preset_overrides_applied(self) -> None:
        params = load_body_preset("athletic", height_m=1.95, mass_kg=95.0)
        assert params.height_m == pytest.approx(1.95)
        assert params.mass_kg == pytest.approx(95.0)

    def test_preset_file_round_trip(self, tmp_path: Path) -> None:
        """Saving then loading a preset reproduces the core parameters."""
        original = load_body_preset("average")
        preset_file = tmp_path / "custom.yaml"
        save_preset_to_file(original, preset_file)
        assert preset_file.exists()
        restored = load_preset_from_file(preset_file)
        assert restored.height_m == pytest.approx(original.height_m)
        assert restored.mass_kg == pytest.approx(original.mass_kg)

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            load_body_preset("definitely_not_a_preset")
