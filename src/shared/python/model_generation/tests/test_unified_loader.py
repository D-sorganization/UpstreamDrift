# ruff: noqa: E501
"""
Tests for the unified model loader, bundled library, and model explorer.

Covers:
- Format detection (URDF vs MJCF)
- Loading bundled models (all 4 bundled models)
- MJCF parsing into ParsedModel
- User preferences (save, load, default model)
- Display checkbox defaults
- Library manifest integrity
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest


class TestFormatDetection:
    """Tests for automatic format detection."""

    def test_urdf_extension_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "robot.urdf"
        f.write_text("<robot name='test'></robot>")
        from model_generation.library.unified_loader import ModelFormat, detect_format

        assert detect_format(f) == ModelFormat.URDF

    def test_mjcf_extension_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "model.mjcf"
        f.write_text("<mujoco model='test'></mujoco>")
        from model_generation.library.unified_loader import ModelFormat, detect_format

        assert detect_format(f) == ModelFormat.MJCF

    def test_xml_with_mujoco_content(self, tmp_path: Path) -> None:
        f = tmp_path / "humanoid.xml"
        f.write_text('<mujoco model="humanoid"><worldbody/></mujoco>')
        from model_generation.library.unified_loader import ModelFormat, detect_format

        assert detect_format(f) == ModelFormat.MJCF

    def test_xml_with_urdf_content(self, tmp_path: Path) -> None:
        f = tmp_path / "robot.xml"
        f.write_text('<robot name="test"><link name="base"/></robot>')
        from model_generation.library.unified_loader import ModelFormat, detect_format

        assert detect_format(f) == ModelFormat.URDF

    def test_unknown_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "model.txt"
        f.write_text("not a model")
        from model_generation.library.unified_loader import ModelFormat, detect_format

        assert detect_format(f) == ModelFormat.UNKNOWN


class TestUserPreferences:
    """Tests for user preferences persistence."""

    def test_default_preferences(self) -> None:
        from model_generation.library.unified_loader import UserPreferences

        prefs = UserPreferences()
        assert prefs.default_model_id == "mujoco_humanoid"
        assert prefs.show_segments is True
        assert prefs.show_joints is True
        assert prefs.show_collisions is True
        assert prefs.show_inertias is True
        assert prefs.show_frames is False

    def test_preferences_roundtrip(self) -> None:
        from model_generation.library.unified_loader import UserPreferences

        prefs = UserPreferences(
            default_model_id="simple_arm",
            show_frames=True,
            show_segments=False,
        )
        data = prefs.to_dict()
        restored = UserPreferences.from_dict(data)
        assert restored.default_model_id == "simple_arm"
        assert restored.show_frames is True
        assert restored.show_segments is False

    def test_recent_models_tracking(self) -> None:
        from model_generation.library.unified_loader import UserPreferences

        prefs = UserPreferences()
        prefs.add_recent("model_a")
        prefs.add_recent("model_b")
        prefs.add_recent("model_a")  # Should move to front

        assert prefs.recent_models[0] == "model_a"
        assert prefs.recent_models[1] == "model_b"
        assert len(prefs.recent_models) == 2

    def test_recent_models_max_limit(self) -> None:
        from model_generation.library.unified_loader import UserPreferences

        prefs = UserPreferences(max_recent=3)
        for i in range(5):
            prefs.add_recent(f"model_{i}")
        assert len(prefs.recent_models) == 3

    def test_save_and_load_preferences(self, tmp_path: Path) -> None:
        from model_generation.library.unified_loader import UnifiedModelLoader

        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        loader.set_default_model("simple_arm")

        # Create a new loader from same dir to test persistence
        loader2 = UnifiedModelLoader(prefs_dir=tmp_path)
        assert loader2.preferences.default_model_id == "simple_arm"


class TestBundledManifest:
    """Tests for the bundled model manifest integrity."""

    def test_manifest_exists(self) -> None:
        manifest_path = (
            Path(__file__).parent.parent / "library" / "bundled" / "manifest.json"
        )
        assert manifest_path.exists(), "Bundled manifest.json must exist"

    def test_manifest_valid_json(self) -> None:
        manifest_path = (
            Path(__file__).parent.parent / "library" / "bundled" / "manifest.json"
        )
        data = json.loads(manifest_path.read_text())
        assert "models" in data
        assert len(data["models"]) >= 4

    def test_all_manifest_files_exist(self) -> None:
        bundled_dir = Path(__file__).parent.parent / "library" / "bundled"
        manifest = json.loads((bundled_dir / "manifest.json").read_text())
        for entry in manifest["models"]:
            model_path = bundled_dir / entry["file"]
            assert model_path.exists(), f"Missing bundled file: {entry['file']}"

    def test_manifest_entries_have_required_fields(self) -> None:
        bundled_dir = Path(__file__).parent.parent / "library" / "bundled"
        manifest = json.loads((bundled_dir / "manifest.json").read_text())
        required = {"id", "name", "format", "file", "category"}
        for entry in manifest["models"]:
            missing = required - set(entry.keys())
            assert not missing, f"Entry {entry.get('id', '?')} missing: {missing}"

    def test_mujoco_humanoid_is_default(self) -> None:
        """The MuJoCo humanoid must be present and tagged as default."""
        bundled_dir = Path(__file__).parent.parent / "library" / "bundled"
        manifest = json.loads((bundled_dir / "manifest.json").read_text())
        ids = [e["id"] for e in manifest["models"]]
        assert "mujoco_humanoid" in ids


class TestConversionApi:
    """Regression tests for conversion error handling."""

    def test_convert_to_urdf_raises_for_missing_source(self, tmp_path: Path) -> None:
        from model_generation.library.unified_loader import (
            ModelNotFoundError,
            UnifiedModelLoader,
        )

        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        with pytest.raises(ModelNotFoundError):
            loader.convert_to_urdf(tmp_path / "missing.mjcf")

    def test_convert_to_urdf_raises_for_unsupported_format(
        self, tmp_path: Path
    ) -> None:
        from model_generation.library.unified_loader import (
            UnifiedModelLoader,
            UnsupportedFormatError,
        )

        source = tmp_path / "bad.txt"
        source.write_text("not a model")
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        with pytest.raises(UnsupportedFormatError):
            loader.convert_to_urdf(source)

    def test_convert_to_urdf_raises_conversion_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from model_generation.library.unified_loader import (
            ConversionError,
            UnifiedModelLoader,
        )

        source = tmp_path / "model.mjcf"
        source.write_text("<mujoco model='x'/>")
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        monkeypatch.setattr(
            loader._mjcf_converter,
            "mjcf_to_urdf",
            lambda _source: (_ for _ in ()).throw(ValueError("boom")),
        )

        with pytest.raises(ConversionError):
            loader.convert_to_urdf(source)

    def test_convert_to_mjcf_raises_for_unsupported_format(
        self, tmp_path: Path
    ) -> None:
        from model_generation.library.unified_loader import (
            UnifiedModelLoader,
            UnsupportedFormatError,
        )

        source = tmp_path / "bad.xml"
        source.write_text("<mujoco model='x'/>")
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        with pytest.raises(UnsupportedFormatError):
            loader.convert_to_mjcf(source)

    def test_convert_to_mjcf_raises_conversion_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from model_generation.library.unified_loader import (
            ConversionError,
            UnifiedModelLoader,
        )

        source = tmp_path / "model.urdf"
        source.write_text('<robot name="x"/>')
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        monkeypatch.setattr(
            loader._mjcf_converter,
            "urdf_to_mjcf",
            lambda _source: (_ for _ in ()).throw(ValueError("boom")),
        )
        with pytest.raises(ConversionError):
            loader.convert_to_mjcf(source)

    @pytest.mark.parametrize(
        ("method_name", "converter_method", "filename", "source_text"),
        [
            (
                "convert_to_urdf",
                "mjcf_to_urdf",
                "model.mjcf",
                "<mujoco model='x'/>",
            ),
            (
                "convert_to_mjcf",
                "urdf_to_mjcf",
                "model.urdf",
                '<robot name="x"/>',
            ),
        ],
    )
    def test_convert_methods_preserve_conversion_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        method_name: str,
        converter_method: str,
        filename: str,
        source_text: str,
    ) -> None:
        from model_generation.library.unified_loader import (
            ConversionError,
            UnifiedModelLoader,
        )

        source = tmp_path / filename
        source.write_text(source_text)
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        monkeypatch.setattr(
            loader._mjcf_converter,
            converter_method,
            lambda _source: (_ for _ in ()).throw(ConversionError("boom")),
        )

        with pytest.raises(ConversionError, match="boom"):
            getattr(loader, method_name)(source)

    def test_convert_to_urdf_wraps_malformed_mjcf_parse_error(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from model_generation.library.unified_loader import (
            ConversionError,
            UnifiedModelLoader,
        )

        source = tmp_path / "broken.mjcf"
        source.write_text("<mujoco model='x'><worldbody></mujoco>")
        loader = UnifiedModelLoader(prefs_dir=tmp_path)

        with (
            caplog.at_level(
                logging.ERROR,
                logger="model_generation.library.unified_loader",
            ),
            pytest.raises(ConversionError) as exc_info,
        ):
            loader.convert_to_urdf(source)

        assert "Unable to convert MJCF source to URDF" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert "MJCF to URDF conversion failed" in caplog.text

    def test_convert_to_mjcf_wraps_malformed_urdf_parse_error(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from model_generation.library.unified_loader import (
            ConversionError,
            UnifiedModelLoader,
        )

        source = tmp_path / "broken.urdf"
        source.write_text("<robot name='x'><link name='base'></robot>")
        loader = UnifiedModelLoader(prefs_dir=tmp_path)

        with (
            caplog.at_level(
                logging.ERROR,
                logger="model_generation.library.unified_loader",
            ),
            pytest.raises(ConversionError) as exc_info,
        ):
            loader.convert_to_mjcf(source)

        assert "Unable to convert URDF source to MJCF" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert "URDF to MJCF conversion failed" in caplog.text

    def test_convert_to_urdf_succeeds_for_valid_source(self, tmp_path: Path) -> None:
        from model_generation.library.unified_loader import UnifiedModelLoader

        source = tmp_path / "good.mjcf"
        source.write_text(
            "<mujoco model='test'><worldbody><body name='base'><geom type='sphere' size='0.1'/></body></worldbody></mujoco>"  # noqa: E501
        )
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        assert loader.convert_to_urdf(source).startswith("<")

    def test_convert_to_mjcf_succeeds_for_valid_source(self, tmp_path: Path) -> None:
        from model_generation.library.unified_loader import UnifiedModelLoader

        source = tmp_path / "good.urdf"
        source.write_text("<robot name='x'><link name='base'/></robot>")
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        assert loader.convert_to_mjcf(source).startswith("<")


class TestUnifiedLoader:
    """Tests for loading bundled models via UnifiedModelLoader."""

    def _make_loader(self, tmp_path: Path) -> Any:
        from model_generation.library.unified_loader import UnifiedModelLoader

        return UnifiedModelLoader(prefs_dir=tmp_path)

    def test_list_bundled_models(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        models = loader.list_bundled_models()
        assert len(models) >= 4
        ids = {m["id"] for m in models}
        assert "mujoco_humanoid" in ids
        assert "simple_humanoid" in ids
        assert "simple_arm" in ids
        assert "simple_quadruped" in ids

    def test_load_mujoco_humanoid(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        result = loader.load_bundled("mujoco_humanoid")
        assert result.success, f"Failed to load mujoco_humanoid: {result.error}"
        assert result.model is not None
        assert result.source_format.value == "mjcf"
        assert len(result.model.links) > 0
        assert len(result.model.joints) > 0

    def test_load_simple_humanoid(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        result = loader.load_bundled("simple_humanoid")
        assert result.success, f"Failed to load simple_humanoid: {result.error}"
        assert result.model is not None
        assert result.source_format.value == "urdf"
        assert len(result.model.links) >= 13

    def test_load_simple_arm(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        result = loader.load_bundled("simple_arm")
        assert result.success, f"Failed to load simple_arm: {result.error}"
        assert result.model is not None
        assert len(result.model.links) >= 5

    def test_load_simple_quadruped(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        result = loader.load_bundled("simple_quadruped")
        assert result.success, f"Failed to load simple_quadruped: {result.error}"
        assert result.model is not None
        assert len(result.model.links) >= 9

    def test_load_default_is_mujoco_humanoid(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        result = loader.load_default()
        assert result.success
        assert result.model is not None
        assert result.model.name == "humanoid"

    def test_load_nonexistent_bundled_model(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        result = loader.load_bundled("nonexistent_model_xyz")
        assert not result.success
        assert result.error is not None

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        loader = self._make_loader(tmp_path)
        result = loader.load_file(tmp_path / "nope.urdf")
        assert not result.success
        assert "not found" in (result.error or "").lower()

    def test_fallback_when_default_missing(self, tmp_path: Path) -> None:
        """If configured default is invalid, fall back to mujoco_humanoid."""
        loader = self._make_loader(tmp_path)
        loader._preferences.default_model_id = "nonexistent_model"
        result = loader.load_default()
        assert result.success
        assert result.model is not None

    def test_load_arbitrary_urdf_file(self, tmp_path: Path) -> None:
        urdf_content = """<?xml version="1.0"?>
<robot name="test_bot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>"""
        f = tmp_path / "test_bot.urdf"
        f.write_text(urdf_content)

        loader = self._make_loader(tmp_path)
        result = loader.load_file(f)
        assert result.success
        assert result.model is not None
        assert result.model.name == "test_bot"

    def test_load_arbitrary_mjcf_file(self, tmp_path: Path) -> None:
        mjcf_content = """<mujoco model="test_mj">
  <worldbody>
    <body name="torso" pos="0 0 1">
      <geom type="sphere" size="0.1"/>
      <joint type="free" name="root"/>
    </body>
  </worldbody>
</mujoco>"""
        f = tmp_path / "test_mj.xml"
        f.write_text(mjcf_content)

        loader = self._make_loader(tmp_path)
        result = loader.load_file(f)
        assert result.success
        assert result.model is not None
        assert result.model.name == "test_mj"


class TestMJCFParsing:
    """Tests for MJCF-specific parsing correctness."""

    def test_mujoco_humanoid_has_bodies(self, tmp_path: Path) -> None:
        from model_generation.library.unified_loader import UnifiedModelLoader

        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        result = loader.load_bundled("mujoco_humanoid")
        assert result.success
        model = result.model
        link_names = {link.name for link in model.links}
        # Key bodies from the MuJoCo humanoid
        assert "torso" in link_names
        assert "pelvis" in link_names
        assert "right_thigh" in link_names
        assert "left_thigh" in link_names
        assert "head" not in link_names  # head is a geom, not a body

    def test_mujoco_humanoid_has_joints(self, tmp_path: Path) -> None:
        from model_generation.library.unified_loader import UnifiedModelLoader

        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        result = loader.load_bundled("mujoco_humanoid")
        model = result.model
        joint_names = {j.name for j in model.joints}
        # Key joints
        assert "right_hip_y" in joint_names or any(
            "hip" in name for name in joint_names
        )

    def test_mjcf_geom_parsing_capsule(self) -> None:
        """Test that capsule geoms are parsed from fromto attribute."""
        import defusedxml.ElementTree as ET
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        xml = """<mujoco model="test">
  <worldbody>
    <body name="link1" pos="0 0 0">
      <geom fromto="0 -0.07 0 0 0.07 0" name="g1" size="0.05" type="capsule"/>
    </body>
  </worldbody>
</mujoco>"""
        root = ET.fromstring(xml)
        model = converter._parse_mjcf(root)
        assert len(model.links) == 1
        link = model.links[0]
        assert link.visual_geometry is not None
        assert link.visual_geometry.geometry_type.value == "capsule"

    def test_capsule_fromto_non_numeric_returns_none(self) -> None:
        """Non-numeric fromto values must be rejected gracefully (#1073)."""
        import defusedxml.ElementTree as ET
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        xml = """<mujoco model="test">
  <worldbody>
    <body name="bad" pos="0 0 0">
      <geom fromto="a b c d e f" size="0.05" type="capsule"/>
    </body>
  </worldbody>
</mujoco>"""
        root = ET.fromstring(xml)
        model = converter._parse_mjcf(root)
        link = model.links[0]
        # Non-numeric fromto should be rejected: geometry is None
        assert link.visual_geometry is None

    def test_capsule_fromto_short_values_returns_none(self) -> None:
        """fromto with fewer than 6 values must be rejected (#1073)."""
        import defusedxml.ElementTree as ET
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        xml = """<mujoco model="test">
  <worldbody>
    <body name="short" pos="0 0 0">
      <geom fromto="0 0 0 1" size="0.05" type="capsule"/>
    </body>
  </worldbody>
</mujoco>"""
        root = ET.fromstring(xml)
        model = converter._parse_mjcf(root)
        link = model.links[0]
        assert link.visual_geometry is None

    def test_capsule_zero_length_degrades_to_sphere(self) -> None:
        """Zero-length capsule (identical from/to) must degrade to sphere (#1073)."""
        import defusedxml.ElementTree as ET
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        xml = """<mujoco model="test">
  <worldbody>
    <body name="zero" pos="0 0 0">
      <geom fromto="0.5 0.5 0.5 0.5 0.5 0.5" size="0.03" type="capsule"/>
    </body>
  </worldbody>
</mujoco>"""
        root = ET.fromstring(xml)
        model = converter._parse_mjcf(root)
        link = model.links[0]
        assert link.visual_geometry is not None
        assert link.visual_geometry.geometry_type.value == "sphere"
        assert abs(link.visual_geometry.dimensions[0] - 0.03) < 0.001

    def test_mjcf_geom_parsing_box(self) -> None:
        import defusedxml.ElementTree as ET
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        xml = """<mujoco model="test">
  <worldbody>
    <body name="foot" pos="0 0 0">
      <geom name="foot_geom" pos="0 0 0.028" size="0.075 0.045 0.025" type="box"/>
    </body>
  </worldbody>
</mujoco>"""
        root = ET.fromstring(xml)
        model = converter._parse_mjcf(root)
        link = model.links[0]
        assert link.visual_geometry is not None
        assert link.visual_geometry.geometry_type.value == "box"
        # MuJoCo uses half-sizes, parser should convert to full sizes
        dims = link.visual_geometry.dimensions
        assert abs(dims[0] - 0.15) < 0.001  # 0.075 * 2
        assert abs(dims[1] - 0.09) < 0.001  # 0.045 * 2

    def test_mjcf_geom_parsing_sphere(self) -> None:
        import defusedxml.ElementTree as ET
        from model_generation.converters.mjcf_converter import MJCFConverter

        converter = MJCFConverter()
        xml = """<mujoco model="test">
  <worldbody>
    <body name="head" pos="0 0 0.19">
      <geom name="head" size="0.09" type="sphere"/>
    </body>
  </worldbody>
</mujoco>"""
        root = ET.fromstring(xml)
        model = converter._parse_mjcf(root)
        link = model.links[0]
        assert link.visual_geometry is not None
        assert link.visual_geometry.geometry_type.value == "sphere"
        assert abs(link.visual_geometry.dimensions[0] - 0.09) < 0.001


class TestModelLibraryBundled:
    """Tests for bundled model registration in ModelLibrary."""

    def test_library_contains_bundled_models(self) -> None:
        from model_generation.library import ModelLibrary

        library = ModelLibrary()
        models = library.list_models()
        ids = {m.id for m in models}
        assert "mujoco_humanoid" in ids

    def test_library_loads_bundled_mjcf(self) -> None:
        from model_generation.library import ModelLibrary

        library = ModelLibrary()
        model = library.load_model("mujoco_humanoid")
        assert model is not None
        assert len(model.links) > 0

    def test_library_loads_bundled_urdf(self) -> None:
        from model_generation.library import ModelLibrary

        library = ModelLibrary()
        model = library.load_model("simple_humanoid")
        assert model is not None
        assert len(model.links) >= 13


class TestDisplayDefaults:
    """Tests for display preview checkbox default values."""

    def test_first_checkbox_is_segments(self) -> None:
        """The first display option must be 'Segments', not 'Collisions'."""
        from model_generation.explorer.display_config import DISPLAY_OPTIONS

        assert DISPLAY_OPTIONS[0][0] == "segments"
        assert DISPLAY_OPTIONS[0][1] == "Segments"

    def test_all_checked_except_frames(self) -> None:
        """All boxes checked by default except Frames."""
        from model_generation.explorer.display_config import DISPLAY_OPTIONS

        for key, label, default in DISPLAY_OPTIONS:
            if key == "frames":
                assert default is False, "Frames should be unchecked by default"
            else:
                assert default is True, f"{label} should be checked by default"

    def test_checkbox_order(self) -> None:
        from model_generation.explorer.display_config import DISPLAY_OPTIONS

        expected_order = ["segments", "joints", "collisions", "inertias", "frames"]
        actual_order = [opt[0] for opt in DISPLAY_OPTIONS]
        assert actual_order == expected_order

    def test_preferences_match_display_defaults(self) -> None:
        from model_generation.library.unified_loader import UserPreferences

        prefs = UserPreferences()
        assert prefs.show_segments is True
        assert prefs.show_joints is True
        assert prefs.show_collisions is True
        assert prefs.show_inertias is True
        assert prefs.show_frames is False


class TestURDFDeterministicFormatting:
    """Verify URDF writer outputs deterministic numeric formatting (#1065)."""

    def test_urdf_deterministic_numeric_formatting(self) -> None:
        """Same model must produce bit-identical URDF on two consecutive writes."""
        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import (
            Geometry,
            GeometryType,
            Inertia,
            Joint,
            JointLimits,
            JointType,
            Link,
            Origin,
        )

        link_a = Link(
            name="base",
            inertia=Inertia(
                ixx=0.00123456789,
                iyy=0.009876,
                izz=0.001010101,
                mass=2.5,
                center_of_mass=(0.0001, -0.002, 0.03),
            ),
            visual_geometry=Geometry(
                geometry_type=GeometryType.BOX,
                dimensions=(0.1, 0.2, 0.3),
            ),
            visual_origin=Origin(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)),
            collision_origin=Origin(),
        )
        link_b = Link(
            name="child",
            inertia=Inertia(ixx=1e-6, iyy=1e-6, izz=1e-6, mass=0.001),
            visual_origin=Origin(),
            collision_origin=Origin(),
        )
        joint = Joint(
            name="j1",
            joint_type=JointType.REVOLUTE,
            parent="base",
            child="child",
            origin=Origin(xyz=(0.0, 0.0, 0.15), rpy=(0.0, 0.0, 0.0)),
            axis=(0, 0, 1),
            limits=JointLimits(lower=-3.14, upper=3.14, effort=10.0, velocity=2.0),
        )

        writer = URDFWriter()
        output_1 = writer.write("test_robot", [link_a, link_b], [joint])
        output_2 = writer.write("test_robot", [link_a, link_b], [joint])
        assert output_1 == output_2, "URDF output is not deterministic across writes"

    def test_urdf_uses_bounded_precision(self) -> None:
        """Numeric values must not exceed 6 significant figures."""
        import re

        from model_generation.builders.urdf_writer import URDFWriter
        from model_generation.core.types import (
            Inertia,
            Joint,
            JointType,
            Link,
            Origin,
        )

        link = Link(
            name="x",
            inertia=Inertia(
                ixx=0.123456789012345,
                iyy=0.987654321098765,
                izz=0.111111111111111,
                mass=1.999999999,
                center_of_mass=(0.123456789, 0.0, 0.0),
            ),
            visual_origin=Origin(),
            collision_origin=Origin(),
        )
        link_b = Link(
            name="y",
            inertia=Inertia(ixx=1e-6, iyy=1e-6, izz=1e-6, mass=0.1),
            visual_origin=Origin(),
            collision_origin=Origin(),
        )
        joint = Joint(
            name="jj",
            joint_type=JointType.FIXED,
            parent="x",
            child="y",
            origin=Origin(),
        )

        writer = URDFWriter()
        xml = writer.write("precision_test", [link, link_b], [joint])

        # Extract all numeric values from attributes
        numbers = re.findall(r'="([^"]*)"', xml)
        for value_str in numbers:
            for part in value_str.split():
                try:
                    float(part)
                    # Verify no more than 6 significant digits in the decimal
                    # representation (:.6g guarantees this)
                    stripped = part.lstrip("-").lstrip("0").replace(".", "")
                    stripped = stripped.lstrip("0")
                    # :.6g can produce up to 6 sig figs
                    assert len(stripped) <= 6, (
                        f"Value '{part}' has more than 6 significant digits"
                    )
                except ValueError:
                    pass  # non-numeric attribute value
