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
from pathlib import Path


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
        manifest_path = Path(__file__).parent.parent / "library" / "bundled" / "manifest.json"
        assert manifest_path.exists(), "Bundled manifest.json must exist"

    def test_manifest_valid_json(self) -> None:
        manifest_path = Path(__file__).parent.parent / "library" / "bundled" / "manifest.json"
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


class TestUnifiedLoader:
    """Tests for loading bundled models via UnifiedModelLoader."""

    def _make_loader(self, tmp_path: Path):
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
        assert "right_hip_y" in joint_names or any("hip" in name for name in joint_names)

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
