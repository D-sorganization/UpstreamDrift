"""
Tests for the editor module (Frankenstein Editor and Text Editor).
"""

import tempfile
from pathlib import Path

# Sample URDF for testing
SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="simple_robot">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <link name="arm_link">
        <inertial>
            <mass value="0.5"/>
            <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <joint name="base_to_arm" type="revolute">
        <parent link="base_link"/>
        <child link="arm_link"/>
        <origin xyz="0 0 0.5" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>
</robot>
"""

TWO_ARM_URDF = """<?xml version="1.0"?>
<robot name="two_arm_robot">
    <link name="torso">
        <inertial>
            <mass value="5.0"/>
            <inertia ixx="0.5" iyy="0.5" izz="0.5" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <link name="left_arm">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <link name="right_arm">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <joint name="torso_to_left" type="revolute">
        <parent link="torso"/>
        <child link="left_arm"/>
        <origin xyz="0 0.3 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>
    <joint name="torso_to_right" type="revolute">
        <parent link="torso"/>
        <child link="right_arm"/>
        <origin xyz="0 -0.3 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>
</robot>
"""


class TestFrankensteinEditor:
    """Tests for FrankensteinEditor class."""

    def test_editor_creation(self):
        """Test editor instantiation."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        assert editor is not None
        assert editor.list_models() == []

    def test_load_model_from_string(self):
        """Test loading a model from URDF string."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        model = editor.load_model("test", SIMPLE_URDF)

        assert model is not None
        assert model.name == "simple_robot"
        assert len(model.links) == 2
        assert len(model.joints) == 1
        assert "test" in editor.list_models()

    def test_create_model(self):
        """Test creating a new empty model."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        model = editor.create_model("new_model", "my_robot")

        assert model is not None
        assert model.name == "my_robot"
        assert len(model.links) == 1  # Base link
        assert "new_model" in editor.list_models()

    def test_duplicate_model(self):
        """Test duplicating a model."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("original", SIMPLE_URDF)

        copy = editor.duplicate_model("original", "copy")
        assert copy is not None
        assert "copy" in editor.list_models()

        # Modifications to copy shouldn't affect original
        original = editor.get_model("original")
        assert original is not None
        assert len(original.links) == len(copy.links)

    def test_copy_link(self):
        """Test copying a single link to clipboard."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("source", SIMPLE_URDF)

        result = editor.copy_link("source", "arm_link")
        assert result is True

        info = editor.get_clipboard_info()
        assert not info["empty"]
        assert info["link_count"] == 1
        assert "arm_link" in info["link_names"]

    def test_copy_subtree(self):
        """Test copying a subtree to clipboard."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("source", TWO_ARM_URDF)

        result = editor.copy_subtree("source", "left_arm")
        assert result is True

        info = editor.get_clipboard_info()
        assert not info["empty"]
        assert info["link_count"] >= 1

    def test_paste_to_model(self):
        """Test pasting clipboard contents to a model."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("source", SIMPLE_URDF)
        editor.create_model("target", "target_robot")

        # Copy arm_link from source
        editor.copy_subtree("source", "arm_link")

        # Paste to target
        created = editor.paste("target", attach_to="base_link")
        assert len(created) >= 1

        target = editor.get_model("target")
        assert target is not None
        link_names = [link.name for link in target.links]
        assert "arm_link" in link_names or any("arm" in n for n in link_names)

    def test_delete_link(self):
        """Test deleting a link."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("test", SIMPLE_URDF, read_only=False)

        # Duplicate to make editable
        editor.duplicate_model("test", "editable")

        result = editor.delete_link("editable", "arm_link", reparent_children=False)
        assert result is True

        model = editor.get_model("editable")
        assert model is not None
        link_names = [link.name for link in model.links]
        assert "arm_link" not in link_names

    def test_rename_link(self):
        """Test renaming a link."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("test", SIMPLE_URDF)
        editor.duplicate_model("test", "editable")

        result = editor.rename_link("editable", "arm_link", "new_arm_name")
        assert result is True

        model = editor.get_model("editable")
        assert model is not None
        link_names = [link.name for link in model.links]
        assert "new_arm_name" in link_names
        assert "arm_link" not in link_names

        # Check joint references updated
        joint = model.joints[0]
        assert joint.child == "new_arm_name"

    def test_undo_redo(self):
        """Test undo/redo functionality."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("test", SIMPLE_URDF)
        editor.duplicate_model("test", "editable")

        model = editor.get_model("editable")
        assert model is not None
        original_count = len(model.links)

        # Delete a link
        editor.delete_link("editable", "arm_link", reparent_children=False)
        model = editor.get_model("editable")
        assert model is not None
        assert len(model.links) < original_count

        # Undo
        result = editor.undo()
        assert result is True
        model = editor.get_model("editable")
        assert model is not None
        assert len(model.links) == original_count

        # Redo
        result = editor.redo()
        assert result is True
        model = editor.get_model("editable")
        assert model is not None
        assert len(model.links) < original_count

    def test_export_model(self):
        """Test exporting a model to URDF string."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("test", SIMPLE_URDF)

        urdf_string = editor.export_model("test")
        assert urdf_string is not None
        assert "<robot" in urdf_string
        assert "simple_robot" in urdf_string

    def test_compare_models(self):
        """Test comparing two models."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("a", SIMPLE_URDF)
        editor.load_model("b", TWO_ARM_URDF)

        comparison = editor.compare_models("a", "b")
        assert "links" in comparison
        assert "joints" in comparison
        assert "stats" in comparison

    def test_get_model_statistics(self):
        """Test getting model statistics."""
        from model_generation.editor import FrankensteinEditor

        editor = FrankensteinEditor()
        editor.load_model("test", SIMPLE_URDF)

        stats = editor.get_model_statistics("test")
        assert stats["link_count"] == 2
        assert stats["joint_count"] == 1
        assert "total_mass" in stats


class TestURDFTextEditor:
    """Tests for URDFTextEditor class."""

    def test_editor_creation(self):
        """Test editor instantiation."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        assert editor is not None

    def test_load_string(self):
        """Test loading URDF from string."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF)

        content = editor.get_content()
        assert content == SIMPLE_URDF

    def test_load_file(self):
        """Test loading URDF from file."""
        from model_generation.editor import URDFTextEditor

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(SIMPLE_URDF)
            temp_path = Path(f.name)

        try:
            editor = URDFTextEditor()
            content = editor.load_file(temp_path)
            assert content == SIMPLE_URDF
        finally:
            temp_path.unlink()

    def test_validate_valid_urdf(self):
        """Test validation of valid URDF."""
        from model_generation.editor import URDFTextEditor, ValidationSeverity

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF)

        messages = editor.validate()
        errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0

    def test_validate_invalid_xml(self):
        """Test validation catches invalid XML."""
        from model_generation.editor import URDFTextEditor, ValidationSeverity

        invalid_urdf = "<robot><link name='test'></robot>"  # Missing closing tag

        editor = URDFTextEditor()
        editor.load_string(invalid_urdf)

        messages = editor.validate()
        errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]
        assert len(errors) > 0

    def test_validate_missing_link_reference(self):
        """Test validation catches missing link references."""
        from model_generation.editor import URDFTextEditor, ValidationSeverity

        urdf_with_bad_ref = """<?xml version="1.0"?>
        <robot name="bad_robot">
            <link name="base_link"/>
            <joint name="test_joint" type="fixed">
                <parent link="base_link"/>
                <child link="nonexistent_link"/>
            </joint>
        </robot>
        """

        editor = URDFTextEditor()
        editor.load_string(urdf_with_bad_ref)

        messages = editor.validate()
        errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]
        assert len(errors) > 0
        assert any("nonexistent" in m.message for m in errors)

    def test_set_content(self):
        """Test setting content."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF)

        new_content = SIMPLE_URDF.replace("simple_robot", "modified_robot")
        editor.set_content(new_content, "Rename robot")

        assert "modified_robot" in editor.get_content()

    def test_diff_from_original(self):
        """Test generating diff from original."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF)

        # Make a change
        new_content = SIMPLE_URDF.replace("simple_robot", "modified_robot")
        editor.set_content(new_content, validate=False)

        diff = editor.get_diff_from_original()
        assert diff.has_changes
        assert diff.additions > 0 or diff.deletions > 0
        assert "modified_robot" in diff.unified_diff

    def test_undo_redo(self):
        """Test undo/redo functionality."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF)

        # Make changes
        editor.set_content(SIMPLE_URDF + "<!-- comment -->", validate=False)
        assert "<!-- comment -->" in editor.get_content()

        # Undo
        result = editor.undo()
        assert result is True
        assert "<!-- comment -->" not in editor.get_content()

        # Redo
        result = editor.redo()
        assert result is True
        assert "<!-- comment -->" in editor.get_content()

    def test_has_unsaved_changes(self):
        """Test unsaved changes detection."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF)

        assert not editor.has_unsaved_changes()

        editor.set_content(SIMPLE_URDF + "<!-- change -->", validate=False)
        assert editor.has_unsaved_changes()

    def test_find_text(self):
        """Test text search."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF)

        results = editor.find_text("link")
        assert len(results) > 0

        # Test regex
        results = editor.find_text(r"mass\s+value", regex=True)
        assert len(results) > 0

    def test_replace_all(self):
        """Test replace all functionality."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF)

        count = editor.replace_all("link", "LINK")
        assert count > 0
        assert "LINK" in editor.get_content()

    def test_get_history(self):
        """Test getting version history."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF, "Initial load")
        editor.set_content(SIMPLE_URDF + "<!-- v2 -->", "Version 2", validate=False)
        editor.set_content(SIMPLE_URDF + "<!-- v3 -->", "Version 3", validate=False)

        history = editor.get_history()
        assert len(history) == 3

    def test_go_to_version(self):
        """Test navigating to specific version."""
        from model_generation.editor import URDFTextEditor

        editor = URDFTextEditor()
        editor.load_string(SIMPLE_URDF, "v1")
        editor.set_content(SIMPLE_URDF + "<!-- v2 -->", "v2", validate=False)
        editor.set_content(SIMPLE_URDF + "<!-- v3 -->", "v3", validate=False)

        # Go to first version
        result = editor.go_to_version(0)
        assert result is True
        assert "<!-- v3 -->" not in editor.get_content()
