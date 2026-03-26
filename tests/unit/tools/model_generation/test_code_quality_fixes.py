"""
Tests for code quality fixes from the URDF review.

Validates:
1. Deque optimization doesn't break _get_descendants() or _sort_links_by_hierarchy()
2. Preset extraction works consistently in quick_urdf() and quick_build()
3. Capsule geometry produces a warning when downgraded to cylinder
4. _get_writable_model() helper works correctly in editor_modifications
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from model_generation.builders.manual_builder import ManualBuilder
from model_generation.builders.urdf_writer import URDFWriter
from model_generation.core.types import (
    Geometry,
    Inertia,
    Joint,
    JointType,
    Link,
)

# ============================================================
# Issue 1: Deque optimization - _get_descendants()
# ============================================================


class TestGetDescendants:
    """Validate _get_descendants() returns correct results after deque refactor."""

    def _build_chain(self) -> ManualBuilder:
        """Build a simple chain: base -> arm -> hand."""
        builder = ManualBuilder("test_robot", validate_on_add=False)
        builder.add_link(Link(name="base", inertia=Inertia.from_box(10, 1, 1, 0.5)))
        builder.add_link(Link(name="arm", inertia=Inertia.from_cylinder(2, 0.05, 0.5)))
        builder.add_link(Link(name="hand", inertia=Inertia.from_sphere(0.5, 0.05)))
        builder.add_joint(
            Joint(
                name="base_to_arm",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="arm",
            )
        )
        builder.add_joint(
            Joint(
                name="arm_to_hand",
                joint_type=JointType.REVOLUTE,
                parent="arm",
                child="hand",
            )
        )
        return builder

    def test_descendants_of_root(self) -> None:
        """Root should have all other links as descendants."""
        builder = self._build_chain()
        descendants = builder._get_descendants("base")
        assert descendants == {"arm", "hand"}

    def test_descendants_of_middle(self) -> None:
        """Middle link should have only its children as descendants."""
        builder = self._build_chain()
        descendants = builder._get_descendants("arm")
        assert descendants == {"hand"}

    def test_descendants_of_leaf(self) -> None:
        """Leaf node should have no descendants."""
        builder = self._build_chain()
        descendants = builder._get_descendants("hand")
        assert descendants == set()

    def test_descendants_branching(self) -> None:
        """Test with branching structure: base -> (left_arm, right_arm)."""
        builder = ManualBuilder("test_robot", validate_on_add=False)
        builder.add_link(Link(name="base", inertia=Inertia.from_box(10, 1, 1, 0.5)))
        builder.add_link(Link(name="left_arm", inertia=Inertia.from_cylinder(2, 0.05, 0.5)))
        builder.add_link(Link(name="right_arm", inertia=Inertia.from_cylinder(2, 0.05, 0.5)))
        builder.add_link(Link(name="left_hand", inertia=Inertia.from_sphere(0.5, 0.05)))
        builder.add_joint(
            Joint(
                name="base_to_left",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="left_arm",
            )
        )
        builder.add_joint(
            Joint(
                name="base_to_right",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="right_arm",
            )
        )
        builder.add_joint(
            Joint(
                name="left_to_hand",
                joint_type=JointType.REVOLUTE,
                parent="left_arm",
                child="left_hand",
            )
        )
        descendants = builder._get_descendants("base")
        assert descendants == {"left_arm", "right_arm", "left_hand"}


# ============================================================
# Issue 1: Deque optimization - _sort_links_by_hierarchy()
# ============================================================


class TestSortLinksByHierarchy:
    """Validate _sort_links_by_hierarchy() returns parents before children."""

    def test_simple_chain(self) -> None:
        """Parents must appear before children in sorted order."""
        writer = URDFWriter()
        links = [
            Link(name="hand", inertia=Inertia.from_sphere(0.5, 0.05)),
            Link(name="base", inertia=Inertia.from_box(10, 1, 1, 0.5)),
            Link(name="arm", inertia=Inertia.from_cylinder(2, 0.05, 0.5)),
        ]
        joints = [
            Joint(
                name="base_to_arm",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="arm",
            ),
            Joint(
                name="arm_to_hand",
                joint_type=JointType.REVOLUTE,
                parent="arm",
                child="hand",
            ),
        ]
        sorted_links = writer._sort_links_by_hierarchy(links, joints)
        names = [link.name for link in sorted_links]
        assert names.index("base") < names.index("arm")
        assert names.index("arm") < names.index("hand")

    def test_all_links_present(self) -> None:
        """All links should be present in sorted output."""
        writer = URDFWriter()
        links = [
            Link(name="a", inertia=Inertia.from_box(1, 0.1, 0.1, 0.1)),
            Link(name="b", inertia=Inertia.from_box(1, 0.1, 0.1, 0.1)),
            Link(name="c", inertia=Inertia.from_box(1, 0.1, 0.1, 0.1)),
        ]
        joints = [
            Joint(
                name="a_to_b",
                joint_type=JointType.FIXED,
                parent="a",
                child="b",
            ),
            Joint(
                name="b_to_c",
                joint_type=JointType.FIXED,
                parent="b",
                child="c",
            ),
        ]
        sorted_links = writer._sort_links_by_hierarchy(links, joints)
        assert len(sorted_links) == 3
        assert {link.name for link in sorted_links} == {"a", "b", "c"}


# ============================================================
# Issue 3: Preset duplication - quick_urdf() and quick_build()
# ============================================================


class TestPresetConsistency:
    """Ensure presets produce consistent results in quick_urdf and quick_build."""

    def test_presets_dict_exists_as_module_constant(self) -> None:
        """_HUMANOID_PRESETS should be a module-level constant after extraction."""
        import model_generation

        assert hasattr(model_generation, "_HUMANOID_PRESETS")
        presets = model_generation._HUMANOID_PRESETS
        assert "athletic" in presets
        assert "average" in presets
        assert "heavy" in presets
        assert "lean" in presets

    def test_athletic_preset_config(self) -> None:
        """Athletic preset should have expected parameters."""
        import model_generation

        presets = model_generation._HUMANOID_PRESETS
        assert presets["athletic"]["gender_factor"] == 0.7
        assert presets["athletic"]["shoulder_width_factor"] == 1.1

    @patch("model_generation.builders.parametric_builder.ParametricBuilder")
    def test_quick_urdf_uses_shared_presets(self, mock_builder_cls: MagicMock) -> None:
        """quick_urdf should use _HUMANOID_PRESETS, not an inline dict."""
        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.urdf_xml = "<robot/>"
        mock_builder.build.return_value = mock_result

        from model_generation import quick_urdf

        quick_urdf(height_m=1.80, preset="athletic")

        mock_builder.set_parameters.assert_called_once_with(
            height_m=1.80,
            mass_kg=75.0,
            gender_factor=0.7,
            shoulder_width_factor=1.1,
        )

    @patch("model_generation.builders.parametric_builder.ParametricBuilder")
    def test_quick_build_uses_shared_presets(self, mock_builder_cls: MagicMock) -> None:
        """quick_build should use _HUMANOID_PRESETS, not an inline dict."""
        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.urdf_xml = None
        mock_builder.build.return_value = mock_result

        from model_generation import quick_build

        quick_build(height_m=1.80, preset="athletic")

        mock_builder.set_parameters.assert_called_once_with(
            height_m=1.80,
            mass_kg=75.0,
            gender_factor=0.7,
            shoulder_width_factor=1.1,
        )


# ============================================================
# Issue 4: Capsule geometry warning
# ============================================================


class TestCapsuleWarning:
    """Capsule downgrade to cylinder should produce a warning."""

    def test_urdf_writer_capsule_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """URDFWriter._write_geometry() should warn about capsule approximation."""
        writer = URDFWriter()
        capsule = Geometry.capsule(radius=0.05, length=0.3)

        with caplog.at_level(logging.WARNING):
            lines = writer._write_geometry(capsule, level=0)

        # Should still produce valid cylinder XML
        xml = "\n".join(lines)
        assert "cylinder" in xml
        assert 'radius="0.05"' in xml

        # Should produce a warning
        assert any(
            "Capsule geometry approximated as cylinder" in record.message
            for record in caplog.records
        ), f"Expected capsule warning, got: {[r.message for r in caplog.records]}"

    def test_geometry_to_urdf_string_capsule_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Geometry.to_urdf_string() should warn about capsule approximation."""
        capsule = Geometry.capsule(radius=0.05, length=0.3)

        with caplog.at_level(logging.WARNING):
            result = capsule.to_urdf_string()

        # Should produce cylinder XML
        assert "cylinder" in result

        # Should produce a warning
        assert any(
            "Capsule geometry approximated as cylinder" in record.message
            for record in caplog.records
        ), f"Expected capsule warning, got: {[r.message for r in caplog.records]}"

    def test_non_capsule_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Non-capsule geometries should NOT produce the capsule warning."""
        writer = URDFWriter()
        cylinder = Geometry.cylinder(radius=0.05, length=0.3)

        with caplog.at_level(logging.WARNING):
            writer._write_geometry(cylinder, level=0)

        assert not any(
            "Capsule geometry approximated as cylinder" in record.message
            for record in caplog.records
        )


# ============================================================
# Issue 5: _get_writable_model() helper in editor_modifications
# ============================================================


class TestGetWritableModel:
    """Validate _get_writable_model() helper extraction."""

    def test_helper_exists(self) -> None:
        """_get_writable_model should exist as a method on ModificationMixin."""
        from model_generation.editor.editor_modifications import ModificationMixin

        assert hasattr(ModificationMixin, "_get_writable_model")

    def test_helper_returns_none_for_missing_model(self) -> None:
        """Should return None and log error for missing model."""
        from model_generation.editor.editor_modifications import ModificationMixin

        mixin = ModificationMixin.__new__(ModificationMixin)
        mixin._models = {}

        result = mixin._get_writable_model("nonexistent")
        assert result is None

    def test_helper_returns_none_for_read_only_model(self) -> None:
        """Should return None and log error for read-only model."""
        from model_generation.editor.editor_modifications import ModificationMixin

        mixin = ModificationMixin.__new__(ModificationMixin)
        mock_model = MagicMock()
        mock_model.read_only = True
        mixin._models = {"test": mock_model}

        result = mixin._get_writable_model("test")
        assert result is None

    def test_helper_returns_model_for_writable(self) -> None:
        """Should return the model for a writable model."""
        from model_generation.editor.editor_modifications import ModificationMixin

        mixin = ModificationMixin.__new__(ModificationMixin)
        mock_model = MagicMock()
        mock_model.read_only = False
        mixin._models = {"test": mock_model}

        result = mixin._get_writable_model("test")
        assert result is mock_model


# ============================================================
# Issue 2: Redundant imports - just verify math works at module level
# ============================================================


class TestRedundantImports:
    """Verify math.radians works correctly after removing inline imports."""

    def test_link_from_dict_uses_math(self) -> None:
        """_link_from_dict should work correctly with module-level math import."""
        builder = ManualBuilder("test", validate_on_add=False)
        data = {
            "name": "test_link",
            "geometry": {
                "shape": "box",
                "dimensions": {"width": 0.1, "height": 0.2, "length": 0.3},
                "orientation": {"roll": 90.0, "pitch": 0.0, "yaw": 0.0},
            },
        }
        link = builder._link_from_dict(data)
        assert link.name == "test_link"
        # 90 degrees should be approximately pi/2
        import math

        assert abs(link.visual_origin.rpy[0] - math.pi / 2) < 1e-10

    def test_joint_from_dict_uses_math(self) -> None:
        """_joint_from_dict should work correctly with module-level math import."""
        builder = ManualBuilder("test", validate_on_add=False)
        data = {
            "name": "child",
            "parent": "parent",
            "geometry": {
                "orientation": {"roll": 0.0, "pitch": 45.0, "yaw": 0.0},
            },
            "joint": {"type": "revolute", "limits": {"lower": -90, "upper": 90}},
        }
        joint = builder._joint_from_dict(data)
        import math

        assert abs(joint.origin.rpy[1] - math.radians(45.0)) < 1e-10
        assert joint.limits is not None
        assert abs(joint.limits.lower - math.radians(-90)) < 1e-10
