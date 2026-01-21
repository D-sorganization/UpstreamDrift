"""Tests for inertia ellipse visualization module.

Tests for inertia ellipsoid visualization across all physics engines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.python.inertia_ellipse import (
    BodyInertiaData,
    InertiaEllipseConfig,
    InertiaEllipseData,
    InertiaEllipseVisualizer,
    SegmentGroup,
    compute_body_inertia_ellipse,
    compute_composite_inertia,
    compute_composite_inertia_ellipse,
    generate_inertia_ellipse_mesh,
    get_segment_bodies,
    resolve_body_name,
)


class TestBodyInertiaData:
    """Tests for BodyInertiaData class."""

    def test_inertia_world_rotation(self) -> None:
        """Inertia tensor should rotate correctly to world frame."""
        # Diagonal inertia in local frame
        inertia_local = np.diag([1.0, 2.0, 3.0])

        # 90-degree rotation about Z-axis
        rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        body = BodyInertiaData(
            name="test",
            mass=1.0,
            com_world=np.zeros(3),
            inertia_local=inertia_local,
            rotation=rotation,
        )

        I_world = body.inertia_world

        # After 90-degree Z rotation, Ixx and Iyy should swap
        np.testing.assert_allclose(I_world[0, 0], 2.0, atol=1e-10)  # was Iyy
        np.testing.assert_allclose(I_world[1, 1], 1.0, atol=1e-10)  # was Ixx
        np.testing.assert_allclose(I_world[2, 2], 3.0, atol=1e-10)  # Izz unchanged


class TestInertiaEllipseConfig:
    """Tests for InertiaEllipseConfig class."""

    def test_default_config(self) -> None:
        """Default config should have sensible defaults."""
        config = InertiaEllipseConfig()

        assert config.enabled is False
        assert config.segment_group == SegmentGroup.FULL_BODY
        assert config.show_composite is True
        assert config.show_individual is False
        assert 0 <= config.opacity <= 1
        assert config.scale_factor > 0

    def test_config_with_custom_segments(self) -> None:
        """Config should allow custom segment list."""
        config = InertiaEllipseConfig(
            enabled=True,
            segment_group=SegmentGroup.CUSTOM,
            custom_segments=["body1", "body2"],
        )

        assert config.segment_group == SegmentGroup.CUSTOM
        assert config.custom_segments == ["body1", "body2"]


class TestInertiaEllipseComputation:
    """Tests for inertia ellipse computation functions."""

    def test_single_body_ellipse(self) -> None:
        """Single body should produce valid ellipsoid."""
        body = BodyInertiaData(
            name="test",
            mass=1.0,
            com_world=np.array([1.0, 2.0, 3.0]),
            inertia_local=np.diag([1.0, 2.0, 4.0]),
            rotation=np.eye(3),
        )

        ellipse = compute_body_inertia_ellipse(body, scale_factor=1.0)

        assert ellipse is not None
        assert ellipse.total_mass == 1.0
        np.testing.assert_array_equal(ellipse.center, [1.0, 2.0, 3.0])
        assert len(ellipse.principal_moments) == 3
        assert len(ellipse.radii) == 3

        # Radii should be inversely related to sqrt of principal moments
        # Smallest moment -> largest radius
        assert ellipse.radii[0] > ellipse.radii[1] > ellipse.radii[2]

    def test_composite_inertia_two_bodies(self) -> None:
        """Composite inertia should combine two bodies correctly."""
        body1 = BodyInertiaData(
            name="body1",
            mass=1.0,
            com_world=np.array([0.0, 0.0, 0.0]),
            inertia_local=np.diag([1.0, 1.0, 1.0]),
            rotation=np.eye(3),
        )

        body2 = BodyInertiaData(
            name="body2",
            mass=1.0,
            com_world=np.array([2.0, 0.0, 0.0]),
            inertia_local=np.diag([1.0, 1.0, 1.0]),
            rotation=np.eye(3),
        )

        total_mass, com, I_total = compute_composite_inertia([body1, body2])

        # Total mass should be sum
        assert total_mass == 2.0

        # COM should be midpoint
        np.testing.assert_allclose(com, [1.0, 0.0, 0.0])

        # Inertia should be larger due to parallel axis theorem
        # Each body contributes its local inertia plus m * d^2
        # where d is distance from combined COM to body COM
        assert I_total[0, 0] > 2.0  # Ixx increases due to parallel axis
        assert I_total[1, 1] > 2.0  # Iyy increases
        assert I_total[2, 2] > 2.0  # Izz increases

    def test_composite_ellipse_from_bodies(self) -> None:
        """Composite ellipse should be computed from multiple bodies."""
        bodies = [
            BodyInertiaData(
                name=f"body{i}",
                mass=1.0,
                com_world=np.array([float(i), 0.0, 0.0]),
                inertia_local=np.diag([0.1, 0.1, 0.1]),
                rotation=np.eye(3),
            )
            for i in range(3)
        ]

        ellipse = compute_composite_inertia_ellipse(
            bodies,
            segment_group=SegmentGroup.FULL_BODY,
            scale_factor=1.0,
        )

        assert ellipse is not None
        assert ellipse.total_mass == 3.0
        assert len(ellipse.body_names) == 3
        assert ellipse.segment_group == SegmentGroup.FULL_BODY

    def test_empty_body_list_returns_none(self) -> None:
        """Empty body list should return None."""
        ellipse = compute_composite_inertia_ellipse([])
        assert ellipse is None


class TestBodyNameResolution:
    """Tests for body name resolution with aliases."""

    def test_direct_match(self) -> None:
        """Direct name match should work."""
        available = ["pelvis", "torso", "head"]
        result = resolve_body_name("torso", available)
        assert result == "torso"

    def test_alias_match(self) -> None:
        """Alias should be resolved to available name."""
        available = ["Pelvis", "Torso", "Head"]
        result = resolve_body_name("pelvis", available)
        assert result == "Pelvis"

    def test_case_insensitive_fallback(self) -> None:
        """Case-insensitive search should work as fallback."""
        available = ["PELVIS", "TORSO", "HEAD"]
        result = resolve_body_name("pelvis", available)
        assert result == "PELVIS"

    def test_unknown_name_returns_none(self) -> None:
        """Unknown name should return None."""
        available = ["pelvis", "torso", "head"]
        result = resolve_body_name("unknown_body", available)
        assert result is None


class TestSegmentGroupResolution:
    """Tests for segment group body resolution."""

    def test_full_body_group(self) -> None:
        """Full body group should resolve available bodies."""
        available = ["pelvis", "torso", "head", "l_hand", "r_hand"]
        bodies = get_segment_bodies(SegmentGroup.FULL_BODY, available)

        # Should include available bodies from the full body list
        assert "pelvis" in bodies
        assert "torso" in bodies
        assert "head" in bodies

    def test_upper_body_excludes_legs(self) -> None:
        """Upper body group should not include leg bodies."""
        available = ["pelvis", "torso", "head", "l_thigh", "r_thigh"]
        bodies = get_segment_bodies(SegmentGroup.UPPER_BODY, available)

        # Should include upper body parts
        assert "torso" in bodies
        assert "head" in bodies

        # Should not include leg parts
        assert "l_thigh" not in bodies
        assert "r_thigh" not in bodies

    def test_custom_group(self) -> None:
        """Custom group should use provided body list."""
        available = ["body1", "body2", "body3"]
        custom = ["body1", "body3"]
        bodies = get_segment_bodies(
            SegmentGroup.CUSTOM, available, custom_segments=custom
        )

        assert bodies == ["body1", "body3"]


class TestInertiaEllipseMesh:
    """Tests for inertia ellipse mesh generation."""

    def test_mesh_generation(self) -> None:
        """Generated mesh should have valid vertices and faces."""
        ellipse = InertiaEllipseData(
            center=np.array([0.0, 0.0, 0.0]),
            radii=np.array([1.0, 2.0, 0.5]),
            axes=np.eye(3),
            principal_moments=np.array([1.0, 0.25, 4.0]),
            total_mass=1.0,
            body_names=["test"],
        )

        vertices, faces = generate_inertia_ellipse_mesh(ellipse)

        assert vertices.shape[0] > 0
        assert vertices.shape[1] == 3
        assert faces.shape[0] > 0
        assert faces.shape[1] == 3
        assert np.all(faces >= 0)
        assert np.all(faces < vertices.shape[0])

    def test_mesh_extent_matches_radii(self) -> None:
        """Mesh extent should match ellipsoid radii."""
        radii = np.array([2.0, 3.0, 1.0])
        ellipse = InertiaEllipseData(
            center=np.array([0.0, 0.0, 0.0]),
            radii=radii,
            axes=np.eye(3),
            principal_moments=1.0 / radii**2,
            total_mass=1.0,
            body_names=["test"],
        )

        vertices, _ = generate_inertia_ellipse_mesh(ellipse)

        max_extent = np.max(np.abs(vertices), axis=0)
        np.testing.assert_allclose(max_extent, radii, atol=0.1)


class TestInertiaEllipseVisualizer:
    """Tests for InertiaEllipseVisualizer class."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for visualizer testing."""

        class MockEngine:
            def get_body_names(self) -> list[str]:
                return ["pelvis", "torso", "head", "l_hand", "r_hand"]

            def get_body_inertia_data(self, body_name: str) -> BodyInertiaData | None:
                if body_name in self.get_body_names():
                    return BodyInertiaData(
                        name=body_name,
                        mass=1.0,
                        com_world=np.array([0.0, 0.0, 0.0]),
                        inertia_local=np.diag([0.1, 0.1, 0.1]),
                        rotation=np.eye(3),
                    )
                return None

            def get_all_body_inertia_data(self) -> list[BodyInertiaData]:
                return [
                    self.get_body_inertia_data(name)
                    for name in self.get_body_names()
                    if self.get_body_inertia_data(name) is not None
                ]

        return MockEngine()

    def test_compute_ellipses_disabled(self, mock_engine) -> None:
        """Disabled visualizer should return empty dict."""
        config = InertiaEllipseConfig(enabled=False)
        viz = InertiaEllipseVisualizer(mock_engine, config)

        results = viz.compute_ellipses()
        assert results == {}

    def test_compute_ellipses_composite_only(self, mock_engine) -> None:
        """Should compute only composite ellipse when configured."""
        config = InertiaEllipseConfig(
            enabled=True,
            segment_group=SegmentGroup.FULL_BODY,
            show_composite=True,
            show_individual=False,
        )
        viz = InertiaEllipseVisualizer(mock_engine, config)

        results = viz.compute_ellipses()

        assert "composite" in results
        # Individual bodies should not be in results
        assert "pelvis" not in results
        assert "torso" not in results

    def test_compute_ellipses_with_individual(self, mock_engine) -> None:
        """Should include individual ellipses when configured."""
        config = InertiaEllipseConfig(
            enabled=True,
            segment_group=SegmentGroup.FULL_BODY,
            show_composite=True,
            show_individual=True,
        )
        viz = InertiaEllipseVisualizer(mock_engine, config)

        results = viz.compute_ellipses()

        assert "composite" in results
        # Should include individual bodies that are in the segment group
        assert any(key in results for key in ["pelvis", "torso", "head"])

    def test_set_segment_group(self, mock_engine) -> None:
        """Changing segment group should clear cache."""
        config = InertiaEllipseConfig(enabled=True)
        viz = InertiaEllipseVisualizer(mock_engine, config)

        viz.compute_ellipses()
        assert len(viz.get_cached_ellipses()) > 0

        viz.set_segment_group(SegmentGroup.UPPER_BODY)
        assert len(viz.get_cached_ellipses()) == 0

    def test_get_inertia_summary(self, mock_engine) -> None:
        """Summary should contain expected metrics."""
        config = InertiaEllipseConfig(
            enabled=True,
            show_composite=True,
        )
        viz = InertiaEllipseVisualizer(mock_engine, config)

        summary = viz.get_inertia_summary()

        assert "segment_group" in summary
        assert "total_mass" in summary
        assert "center_of_mass" in summary
        assert "principal_moments" in summary
        assert "body_count" in summary


class TestSegmentGroups:
    """Tests for specific segment group definitions."""

    def test_all_segment_groups_defined(self) -> None:
        """All segment groups should have default mappings."""
        from shared.python.inertia_ellipse import DEFAULT_SEGMENT_MAPPINGS

        for group in SegmentGroup:
            assert group in DEFAULT_SEGMENT_MAPPINGS

    def test_club_included_in_with_club_groups(self) -> None:
        """Club bodies should be in '*_with_club' groups."""
        from shared.python.inertia_ellipse import DEFAULT_SEGMENT_MAPPINGS

        full_body_with_club = DEFAULT_SEGMENT_MAPPINGS[SegmentGroup.FULL_BODY_WITH_CLUB]
        upper_body_with_club = DEFAULT_SEGMENT_MAPPINGS[SegmentGroup.UPPER_BODY_WITH_CLUB]

        assert any("club" in body for body in full_body_with_club)
        assert any("club" in body for body in upper_body_with_club)

    def test_club_not_in_regular_groups(self) -> None:
        """Club bodies should not be in regular groups."""
        from shared.python.inertia_ellipse import DEFAULT_SEGMENT_MAPPINGS

        full_body = DEFAULT_SEGMENT_MAPPINGS[SegmentGroup.FULL_BODY]
        upper_body = DEFAULT_SEGMENT_MAPPINGS[SegmentGroup.UPPER_BODY]

        assert not any("club" in body for body in full_body)
        assert not any("club" in body for body in upper_body)


class TestConfigurationManagerIntegration:
    """Tests for configuration manager integration."""

    def test_config_to_inertia_ellipse_config(self) -> None:
        """SimulationConfig should convert to InertiaEllipseConfig."""
        from shared.python.configuration_manager import SimulationConfig

        sim_config = SimulationConfig(
            inertia_ellipse_enabled=True,
            inertia_ellipse_segment_group="upper_body_with_club",
            inertia_ellipse_show_individual=True,
            inertia_ellipse_opacity=0.7,
            inertia_ellipse_scale=2.0,
        )

        ellipse_config = sim_config.get_inertia_ellipse_config()

        assert ellipse_config.enabled is True
        assert ellipse_config.segment_group == SegmentGroup.UPPER_BODY_WITH_CLUB
        assert ellipse_config.show_individual is True
        assert ellipse_config.opacity == 0.7
        assert ellipse_config.scale_factor == 2.0

    def test_config_validation_invalid_segment_group(self) -> None:
        """Invalid segment group should raise validation error."""
        from shared.python.common_utils import GolfModelingError
        from shared.python.configuration_manager import SimulationConfig

        config = SimulationConfig(
            inertia_ellipse_segment_group="invalid_group"
        )

        with pytest.raises(GolfModelingError):
            config.validate()

    def test_config_validation_invalid_opacity(self) -> None:
        """Invalid opacity should raise validation error."""
        from shared.python.common_utils import GolfModelingError
        from shared.python.configuration_manager import SimulationConfig

        config = SimulationConfig(
            inertia_ellipse_opacity=1.5  # > 1
        )

        with pytest.raises(GolfModelingError):
            config.validate()
