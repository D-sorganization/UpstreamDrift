"""Unit tests for parallel mechanism generators."""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms.parallel_mechanisms import (
    generate_delta_robot_xml,
    generate_five_bar_parallel_xml,
    generate_pantograph_xml,
    generate_stewart_platform_xml,
)


def test_generate_pantograph_xml():
    """Test generating pantograph mechanism XML."""
    xml = generate_pantograph_xml()
    assert isinstance(xml, str)
    assert '<mujoco model="pantograph">' in xml
    assert "arm1_motor" in xml


def test_generate_delta_robot_xml():
    """Test generating Delta robot mechanism XML."""
    xml = generate_delta_robot_xml(base_radius=3.0, platform_radius=1.0)
    assert isinstance(xml, str)
    assert '<mujoco model="delta_robot">' in xml
    assert "motor1" in xml
    assert "motor2" in xml
    assert "motor3" in xml
    assert 'size="3.0 0.2"' in xml
    assert 'size="1.0 0.1"' in xml


def test_generate_five_bar_parallel_xml():
    """Test generating 5-bar parallel mechanism XML."""
    xml = generate_five_bar_parallel_xml(link_length=2.5)
    assert isinstance(xml, str)
    assert '<mujoco model="five_bar_parallel">' in xml
    assert "left_motor" in xml
    assert "right_motor" in xml
    # Checks link length
    assert "2.5 0 0" in xml


def test_generate_stewart_platform_xml():
    """Test generating Stewart platform XML."""
    xml = generate_stewart_platform_xml(base_radius=2.0, platform_radius=1.0)
    assert isinstance(xml, str)
    assert '<mujoco model="stewart_platform">' in xml
    for i in range(1, 7):
        assert f"leg{i}_motor" in xml
        assert f"leg{i}_lower" in xml
    assert 'size="2.0 0.15"' in xml
    assert 'size="1.0 0.12"' in xml
