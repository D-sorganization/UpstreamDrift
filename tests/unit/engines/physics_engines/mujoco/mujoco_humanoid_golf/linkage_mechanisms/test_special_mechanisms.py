"""Unit tests for special mechanism generators."""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms.special_mechanisms import (
    generate_geneva_mechanism_xml,
    generate_oldham_coupling_xml,
)


def test_generate_geneva_mechanism_xml():
    """Test generating Geneva mechanism XML."""
    xml = generate_geneva_mechanism_xml(num_slots=6, drive_radius=3.0)
    assert isinstance(xml, str)
    assert '<mujoco model="geneva_mechanism">' in xml
    assert "drive_motor" in xml
    assert 'size="3.0 0.15"' in xml


def test_generate_oldham_coupling_xml():
    """Test generating Oldham coupling XML."""
    xml = generate_oldham_coupling_xml(offset=1.5)
    assert isinstance(xml, str)
    assert '<mujoco model="oldham_coupling">' in xml
    assert "input_motor" in xml
    assert 'pos="1.5 0 1.9"' in xml
