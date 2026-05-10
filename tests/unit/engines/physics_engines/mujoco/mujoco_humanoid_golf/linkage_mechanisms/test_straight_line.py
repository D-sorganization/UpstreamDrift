"""Unit tests for straight line mechanism generators."""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms.straight_line import (
    generate_chebyshev_linkage_xml,
    generate_peaucellier_linkage_xml,
    generate_watt_linkage_xml,
)


def test_generate_peaucellier_linkage_xml():
    """Test generating Peaucellier-Lipkin linkage XML."""
    xml = generate_peaucellier_linkage_xml(scale=2.0)
    assert isinstance(xml, str)
    assert '<mujoco model="peaucellier_linkage">' in xml
    assert "drive_motor" in xml
    assert 'pos="5.0 0 1"' in xml


def test_generate_chebyshev_linkage_xml():
    """Test generating Chebyshev linkage XML."""
    xml = generate_chebyshev_linkage_xml()
    assert isinstance(xml, str)
    assert '<mujoco model="chebyshev_linkage">' in xml
    assert "left_crank_motor" in xml


def test_generate_watt_linkage_xml():
    """Test generating Watt linkage XML."""
    xml = generate_watt_linkage_xml()
    assert isinstance(xml, str)
    assert '<mujoco model="watt_linkage">' in xml
    assert "left_motor" in xml
