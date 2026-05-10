"""Unit tests for slider mechanism generators."""

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms.slider_mechanisms import (
    generate_scotch_yoke_xml,
    generate_slider_crank_xml,
)


def test_generate_slider_crank_xml_horizontal():
    """Test generating horizontal slider crank XML."""
    xml = generate_slider_crank_xml(orientation="horizontal")
    assert isinstance(xml, str)
    assert '<mujoco model="slider_crank">' in xml
    assert 'axis="1 0 0"' in xml


def test_generate_slider_crank_xml_vertical():
    """Test generating vertical slider crank XML."""
    xml = generate_slider_crank_xml(orientation="vertical")
    assert isinstance(xml, str)
    assert '<mujoco model="slider_crank">' in xml
    assert 'axis="0 0 1"' in xml


def test_generate_slider_crank_xml_custom_lengths():
    """Test generating slider crank XML with custom lengths."""
    xml = generate_slider_crank_xml(crank_length=2.0, rod_length=4.0)
    assert isinstance(xml, str)
    # The start/end should be dynamically calculated
    assert 'range="-6.0 6.0"' in xml


def test_generate_scotch_yoke_xml():
    """Test generating scotch yoke XML."""
    xml = generate_scotch_yoke_xml(crank_radius=2.0)
    assert isinstance(xml, str)
    assert '<mujoco model="scotch_yoke">' in xml
    assert 'range="-2.5 2.5"' in xml
    assert "crank_motor" in xml
