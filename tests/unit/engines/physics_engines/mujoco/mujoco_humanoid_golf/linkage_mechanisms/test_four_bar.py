"""Unit tests for four_bar.py linkage mechanism generators."""

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.linkage_mechanisms.four_bar import (
    generate_four_bar_linkage_xml,
)


def test_generate_four_bar_linkage_xml_defaults():
    """Test generating four-bar linkage XML with default arguments."""
    xml = generate_four_bar_linkage_xml()
    assert isinstance(xml, str)
    assert '<mujoco model="four_bar_linkage_grashof_crank_rocker">' in xml
    assert "<worldbody>" in xml
    assert "crank_motor" in xml


@pytest.mark.parametrize(
    "link_type, expected_model_name",
    [
        ("grashof_crank_rocker", "four_bar_linkage_grashof_crank_rocker"),
        ("grashof_double_crank", "four_bar_linkage_grashof_double_crank"),
        ("grashof_double_rocker", "four_bar_linkage_grashof_double_rocker"),
        ("non_grashof", "four_bar_linkage_non_grashof"),
        ("parallel", "four_bar_linkage_parallel"),
        ("antiparallel", "four_bar_linkage_antiparallel"),
    ],
)
def test_generate_four_bar_linkage_xml_types(link_type, expected_model_name):
    """Test generating four-bar linkage XML for different types."""
    xml = generate_four_bar_linkage_xml(link_type=link_type)
    assert isinstance(xml, str)
    assert f'<mujoco model="{expected_model_name}">' in xml


def test_generate_four_bar_linkage_xml_custom_lengths():
    """Test generating four-bar linkage XML with custom link lengths."""
    lengths = [5.0, 1.5, 4.0, 3.5]
    xml = generate_four_bar_linkage_xml(link_lengths=lengths)
    assert isinstance(xml, str)
    # The follower pivot is at x=ground, so we check for '5.0 0 0.5'
    assert 'pos="5.0 0 0.5"' in xml
