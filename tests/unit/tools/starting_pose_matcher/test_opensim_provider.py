"""Tests for the OpenSim skeleton provider."""

import pytest


# Test that the provider can be imported even without OpenSim installed
def test_import_without_opensim():
    """Test that importing the module doesn't break without OpenSim."""
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim

    # These should be importable without OpenSim
    assert hasattr(opensim, "OpenSimNotAvailableError")
    assert hasattr(opensim, "OpenSimProviderError")
    assert hasattr(opensim, "OpenSimSkeletonProvider")
    assert hasattr(opensim, "create_provider")
    assert hasattr(opensim, "OPENSIM_TO_MATCHER_VOCAB")
    assert hasattr(opensim, "MATCHER_TO_OPENSIM")


def test_opensim_provider_vocabulary_mapping():
    """Test that the vocabulary mapping is correct."""
    from src.tools.starting_pose_matcher.skeleton_extractors.opensim import (
        OPENSIM_TO_MATCHER_VOCAB,
        MATCHER_TO_OPENSIM,
    )

    # Required vocabulary
    required_names = [
        "hip",
        "spine",
        "torso",
        "hub",
        "ls",
        "rs",
        "le",
        "re",
        "lw",
        "rw",
        "mp",
        "ch",
    ]

    # Check all required names are in the reverse mapping
    for name in required_names:
        assert name in MATCHER_TO_OPENSIM, f"Missing vocabulary mapping for {name}"

    # Check reverse mapping is consistent
    for matcher_name in MATCHER_TO_OPENSIM.keys():
        assert matcher_name in required_names, f"Unexpected vocabulary: {matcher_name}"


def test_opensim_not_available_error():
    """Test that OpenSimNotAvailableError is raised when OpenSim is not installed."""
    from src.tools.starting_pose_matcher.skeleton_extractors.opensim import (
        OpenSimNotAvailableError,
        OpenSimProviderError,
        OpenSimSkeletonProvider,
    )

    # Try to create provider without a valid model path
    # This should raise an error (either OpenSimNotAvailableError or OpenSimProviderError)
    with pytest.raises(
        (OpenSimNotAvailableError, OpenSimProviderError, OpenSimProviderError)
    ):
        OpenSimSkeletonProvider(model_path=None, model_xml=None)


def test_opensim_provider_create_provider_function():
    """Test that create_provider function exists and has correct signature."""
    from src.tools.starting_pose_matcher.skeleton_extractors.opensim import (
        create_provider,
    )

    # Check function signature
    import inspect

    sig = inspect.signature(create_provider)
    params = list(sig.parameters.keys())
    assert "model_path" in params
    assert "model_xml" in params


# Minimal OpenSim model XML for testing
MINIMAL_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
  <Model name="minimal_golfer">
    <Body name="hip" mass="1.0">
      <mass>1.0</mass>
      <mass_center>0 0 0</mass_center>
      <inertia>0.1 0.1 0.1 0 0 0</inertia>
    </Body>
    <Body name="spine" mass="1.0">
      <mass>1.0</mass>
      <mass_center>0 0 0.1</mass_center>
      <inertia>0.1 0.1 0.1 0 0 0</inertia>
    </Body>
    <Body name="torso" mass="1.0">
      <mass>1.0</mass>
      <mass_center>0 0 0.1</mass_center>
      <inertia>0.1 0.1 0.1 0 0 0</inertia>
    </Body>
    <Body name="hub" mass="1.0">
      <mass>1.0</mass>
      <mass_center>0 0 0.1</mass_center>
      <inertia>0.1 0.1 0.1 0 0 0</inertia>
    </Body>
    <Body name="left_shoulder" mass="0.1">
      <mass>0.1</mass>
      <mass_center>0.1 0 0</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
    </Body>
    <Body name="right_shoulder" mass="0.1">
      <mass>0.1</mass>
      <mass_center>-0.1 0 0</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
    </Body>
    <Body name="left_elbow" mass="0.1">
      <mass>0.1</mass>
      <mass_center>0.1 0 0</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
    </Body>
    <Body name="right_elbow" mass="0.1">
      <mass>0.1</mass>
      <mass_center>-0.1 0 0</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
    </Body>
    <Body name="left_wrist" mass="0.1">
      <mass>0.1</mass>
      <mass_center>0.1 0 0</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
    </Body>
    <Body name="right_wrist" mass="0.1">
      <mass>0.1</mass>
      <mass_center>-0.1 0 0</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
    </Body>
    <Body name="midpoint" mass="0.1">
      <mass>0.1</mass>
      <mass_center>0 0 0.5</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
    </Body>
    <Body name="clubhead" mass="0.1">
      <mass>0.1</mass>
      <mass_center>0 0 0.6</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
    </Body>
    <Joint name="hip_spine" type="Pin">
      <parent_frame>hip</parent_frame>
      <child_frame>spine</child_frame>
    </Joint>
    <Joint name="spine_torso" type="Pin">
      <parent_frame>spine</parent_frame>
      <child_frame>torso</child_frame>
    </Joint>
    <Joint name="torso_hub" type="Pin">
      <parent_frame>torso</parent_frame>
      <child_frame>hub</child_frame>
    </Joint>
    <Joint name="hub_lshoulder" type="Pin">
      <parent_frame>hub</parent_frame>
      <child_frame>left_shoulder</child_frame>
    </Joint>
    <Joint name="hub_rshoulder" type="Pin">
      <parent_frame>hub</parent_frame>
      <child_frame>right_shoulder</child_frame>
    </Joint>
    <Joint name="lshoulder_lelbow" type="Pin">
      <parent_frame>left_shoulder</parent_frame>
      <child_frame>left_elbow</child_frame>
    </Joint>
    <Joint name="rshoulder_relbow" type="Pin">
      <parent_frame>right_shoulder</parent_frame>
      <child_frame>right_elbow</child_frame>
    </Joint>
    <Joint name="lelbow_lwrist" type="Pin">
      <parent_frame>left_elbow</parent_frame>
      <child_frame>left_wrist</child_frame>
    </Joint>
    <Joint name="relbow_rwrist" type="Pin">
      <parent_frame>right_elbow</parent_frame>
      <child_frame>right_wrist</child_frame>
    </Joint>
    <Joint name="hub_midpoint" type="Pin">
      <parent_frame>hub</parent_frame>
      <child_frame>midpoint</child_frame>
    </Joint>
    <Joint name="hub_clubhead" type="Pin">
      <parent_frame>hub</parent_frame>
      <child_frame>clubhead</child_frame>
    </Joint>
    <Marker name="hip" body="hip" location="0 0 0"/>
    <Marker name="spine" body="spine" location="0 0 0"/>
    <Marker name="torso" body="torso" location="0 0 0"/>
    <Marker name="hub" body="hub" location="0 0 0"/>
    <Marker name="left_shoulder" body="left_shoulder" location="0 0 0"/>
    <Marker name="right_shoulder" body="right_shoulder" location="0 0 0"/>
    <Marker name="left_elbow" body="left_elbow" location="0 0 0"/>
    <Marker name="right_elbow" body="right_elbow" location="0 0 0"/>
    <Marker name="left_wrist" body="left_wrist" location="0 0 0"/>
    <Marker name="right_wrist" body="right_wrist" location="0 0 0"/>
    <Marker name="midpoint" body="midpoint" location="0 0 0"/>
    <Marker name="clubhead" body="clubhead" location="0 0 0"/>
  </Model>
</OpenSimDocument>
"""
