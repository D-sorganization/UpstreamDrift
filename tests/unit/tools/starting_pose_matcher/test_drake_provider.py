"""Tests for the Drake skeleton provider."""

import pytest


# Test that the provider can be imported even without Drake installed
def test_import_without_drake():
    """Test that importing the module doesn't break without Drake."""
    from src.tools.starting_pose_matcher.providers import drake

    # These should be importable without Drake
    assert hasattr(drake, "DrakeNotAvailableError")
    assert hasattr(drake, "DrakeProviderError")
    assert hasattr(drake, "DrakeSkeletonProvider")
    assert hasattr(drake, "create_provider")
    assert hasattr(drake, "DRAKE_TO_MATCHER_VOCAB")
    assert hasattr(drake, "MATCHER_TO_DRAKE")


def test_drake_provider_vocabulary_mapping():
    """Test that the vocabulary mapping is correct."""
    from src.tools.starting_pose_matcher.providers.drake import (
        DRAKE_TO_MATCHER_VOCAB,
        MATCHER_TO_DRAKE,
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
        assert name in MATCHER_TO_DRAKE, f"Missing vocabulary mapping for {name}"

    # Check reverse mapping is consistent
    for matcher_name in MATCHER_TO_DRAKE.keys():
        assert matcher_name in required_names, f"Unexpected vocabulary: {matcher_name}"


def test_drake_not_available_error():
    """Test that DrakeNotAvailableError is raised when Drake is not installed."""
    from src.tools.starting_pose_matcher.providers.drake import (
        DrakeNotAvailableError,
        DrakeProviderError,
        DrakeSkeletonProvider,
    )

    # Try to create provider without a valid model path
    # This should raise an error (either DrakeNotAvailableError or DrakeProviderError)
    with pytest.raises(
        (DrakeNotAvailableError, DrakeProviderError, DrakeProviderError)
    ):
        DrakeSkeletonProvider(model_path=None, model_xml=None)


def test_drake_provider_create_provider_function():
    """Test that create_provider function exists and has correct signature."""
    from src.tools.starting_pose_matcher.providers.drake import create_provider

    # Check function signature
    import inspect

    sig = inspect.signature(create_provider)
    params = list(sig.parameters.keys())
    assert "model_path" in params
    assert "model_xml" in params


# Test with a minimal URDF model
MINIMAL_URDF = """<?xml version="1.0" encoding="utf-8"?>
<robot name="minimal_golfer">
  <link name="hip">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="spine">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="torso">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="hub">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="left_shoulder">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="right_shoulder">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="left_elbow">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="right_elbow">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="left_wrist">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="right_wrist">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="midpoint">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="clubhead">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="hip_spine" type="fixed">
    <parent link="hip"/>
    <child link="spine"/>
    <origin xyz="0 0 0.1"/>
  </joint>
  <joint name="spine_torso" type="fixed">
    <parent link="spine"/>
    <child link="torso"/>
    <origin xyz="0 0 0.1"/>
  </joint>
  <joint name="torso_hub" type="fixed">
    <parent link="torso"/>
    <child link="hub"/>
    <origin xyz="0 0 0.1"/>
  </joint>
  <joint name="hub_lshoulder" type="fixed">
    <parent link="hub"/>
    <child link="left_shoulder"/>
    <origin xyz="0.1 0 0"/>
  </joint>
  <joint name="hub_rshoulder" type="fixed">
    <parent link="hub"/>
    <child link="right_shoulder"/>
    <origin xyz="-0.1 0 0"/>
  </joint>
  <joint name="lshoulder_lelbow" type="fixed">
    <parent link="left_shoulder"/>
    <child link="left_elbow"/>
    <origin xyz="0.1 0 0"/>
  </joint>
  <joint name="rshoulder_relbow" type="fixed">
    <parent link="right_shoulder"/>
    <child link="right_elbow"/>
    <origin xyz="-0.1 0 0"/>
  </joint>
  <joint name="lelbow_lwrist" type="fixed">
    <parent link="left_elbow"/>
    <child link="left_wrist"/>
    <origin xyz="0.1 0 0"/>
  </joint>
  <joint name="relbow_rwrist" type="fixed">
    <parent link="right_elbow"/>
    <child link="right_wrist"/>
    <origin xyz="-0.1 0 0"/>
  </joint>
  <joint name="hub_midpoint" type="fixed">
    <parent link="hub"/>
    <child link="midpoint"/>
    <origin xyz="0 0 0.5"/>
  </joint>
  <joint name="hub_clubhead" type="fixed">
    <parent link="hub"/>
    <child link="clubhead"/>
    <origin xyz="0 0 0.6"/>
  </joint>
</robot>
"""
