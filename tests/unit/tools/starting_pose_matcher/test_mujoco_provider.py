"""Tests for the MuJoCo skeleton provider."""

import pytest


# Test that the provider can be imported even without MuJoCo installed
def test_import_without_mujoco():
    """Test that importing the module doesn't break without MuJoCo."""
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco

    # These should be importable without MuJoCo
    assert hasattr(mujoco, "MuJoCoNotAvailableError")
    assert hasattr(mujoco, "MuJoCoProviderError")
    assert hasattr(mujoco, "MuJoCoSkeletonProvider")
    assert hasattr(mujoco, "create_provider")
    assert hasattr(mujoco, "MUJOCO_TO_MATCHER_VOCAB")
    assert hasattr(mujoco, "MATCHER_TO_MUJOCO")


def test_mujoco_provider_vocabulary_mapping():
    """Test that the vocabulary mapping is correct."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mujoco import (
        MATCHER_TO_MUJOCO,
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
        assert name in MATCHER_TO_MUJOCO, f"Missing vocabulary mapping for {name}"

    # Check reverse mapping is consistent
    for matcher_name in MATCHER_TO_MUJOCO.keys():
        assert matcher_name in required_names, f"Unexpected vocabulary: {matcher_name}"


def test_mujoco_not_available_error():
    """Test that MuJoCoNotAvailableError is raised when MuJoCo is not installed."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mujoco import (
        MuJoCoNotAvailableError,
        MuJoCoProviderError,
        MuJoCoSkeletonProvider,
    )

    # Try to create provider without a valid model path
    # This should raise an error (either MuJoCoNotAvailableError or MuJoCoProviderError)
    with pytest.raises(
        (MuJoCoNotAvailableError, MuJoCoProviderError, MuJoCoProviderError)
    ):
        MuJoCoSkeletonProvider(model_path=None, model_xml=None)


def test_mujoco_provider_create_provider_function():
    """Test that create_provider function exists and has correct signature."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mujoco import (
        create_provider,
    )

    # Check function signature
    import inspect

    sig = inspect.signature(create_provider)
    params = list(sig.parameters.keys())
    assert "model_path" in params
    assert "model_xml" in params


# Test with a minimal MJCF model
MINIMAL_MJCF = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="minimal_golfer">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hip" pos="0 0 1">
      <joint name="hip_joint" type="free"/>
      <body name="spine" pos="0 0 0.1">
        <body name="torso" pos="0 0 0.1">
          <body name="hub" pos="0 0 0.1">
            <body name="left_shoulder" pos="0.1 0 0">
              <body name="left_elbow" pos="0.1 0 0">
                <body name="left_wrist" pos="0.1 0 0"/>
              </body>
            </body>
            <body name="right_shoulder" pos="-0.1 0 0">
              <body name="right_elbow" pos="-0.1 0 0">
                <body name="right_wrist" pos="-0.1 0 0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="midpoint" pos="0 0 0.5"/>
    <body name="clubhead" pos="0 0 0.6"/>
  </worldbody>
</mujoco>
"""
