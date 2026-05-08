"""Tests for the OpenSim skeleton provider."""

import pytest


# Test that the provider can be imported even without OpenSim installed
def test_import_without_opensim():
    """Test that importing the module doesn't break without OpenSim."""
    from src.tools.starting_pose_matcher.providers import opensim

    # These should be importable without OpenSim
    assert hasattr(opensim, "OpenSimNotAvailableError")
    assert hasattr(opensim, "OpenSimProviderError")
    assert hasattr(opensim, "OpenSimSkeletonProvider")
    assert hasattr(opensim, "create_provider")
    assert hasattr(opensim, "OPENSIM_TO_MATCHER_VOCAB")
    assert hasattr(opensim, "MATCHER_TO_OPENSIM")


def test_vocabulary_mapping():
    """Test that the vocabulary mapping is correct."""
    from src.tools.starting_pose_matcher.providers.opensim import (
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
    from src.tools.starting_pose_matcher.providers.opensim import (
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


def test_create_provider_function():
    """Test that create_provider function exists and has correct signature."""
    from src.tools.starting_pose_matcher.providers.opensim import create_provider

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


def test_provider_with_minimal_model():
    """Test provider with a minimal OpenSim model."""
    try:
        import opensim as osim  # noqa: F401
    except ImportError:
        pytest.skip("OpenSim not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.opensim import (
        OpenSimSkeletonProvider,
        OpenSimProviderError,
    )

    # Write OSIM to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".osim", delete=False) as f:
        f.write(MINIMAL_OSIM)
        model_path = f.name

    try:
        provider = OpenSimSkeletonProvider(model_path=model_path)
        skeleton = provider.get_skeleton()

        # Check all required vocabulary is present
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
        for name in required_names:
            assert name in skeleton, f"Missing skeleton joint: {name}"

        # Check positions are numpy arrays with correct shape
        import numpy as np

        for name, pos in skeleton.keys():
            assert isinstance(pos, np.ndarray), (
                f"Position for {name} should be numpy array"
            )
            assert pos.shape == (3,), f"Position for {name} should have shape (3,)"
    finally:
        import os

        os.unlink(model_path)


def test_provider_with_coordinates():
    """Test provider with coordinate values input."""
    try:
        import opensim as osim  # noqa: F401
    except ImportError:
        pytest.skip("OpenSim not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.opensim import (
        OpenSimSkeletonProvider,
    )

    # Write OSIM to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".osim", delete=False) as f:
        f.write(MINIMAL_OSIM)
        model_path = f.name

    try:
        provider = OpenSimSkeletonProvider(model_path=model_path)

        # Get default skeleton
        skeleton1 = provider.get_skeleton()

        # Apply some coordinate values
        coordinates = {"hip_spine_coord": 0.1}
        skeleton2 = provider.get_skeleton(coordinates=coordinates)

        # Note: With fixed joints, positions may not change significantly
        # This test mainly verifies the coordinate application mechanism works
        assert isinstance(skeleton1, dict)
        assert isinstance(skeleton2, dict)
    finally:
        import os

        os.unlink(model_path)


def test_get_available_bodies():
    """Test get_available_bodies method."""
    try:
        import opensim as osim  # noqa: F401
    except ImportError:
        pytest.skip("OpenSim not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.opensim import (
        OpenSimSkeletonProvider,
    )

    # Write OSIM to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".osim", delete=False) as f:
        f.write(MINIMAL_OSIM)
        model_path = f.name

    try:
        provider = OpenSimSkeletonProvider(model_path=model_path)
        bodies = provider.get_available_bodies()

        # Check that we have the expected bodies
        expected_bodies = [
            "hip",
            "spine",
            "torso",
            "hub",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "midpoint",
            "clubhead",
        ]
        for body in expected_bodies:
            assert body in bodies, f"Missing body: {body}"
    finally:
        import os

        os.unlink(model_path)


def test_get_available_markers():
    """Test get_available_markers method."""
    try:
        import opensim as osim  # noqa: F401
    except ImportError:
        pytest.skip("OpenSim not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.opensim import (
        OpenSimSkeletonProvider,
    )

    # Write OSIM to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".osim", delete=False) as f:
        f.write(MINIMAL_OSIM)
        model_path = f.name

    try:
        provider = OpenSimSkeletonProvider(model_path=model_path)
        markers = provider.get_available_markers()

        # Check that we have the expected markers
        expected_markers = [
            "hip",
            "spine",
            "torso",
            "hub",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "midpoint",
            "clubhead",
        ]
        for marker in expected_markers:
            assert marker in markers, f"Missing marker: {marker}"
    finally:
        import os

        os.unlink(model_path)


def test_missing_vocabulary_error():
    """Test that OpenSimProviderError is raised when vocabulary is missing."""
    try:
        import opensim as osim  # noqa: F401
    except ImportError:
        pytest.skip("OpenSim not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.opensim import (
        OpenSimSkeletonProvider,
        OpenSimProviderError,
    )

    # Create a minimal model that's missing required bodies
    incomplete_osim = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
  <Model name="incomplete">
    <Body name="hip" mass="1.0">
      <mass>1.0</mass>
      <mass_center>0 0 0</mass_center>
      <inertia>0.1 0.1 0.1 0 0 0</inertia>
    </Body>
  </Model>
</OpenSimDocument>
"""

    # Write OSIM to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".osim", delete=False) as f:
        f.write(incomplete_osim)
        model_path = f.name

    try:
        with pytest.raises(OpenSimProviderError, match="Missing required"):
            OpenSimSkeletonProvider(model_path=model_path)
    finally:
        import os

        os.unlink(model_path)
