"""Tests for the Pinocchio skeleton provider."""

import pytest


# Test that the provider can be imported even without Pinocchio installed
def test_import_without_pinocchio():
    """Test that importing the module doesn't break without Pinocchio."""
    from src.tools.starting_pose_matcher.providers import pinocchio

    # These should be importable without Pinocchio
    assert hasattr(pinocchio, "PinocchioNotAvailableError")
    assert hasattr(pinocchio, "PinocchioProviderError")
    assert hasattr(pinocchio, "PinocchioSkeletonProvider")
    assert hasattr(pinocchio, "create_provider")
    assert hasattr(pinocchio, "PINOCCHIO_TO_MATCHER_VOCAB")
    assert hasattr(pinocchio, "MATCHER_TO_PINOCCHIO")


def test_vocabulary_mapping():
    """Test that the vocabulary mapping is correct."""
    from src.tools.starting_pose_matcher.providers.pinocchio import (
        PINOCCHIO_TO_MATCHER_VOCAB,
        MATCHER_TO_PINOCCHIO,
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
        assert name in MATCHER_TO_PINOCCHIO, f"Missing vocabulary mapping for {name}"

    # Check reverse mapping is consistent
    for matcher_name in MATCHER_TO_PINOCCHIO.keys():
        assert matcher_name in required_names, f"Unexpected vocabulary: {matcher_name}"


def test_pinocchio_not_available_error():
    """Test that PinocchioNotAvailableError is raised when Pinocchio is not installed."""
    from src.tools.starting_pose_matcher.providers.pinocchio import (
        PinocchioNotAvailableError,
        PinocchioProviderError,
        PinocchioSkeletonProvider,
    )

    # Try to create provider without a urdf_path
    # This should raise an error (either PinocchioNotAvailableError or PinocchioProviderError)
    with pytest.raises(
        (PinocchioNotAvailableError, PinocchioProviderError, PinocchioProviderError)
    ):
        PinocchioSkeletonProvider(urdf_path=None)


def test_create_provider_function():
    """Test that create_provider function exists and has correct signature."""
    from src.tools.starting_pose_matcher.providers.pinocchio import create_provider

    # Check function signature
    import inspect

    sig = inspect.signature(create_provider)
    params = list(sig.parameters.keys())
    assert "urdf_path" in params
    assert "package_paths" in params


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


def test_provider_with_minimal_model():
    """Test provider with a minimal URDF model."""
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pytest.skip("Pinocchio not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.pinocchio import (
        PinocchioSkeletonProvider,
        PinocchioProviderError,
    )

    # Write URDF to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(MINIMAL_URDF)
        urdf_path = f.name

    try:
        provider = PinocchioSkeletonProvider(urdf_path=urdf_path)
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

        os.unlink(urdf_path)


def test_provider_with_q_config():
    """Test provider with configuration vector input."""
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pytest.skip("Pinocchio not installed")

    import tempfile
    import numpy as np
    from src.tools.starting_pose_matcher.providers.pinocchio import (
        PinocchioSkeletonProvider,
    )

    # Write URDF to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(MINIMAL_URDF)
        urdf_path = f.name

    try:
        provider = PinocchioSkeletonProvider(urdf_path=urdf_path)

        # Get default skeleton
        skeleton1 = provider.get_skeleton()

        # Create a configuration vector
        q = np.zeros(provider.model.nq)
        q[0] = 0.1  # Small translation in x

        skeleton2 = provider.get_skeleton(q=q)

        # Positions should be different after applying new configuration
        assert not np.allclose(skeleton1["hip"], skeleton2["hip"])
    finally:
        import os

        os.unlink(urdf_path)


def test_get_available_frames():
    """Test get_available_frames method."""
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pytest.skip("Pinocchio not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.pinocchio import (
        PinocchioSkeletonProvider,
    )

    # Write URDF to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(MINIMAL_URDF)
        urdf_path = f.name

    try:
        provider = PinocchioSkeletonProvider(urdf_path=urdf_path)
        frames = provider.get_available_frames()

        # Check that we have the expected frames
        expected_frames = [
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
        for frame in expected_frames:
            assert frame in frames, f"Missing frame: {frame}"
    finally:
        import os

        os.unlink(urdf_path)


def test_get_available_joints():
    """Test get_available_joints method."""
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pytest.skip("Pinocchio not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.pinocchio import (
        PinocchioSkeletonProvider,
    )

    # Write URDF to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(MINIMAL_URDF)
        urdf_path = f.name

    try:
        provider = PinocchioSkeletonProvider(urdf_path=urdf_path)
        joints = provider.get_available_joints()

        # Should have at least the root joint and all fixed joints
        assert len(joints) > 0, "Should have at least one joint"
    finally:
        import os

        os.unlink(urdf_path)


def test_missing_vocabulary_error():
    """Test that PinocchioProviderError is raised when vocabulary is missing."""
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pytest.skip("Pinocchio not installed")

    import tempfile
    from src.tools.starting_pose_matcher.providers.pinocchio import (
        PinocchioSkeletonProvider,
        PinocchioProviderError,
    )

    # Create a minimal model that's missing required bodies
    incomplete_urdf = """<?xml version="1.0" encoding="utf-8"?>
<robot name="incomplete">
  <link name="hip">
    <inertial>
      <mass value="1.0"/>
    </inertial>
  </link>
</robot>
"""

    # Write URDF to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(incomplete_urdf)
        urdf_path = f.name

    try:
        with pytest.raises(PinocchioProviderError, match="Missing required"):
            PinocchioSkeletonProvider(urdf_path=urdf_path)
    finally:
        import os

        os.unlink(urdf_path)
