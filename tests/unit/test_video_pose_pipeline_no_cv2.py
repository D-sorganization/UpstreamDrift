import sys
from unittest.mock import MagicMock, patch

import pytest


def test_missing_cv2():
    # Ensure cv2 is not in sys.modules
    if "cv2" in sys.modules:
        del sys.modules["cv2"]

    # Also ensure shared.python.video_pose_pipeline is not in sys.modules
    if "shared.python.video_pose_pipeline" in sys.modules:
        del sys.modules["shared.python.video_pose_pipeline"]

    # Mock sys.modules to simulate cv2 missing
    with patch.dict(sys.modules):
        if "cv2" in sys.modules:
            del sys.modules["cv2"]

        # Import the module
        from shared.python import video_pose_pipeline

        # Check if cv2 is None
        assert video_pose_pipeline.cv2 is None

        # Instantiate pipeline mocking _load_estimator
        with patch.object(video_pose_pipeline.VideoPosePipeline, "_load_estimator"):
            pipeline = video_pose_pipeline.VideoPosePipeline()
            # Set a dummy estimator to bypass "Estimator not loaded" check
            pipeline.estimator = MagicMock()

            # Now test that methods requiring cv2 raise RuntimeError

            # process_video
            with pytest.raises(RuntimeError, match=r"OpenCV \(cv2\) is not installed"):
                pipeline.process_video(MagicMock())

            # _process_frames_individually
            with pytest.raises(RuntimeError, match=r"OpenCV \(cv2\) is not installed"):
                pipeline._process_frames_individually(MagicMock(), 10)
