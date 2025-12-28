from typing import Any

import numpy as np


class PoseEstimator:
    def __init__(self) -> None:
        # OpenPose typically returns 25 keypoints for BODY_25 model
        self.num_keypoints = 25

    def process_frame(
        self, frame: np.ndarray[Any, Any]
    ) -> dict[int, tuple[float, float, float]]:
        """
        Mock processing. Returns a dict of {id: (x, y, confidence)}.
        In a real scenario, this would call the OpenPose wrapper.
        """
        # Generate some dummy keypoints that move slightly to simulate motion
        # We'll just generate random points for now to demonstrate the pipeline
        height, width, _ = frame.shape
        # Using a deterministic random seed based on frame sum
        # to make it stable per frame content
        rng = np.random.default_rng(seed=int(frame.sum()) % 123456789)

        keypoints = {}
        for i in range(self.num_keypoints):
            x = rng.uniform(0, width)
            y = rng.uniform(0, height)
            conf = rng.uniform(0.5, 1.0)
            keypoints[i] = (x, y, conf)
        return keypoints
