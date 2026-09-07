"""OpenCV-DNN OpenPose estimator: peaks, contract, model resolution."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.core.error_utils import ModelError
from src.shared.python.pose_estimation import openpose_models
from src.shared.python.pose_estimation.openpose_dnn_estimator import (
    OpenPoseDnnEstimator,
)

pytestmark = pytest.mark.unit


class _FakeNet:
    """Returns heat maps with one hot pixel per joint at a known place."""

    def __init__(self, peak: float = 0.9) -> None:
        self.peak = peak
        self.inputs: list[np.ndarray] = []

    def setInput(self, blob: np.ndarray) -> None:  # noqa: N802 - cv2 API
        self.inputs.append(blob)

    def forward(self) -> np.ndarray:
        maps = np.zeros((1, 78, 46, 82), dtype=np.float32)
        for idx in range(25):
            maps[0, idx, 10 + idx, 20 + idx] = self.peak
        maps[0, 7, :, :] = 0.0  # left wrist: no detection
        return maps


def _files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Right-sized placeholder files, with the digest pins lifted for the test."""
    unpinned = {
        k: dataclasses.replace(v, sha256=None)
        for k, v in openpose_models.BODY25_FILES.items()
    }
    monkeypatch.setattr(openpose_models, "BODY25_FILES", unpinned)
    for spec in unpinned.values():
        (tmp_path / spec.filename).write_bytes(b"\0" * spec.size_bytes)
    return tmp_path


def test_peaks_map_heatmap_maxima_to_normalized_coordinates() -> None:
    est = OpenPoseDnnEstimator(net_factory=lambda p, w: _FakeNet())
    heat = _FakeNet().forward()[0]
    keypoints, confidences = est.peaks(heat)
    assert keypoints["nose"] == pytest.approx([(20 + 0.5) / 82, (10 + 0.5) / 46])
    assert confidences["nose"] == pytest.approx(0.9)
    assert "left_wrist" not in keypoints and confidences["left_wrist"] == 0.0
    assert len(confidences) == 25


def test_estimate_requires_load_and_returns_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _files(tmp_path, monkeypatch)
    net = _FakeNet()
    est = OpenPoseDnnEstimator(net_factory=lambda p, w: net)
    with pytest.raises(Exception, match="load_model"):
        est.estimate_from_image(np.zeros((720, 1280, 3), dtype=np.uint8))
    est.load_model(cache)
    assert est.is_ready() and est.model_path is not None
    result = est.estimate_from_image(np.zeros((720, 1280, 3), dtype=np.uint8), 40)
    assert result.timestamp == pytest.approx(0.04)
    assert result.raw_keypoints is not None and "right_wrist" in result.raw_keypoints
    assert result.raw_confidences is not None and len(result.raw_confidences) == 25
    assert 0.0 < result.confidence <= 1.0 and result.joint_angles == {}
    blob = net.inputs[0]
    assert blob.shape[2] == 368 and blob.shape[3] % 8 == 0
    est.close()
    assert not est.is_ready()


def test_missing_model_files_name_the_download_command(tmp_path: Path) -> None:
    est = OpenPoseDnnEstimator(net_factory=lambda p, w: _FakeNet())
    with pytest.raises(ModelError, match="openpose_models"):
        est.load_model(tmp_path)


def test_constructor_contracts() -> None:
    with pytest.raises(Exception, match="input_height"):
        OpenPoseDnnEstimator(input_height=8)
    with pytest.raises(Exception, match="min_peak"):
        OpenPoseDnnEstimator(min_peak=1.0)


def test_registry_lists_openpose_dnn() -> None:
    from src.shared.python.pose_estimation.registry import get_estimator_info

    info = get_estimator_info("openpose_dnn")
    assert info.probe_module == "cv2"
    names = [j["name"] for j in info.skeleton]
    assert names[:3] == ["nose", "neck", "right_shoulder"] and len(names) == 25
