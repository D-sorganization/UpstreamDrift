"""Unit tests for the historical capture runner CLI and matrix estimators (#11273)."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CanonicalObservations,
)
from src.shared.python.motion_pipeline.sources.hmr2_adapter import HMR2Adapter
from src.shared.python.pose_estimation.interface import (
    PoseEstimationResult,
    PoseEstimator,
)
from src.shared.python.pose_estimation.registry import (
    EstimatorInfo,
    register_estimator,
    unregister_estimator,
)


pytestmark = pytest.mark.unit


def _create_synthetic_mp4(path: Path, n_frames: int = 10, fps: int = 10) -> Path:
    """Create a minimal synthetic MP4 file for testing."""
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(n_frames):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="bgr24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


class FakeEstimator(PoseEstimator):
    """Stub estimator for testing runner behavior."""

    def __init__(
        self,
        name: str = "fake",
        model_sha256: str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        version: str = "1.2.3",
    ) -> None:
        self.name = name
        self.model_sha256 = model_sha256
        self.version = version
        self.closed = False

    def estimate_from_image(
        self, image: np.ndarray, timestamp_ms: int = 0
    ) -> PoseEstimationResult:
        return PoseEstimationResult(
            joint_angles={},
            raw_keypoints={"nose": np.array([0.5, 0.5, 0.0])},
            confidence=0.9,
            timestamp=float(timestamp_ms),
            raw_confidences={"nose": 0.9},
        )

    def load_model(self, model_path: Path | None = None) -> None:
        pass

    def estimate_from_video(self, video_path: Path) -> list[PoseEstimationResult]:
        return []

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    return _create_synthetic_mp4(tmp_path / "synthetic_test_video.mp4")


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure test estimators are cleaned up."""
    yield
    unregister_estimator("synthetic_test_a")
    unregister_estimator("synthetic_test_b")
    unregister_estimator("unavailable_test_estimator")


def test_estimator_absent_defaults_to_mediapipe(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    """When --estimator is absent, runner uses mediapipe with existing receipt schema."""
    import scripts.historical_capture as runner

    dest = tmp_path / "output_default"
    fake = FakeEstimator(
        name="MediaPipeEstimator",
        model_sha256="4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    )

    monkeypatch.setattr(runner, "create_estimator", lambda name, **opts: fake)
    monkeypatch.setattr(
        runner, "resolve_pose_model", lambda **opts: tmp_path / "mock.task"
    )
    monkeypatch.setattr(runner, "sha256_of", lambda p: fake.model_sha256)

    ret = runner.main(
        [
            str(synthetic_video),
            str(dest),
            "--subject",
            "test-subject",
            "--start",
            "0.2",
            "--end",
            "0.5",
        ]
    )
    assert ret == 0 or ret is None
    assert dest.is_dir()
    receipt_path = dest / "receipt.json"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["schema_version"] == "historical-capture/1.0.0"
    assert receipt["subject_id"] == "test-subject"
    detector = receipt["detector"]
    assert "version" in detector
    assert detector["model_sha256"] == fake.model_sha256
    assert detector["name"] in ("mediapipe", "MediaPipeEstimator")
    assert detector.get("registry_name", "mediapipe") == "mediapipe"
    assert fake.closed


def test_unknown_estimator_exits_nonzero_and_creates_no_dir(
    synthetic_video: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown estimator name leads to non-zero exit and no output directory."""
    import scripts.historical_capture as runner

    dest = tmp_path / "output_unknown"
    with pytest.raises(SystemExit) as exc_info:
        runner.main(
            [
                str(synthetic_video),
                str(dest),
                "--subject",
                "test-subject",
                "--start",
                "0.2",
                "--end",
                "0.5",
                "--estimator",
                "nonexistent_backend_name",
            ]
        )
    assert exc_info.value.code != 0
    assert not dest.exists()


def test_unavailable_estimator_surfaces_reason_and_writes_no_partial_receipt(
    synthetic_video: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unavailable estimator surfaces its reason and does not create receipt."""
    import scripts.historical_capture as runner

    register_estimator(
        EstimatorInfo(
            name="unavailable_test_estimator",
            display_name="Unavailable Backend",
            description="Testing unavailability",
            probe_module="nonexistent_probe_module_xyz_123",
            install_hint="remedy: install missing-xyz-package",
            factory=lambda **opts: FakeEstimator(),
        )
    )

    dest = tmp_path / "output_unavail"
    with pytest.raises(SystemExit) as exc_info:
        runner.main(
            [
                str(synthetic_video),
                str(dest),
                "--subject",
                "test-subject",
                "--start",
                "0.2",
                "--end",
                "0.5",
                "--estimator",
                "unavailable_test_estimator",
            ]
        )
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "missing-xyz-package" in (captured.err + captured.out + str(exc_info.value))
    assert not (dest / "receipt.json").exists()


def test_receipt_records_registry_name_and_weights_hash_differing_by_backend(
    synthetic_video: Path,
    tmp_path: Path,
) -> None:
    """Receipt records registry name and weights hash, differing between backends."""
    import scripts.historical_capture as runner

    hash_a = "1111111111111111111111111111111111111111111111111111111111111111"
    hash_b = "2222222222222222222222222222222222222222222222222222222222222222"

    register_estimator(
        EstimatorInfo(
            name="synthetic_test_a",
            display_name="Backend A",
            description="Test Backend A",
            probe_module="unittest",
            install_hint="ok",
            factory=lambda **opts: FakeEstimator(
                name="synthetic_test_a", model_sha256=hash_a
            ),
        )
    )
    register_estimator(
        EstimatorInfo(
            name="synthetic_test_b",
            display_name="Backend B",
            description="Test Backend B",
            probe_module="unittest",
            install_hint="ok",
            factory=lambda **opts: FakeEstimator(
                name="synthetic_test_b", model_sha256=hash_b
            ),
        )
    )

    dest_a = tmp_path / "out_a"
    dest_b = tmp_path / "out_b"

    ret_a = runner.main(
        [
            str(synthetic_video),
            str(dest_a),
            "--subject",
            "subj",
            "--start",
            "0.2",
            "--end",
            "0.5",
            "--estimator",
            "synthetic_test_a",
        ]
    )
    assert ret_a == 0 or ret_a is None

    ret_b = runner.main(
        [
            str(synthetic_video),
            str(dest_b),
            "--subject",
            "subj",
            "--start",
            "0.2",
            "--end",
            "0.5",
            "--estimator",
            "synthetic_test_b",
        ]
    )
    assert ret_b == 0 or ret_b is None

    receipt_a = json.loads((dest_a / "receipt.json").read_text(encoding="utf-8"))
    receipt_b = json.loads((dest_b / "receipt.json").read_text(encoding="utf-8"))

    assert receipt_a["detector"]["model_sha256"] == hash_a
    assert receipt_b["detector"]["model_sha256"] == hash_b
    assert (
        receipt_a["detector"].get("registry_name", receipt_a["detector"].get("name"))
        == "synthetic_test_a"
    )
    assert (
        receipt_b["detector"].get("registry_name", receipt_b["detector"].get("name"))
        == "synthetic_test_b"
    )
    assert receipt_a["detector"] != receipt_b["detector"]


def test_existing_output_directory_refuses_overwrite(
    synthetic_video: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Existing output directory triggers refusal matching overwrite guard."""
    import scripts.historical_capture as runner

    dest = tmp_path / "existing_dir"
    dest.mkdir()
    sentinel = dest / "precious_data.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        runner.main(
            [
                str(synthetic_video),
                str(dest),
                "--subject",
                "subj",
                "--start",
                "0.2",
                "--end",
                "0.5",
            ]
        )
    assert exc_info.value.code != 0
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    assert not (dest / "receipt.json").exists()


def test_hmr2_adapter_canonicalized_output_contract(tmp_path: Path) -> None:
    """HMR2 adapter canonical output preserves 3D in metres and flags unqualified scale."""
    from src.tools.hmr2_sidecar.run_hmr2 import _write_stub_artifacts

    joints3d, _, metadata = _write_stub_artifacts(tmp_path)
    adapter = HMR2Adapter()

    # Case 1: Missing focal/camera field -> typed unqualified flag
    canonical_unqualified = adapter.to_canonical_observations(joints3d)
    assert isinstance(canonical_unqualified, CanonicalObservations)
    assert canonical_unqualified.num_frames == 2
    assert canonical_unqualified.metadata["unit_system"] == "meters"
    assert canonical_unqualified.metadata["declared_frame"] == "camera"
    assert canonical_unqualified.metadata["qualification"] == "unqualified"
    assert canonical_unqualified.metadata.get("scale_qualification") == "unqualified"
    assert canonical_unqualified.metadata.get("unqualified") is True

    first_frame = canonical_unqualified.frames[0]
    assert "pelvis" in first_frame.markers
    pelvis = first_frame.markers["pelvis"]
    assert pelvis.z is not None

    # Case 2: Camera calibration with focal length provided -> qualified
    calib = Calibration(
        id="test-calib",
        cameras={
            "cam0": {
                "intrinsics": {
                    "focal_length": 1000.0,
                    "fx": 1000.0,
                    "fy": 1000.0,
                    "cx": 500.0,
                    "cy": 500.0,
                }
            }
        },
        source_fps=30.0,
    )
    canonical_qualified = adapter.to_canonical_observations(joints3d, calibration=calib)
    assert canonical_qualified.metadata["qualification"] == "qualified"
    assert canonical_qualified.metadata.get("unqualified") is False
