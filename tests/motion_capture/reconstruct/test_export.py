"""TRC/canonical export read back by the motion pipeline (#9664); joint exclusion (#9662)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.export import (
    canonical_payload,
    export_reconstruction,
    trc_text,
    write_trc,
)
from src.motion_capture.reconstruct.pipeline import _exclude
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

pytestmark = pytest.mark.unit


def _joints(frames: int = 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    joints = rng.normal(size=(frames, len(JOINT_NAMES), 3))
    joints[2, 3] = np.nan  # one missing joint
    return joints


def test_trc_round_trips_through_the_motion_pipeline_adapter(tmp_path: Path) -> None:
    from src.shared.python.motion_pipeline.sources.trc_adapter import TRCAdapter

    joints = _joints()
    path = write_trc(joints, 120.0, tmp_path / "recon.trc")
    assert TRCAdapter.supports(path)
    adapter = TRCAdapter()
    meta = adapter.metadata(path)
    assert meta.fps == pytest.approx(120.0)
    seq = adapter.load(path)
    frames = list(seq.frames)
    assert len(frames) == 5
    first = frames[0].markers
    assert set(first) == set(JOINT_NAMES)
    np.testing.assert_allclose(
        [first["neck"].x, first["neck"].y, first["neck"].z],
        joints[0, JOINT_NAMES.index("neck")],
        atol=1e-3,
    )
    missing = frames[2].markers.get(JOINT_NAMES[3])
    assert missing is None or not np.isfinite(missing.x)  # reader drops NaN markers
    intact = frames[2].markers[JOINT_NAMES[4]]  # the column after it did not shift
    np.testing.assert_allclose([intact.x, intact.y, intact.z], joints[2, 4], atol=1e-3)


def test_trc_text_contract_and_canonical_payload() -> None:
    text = trc_text(_joints(3), 60.0)
    header = text.splitlines()
    assert header[0].startswith("PathFileType\t4\t(X/Y/Z)")
    assert header[2].split("\t")[:5] == ["60", "60", "3", str(len(JOINT_NAMES)), "mm"]
    with pytest.raises(Exception, match="positive"):
        trc_text(_joints(3), 0.0)
    payload = canonical_payload(_joints(3), 60.0)
    assert payload["units"] == "m" and len(payload["frames"]) == 3
    assert payload["frames"][2]["joints_m"][3] == [None, None, None]


def test_export_reconstruction_writes_both_files(tmp_path: Path) -> None:
    np.save(tmp_path / "joints_3d_m.npy", _joints(4))
    (tmp_path / "session_reconstruction.json").write_text(
        json.dumps({"fps": 119.7, "views": ["a", "b"], "rms_px": 1.2}),
        encoding="utf-8",
    )
    written = export_reconstruction(tmp_path)
    assert Path(written["trc"]).is_file() and Path(written["json"]).is_file()
    exported = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert exported["fps"] == pytest.approx(119.7)
    assert exported["provenance"]["views"] == ["a", "b"]


def test_exclude_zeroes_confidence_only_for_named_joints() -> None:
    payload = {
        "frames": [
            {
                "keypoints_px": [[1, 1]] * len(JOINT_NAMES),
                "confidence": [0.9] * len(JOINT_NAMES),
            }
        ]
    }
    out = _exclude(payload, ["nose", "left_ankle"])
    conf = out["frames"][0]["confidence"]
    assert conf[JOINT_NAMES.index("nose")] == 0.0
    assert conf[JOINT_NAMES.index("left_ankle")] == 0.0
    assert sum(1 for c in conf if c == 0.9) == len(JOINT_NAMES) - 2
    assert payload["frames"][0]["confidence"][0] == 0.9  # input untouched
