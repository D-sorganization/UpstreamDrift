"""Tests for the 4D-Humans / HMR2 sidecar joints3d.csv adapter.

Golden fixtures are built in-test from the sidecar's column contract
(``JOINTS3D_COLUMNS`` in :mod:`src.tools.hmr2_sidecar.run_hmr2`), plus
the sidecar's own stub artifacts for contract-roundtrip coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.contracts import KeypointSequence
from src.shared.python.motion_pipeline.sources import detect_format, load_any
from src.shared.python.motion_pipeline.sources.csv_adapter import CSVAdapter
from src.shared.python.motion_pipeline.sources.hmr2_adapter import HMR2Adapter
from src.tools.hmr2_sidecar.run_hmr2 import (
    JOINTS3D_COLUMNS,
    SMPL_BODY_JOINTS,
    _write_stub_artifacts,
)

pytestmark = pytest.mark.unit

_N_JOINTS = len(SMPL_BODY_JOINTS)


def test_joint_contract_matches_sidecar() -> None:
    """The adapter's duplicated joint list must equal the sidecar's.

    The adapter may not import ``src.tools`` (LoD gate), so it carries
    its own copy of the SMPL joint contract; this test is the sync gate.
    """
    from src.shared.python.motion_pipeline.sources import hmr2_adapter

    assert hmr2_adapter.SMPL_BODY_JOINTS == SMPL_BODY_JOINTS


def _write_golden_csv(
    path: Path,
    n_frames: int = 4,
    fps: float = 100.0,
) -> Path:
    """Write a deterministic joints3d.csv following the sidecar contract."""
    lines = [",".join(JOINTS3D_COLUMNS)]
    for i in range(n_frames):
        coords: list[str] = []
        for j in range(_N_JOINTS):
            coords.extend(
                [
                    f"{0.01 * j + 0.001 * i:.6f}",
                    f"{1.0 - 0.02 * j:.6f}",
                    f"{-0.5 + 0.03 * j + 0.002 * i:.6f}",
                ]
            )
        lines.append(f"{i},{i / fps}," + ",".join(coords))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestSupports:
    def test_supports_golden_header(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv")
        assert HMR2Adapter.supports(p) is True

    def test_detect_format_routes_to_hmr2(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv")
        assert detect_format(p) is HMR2Adapter

    def test_supports_stub_artifacts(self, tmp_path: Path) -> None:
        joints3d, _, _ = _write_stub_artifacts(tmp_path)
        assert HMR2Adapter.supports(joints3d) is True
        assert detect_format(joints3d) is HMR2Adapter

    def test_rejects_wrong_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "joints3d.txt"
        _write_golden_csv(p)
        assert HMR2Adapter.supports(p) is False

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        assert HMR2Adapter.supports(tmp_path / "nope.csv") is False

    def test_rejects_generic_trajectory_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "generic.csv"
        p.write_text(
            "frame,timestamp,x_hip,y_hip,z_hip\n0,0.0,1.0,2.0,3.0\n",
            encoding="utf-8",
        )
        assert HMR2Adapter.supports(p) is False

    def test_generic_csv_still_routes_to_csv_adapter(self, tmp_path: Path) -> None:
        """No-regression: HMR2 registration must not steal generic CSVs."""
        p = tmp_path / "generic.csv"
        p.write_text(
            "frame,timestamp,x_hip,y_hip,z_hip\n0,0.0,1.0,2.0,3.0\n",
            encoding="utf-8",
        )
        assert detect_format(p) is CSVAdapter

    def test_rejects_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        assert HMR2Adapter.supports(p) is False

    def test_rejects_triplet_shape_without_metadata(self, tmp_path: Path) -> None:
        """Non-SMPL triplet CSVs need a sidecar metadata.json to be claimed."""
        p = tmp_path / "other.csv"
        p.write_text(
            "frame,time,tip_x,tip_y,tip_z\n0,0.0,1.0,2.0,3.0\n", encoding="utf-8"
        )
        assert HMR2Adapter.supports(p) is False

    def test_accepts_triplet_shape_with_hmr2_metadata(self, tmp_path: Path) -> None:
        p = tmp_path / "other.csv"
        p.write_text(
            "frame,time,tip_x,tip_y,tip_z\n0,0.0,1.0,2.0,3.0\n", encoding="utf-8"
        )
        (tmp_path / "metadata.json").write_text(
            json.dumps({"tool": "4D-Humans", "fps": 50.0}), encoding="utf-8"
        )
        assert HMR2Adapter.supports(p) is True

    def test_ignores_non_hmr2_metadata(self, tmp_path: Path) -> None:
        p = tmp_path / "other.csv"
        p.write_text(
            "frame,time,tip_x,tip_y,tip_z\n0,0.0,1.0,2.0,3.0\n", encoding="utf-8"
        )
        (tmp_path / "metadata.json").write_text(
            json.dumps({"tool": "somethingelse"}), encoding="utf-8"
        )
        assert HMR2Adapter.supports(p) is False


class TestLoad:
    def test_load_any_returns_3d_sequence(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv", n_frames=4, fps=100.0)
        seq = load_any(p)
        assert isinstance(seq, KeypointSequence)
        assert seq.num_frames == 4
        assert seq.num_keypoints == _N_JOINTS
        for frame in seq.frames:
            assert frame.schema_name == "custom"
            assert frame.check_keypoint_depth_consistency()
            assert all(kp.z is not None for kp in frame.keypoints)

    def test_joint_names_preserved_in_smpl_order(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv")
        seq = HMR2Adapter().load_checked(p)
        names = [kp.name for kp in seq.frames[0].keypoints]
        assert names == list(SMPL_BODY_JOINTS)

    def test_timestamps_from_time_column(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv", n_frames=3, fps=100.0)
        seq = HMR2Adapter().load_checked(p)
        timestamps = [f.timestamp for f in seq.frames]
        assert timestamps == pytest.approx([0.0, 0.01, 0.02])
        assert timestamps == sorted(timestamps)
        assert [f.frame_index for f in seq.frames] == [0, 1, 2]

    def test_coordinate_values_roundtrip(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv")
        seq = HMR2Adapter().load_checked(p)
        kp = seq.frames[1].keypoints[2]  # frame 1, right_hip (j=2)
        assert kp.x == pytest.approx(0.01 * 2 + 0.001 * 1)
        assert kp.y == pytest.approx(1.0 - 0.02 * 2)
        assert kp.z == pytest.approx(-0.5 + 0.03 * 2 + 0.002 * 1)

    def test_load_checked_postconditions_pass(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv")
        # load_checked raises AdapterContractError on any violation
        seq = HMR2Adapter().load_checked(p)
        assert seq.metadata["unit_system"] == "meters"
        assert seq.metadata["source_file"] == str(p)

    def test_stub_artifacts_load(self, tmp_path: Path) -> None:
        """The sidecar's stub joints3d.csv satisfies the adapter contract."""
        joints3d, _, _ = _write_stub_artifacts(tmp_path)
        seq = load_any(joints3d)
        assert isinstance(seq, KeypointSequence)
        assert seq.num_frames == 2
        assert seq.num_keypoints == _N_JOINTS

    def test_non_finite_keypoints_dropped(self, tmp_path: Path) -> None:
        p = tmp_path / "joints3d.csv"
        lines = [",".join(JOINTS3D_COLUMNS)]
        coords = ["0.1"] * (3 * _N_JOINTS)
        coords[0] = "nan"  # pelvis_x non-finite -> pelvis dropped
        lines.append("0,0.0," + ",".join(coords))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        seq = HMR2Adapter().load(p)
        names = [kp.name for kp in seq.frames[0].keypoints]
        assert "pelvis" not in names
        assert len(names) == _N_JOINTS - 1


class TestMetadata:
    def test_metadata_fields(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv", n_frames=5, fps=100.0)
        md = HMR2Adapter().metadata(p)
        assert md.format_name == "hmr2"
        assert md.frame_count == 5
        assert md.unit_system == "meters"
        assert md.keypoint_schema == "custom"
        assert md.fps == pytest.approx(100.0)

    def test_metadata_prefers_sidecar_fps(self, tmp_path: Path) -> None:
        p = _write_golden_csv(tmp_path / "joints3d.csv", fps=100.0)
        (tmp_path / "metadata.json").write_text(
            json.dumps({"tool": "4D-Humans", "fps": 240.0}), encoding="utf-8"
        )
        md = HMR2Adapter().metadata(p)
        assert md.fps == pytest.approx(240.0)


class TestMalformedInputs:
    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            HMR2Adapter().load(tmp_path / "nope.csv")

    def test_wrong_header_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "joints3d.csv"
        p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="sidecar\\s+contract"):
            HMR2Adapter().load(p)

    def test_no_data_rows_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "joints3d.csv"
        p.write_text(",".join(JOINTS3D_COLUMNS) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no data rows"):
            HMR2Adapter().load(p)

    def test_bad_time_cell_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "joints3d.csv"
        coords = ",".join(["0.1"] * (3 * _N_JOINTS))
        p.write_text(
            ",".join(JOINTS3D_COLUMNS) + f"\n0,not_a_time,{coords}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="frame/time"):
            HMR2Adapter().load(p)

    def test_bad_coordinate_cell_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "joints3d.csv"
        coords = ["0.1"] * (3 * _N_JOINTS)
        coords[5] = "oops"
        p.write_text(
            ",".join(JOINTS3D_COLUMNS) + "\n0,0.0," + ",".join(coords) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-numeric coordinate"):
            HMR2Adapter().load(p)

    def test_all_non_finite_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "joints3d.csv"
        coords = ",".join(["nan"] * (3 * _N_JOINTS))
        p.write_text(
            ",".join(JOINTS3D_COLUMNS) + f"\n0,0.0,{coords}\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="no usable frames"):
            HMR2Adapter().load(p)
