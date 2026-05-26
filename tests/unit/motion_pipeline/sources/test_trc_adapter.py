"""Tests for the TRC adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.trc_adapter import TRCAdapter

_TRC = "\n".join(
    [
        "PathFileType\t4\t(X/Y/Z)\ttiny.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        "100.0\t100.0\t3\t2\tmm\t100.0\t1\t3",
        "Frame#\tTime\tHEAD\t\t\tHIP\t\t\t",
        "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
        "1\t0.0\t1000\t2000\t3000\t100\t200\t300",
        "2\t0.01\t1010\t2010\t3010\t110\t210\t310",
        "3\t0.02\t1020\t2020\t3020\t120\t220\t320",
        "",
    ]
)


def _write_trc(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.trc"
    p.write_text(_TRC)
    return p


def test_trc_supports(tmp_path: Path) -> None:
    p = _write_trc(tmp_path)
    assert TRCAdapter.supports(p) is True


def test_trc_metadata(tmp_path: Path) -> None:
    md = TRCAdapter().metadata(_write_trc(tmp_path))
    assert md.format_name == "trc"
    assert md.frame_count == 3
    assert md.unit_system == "millimeters"
    assert md.fps == pytest.approx(100.0)


def test_trc_load_converts_mm_to_meters(tmp_path: Path) -> None:
    traj = TRCAdapter().load_checked(_write_trc(tmp_path))
    assert traj.num_frames == 3
    # 1000 mm -> 1.0 m
    head_first = traj.frames[0].markers["HEAD"]
    assert head_first.x == pytest.approx(1.0)
    assert head_first.y == pytest.approx(2.0)
    assert head_first.z == pytest.approx(3.0)


def test_trc_truncated_raises(tmp_path: Path) -> None:
    p = tmp_path / "short.trc"
    p.write_text("PathFileType\t4\nfoo\n")
    with pytest.raises(ValueError):
        TRCAdapter().load(p)


def test_trc_supports_os_error() -> None:
    from unittest.mock import patch

    with patch("builtins.open", side_effect=OSError):
        assert TRCAdapter.supports(Path("dummy.trc")) is False


def test_trc_parse_header_missing_values(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames",
            "100.0\t100.0",
            "Frame#\tTime\tHEAD",
            "\t\tX1\tY1\tZ1",
            "1\t0.0\t1.0\t2.0\t3.0",
        ]
    )
    p = tmp_path / "test.trc"
    p.write_text(content)
    md = TRCAdapter().metadata(p)
    assert md.frame_count == 0


def test_trc_metadata_fps_fallbacks(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames\tUnits",
            "invalid_fps\t60.0\t10\tmm",
            "Frame#\tTime\tHEAD",
            "\t\tX1\tY1\tZ1",
            "1\t0.0\t1.0\t2.0\t3.0",
        ]
    )
    p = tmp_path / "test1.trc"
    p.write_text(content)
    md = TRCAdapter().metadata(p)
    assert md.fps == 60.0
    assert md.frame_count == 10

    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames\tUnits",
            "invalid_fps\tinvalid_cam\tinvalid_frames\tmm",
            "Frame#\tTime\tHEAD",
            "\t\tX1\tY1\tZ1",
            "1\t0.0\t1.0\t2.0\t3.0",
        ]
    )
    p = tmp_path / "test2.trc"
    p.write_text(content)
    md = TRCAdapter().metadata(p)
    assert md.fps == 30.0
    assert md.frame_count == 0


def test_trc_load_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        TRCAdapter().load(Path("non_existent_file.trc"))


def test_trc_load_invalid_lines(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames\tUnits",
            "100.0\t100.0\t3\tmm",
            "Frame#\tTime\tHEAD",
            "\t\tX1\tY1\tZ1",
            "",
            "only_one_token",
            "1\t0.0\t1000\t2000\t3000",
        ]
    )
    p = tmp_path / "invalid1.trc"
    p.write_text(content)
    traj = TRCAdapter().load(p)
    assert traj.num_frames == 1

    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames\tUnits",
            "100.0\t100.0\t3\tmm",
            "Frame#\tTime\tHEAD",
            "\t\tX1\tY1\tZ1",
            "invalid_frame\t0.0\t1000\t2000\t3000",
        ]
    )
    p = tmp_path / "invalid2.trc"
    p.write_text(content)
    with pytest.raises(ValueError, match="has invalid frame/time"):
        TRCAdapter().load(p)

    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames\tUnits",
            "100.0\t100.0\t3\tmm",
            "Frame#\tTime\tHEAD\tHIP",
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
            "1\t0.0\t1000\t2000\t3000\t100",
        ]
    )
    p = tmp_path / "invalid3.trc"
    p.write_text(content)
    traj = TRCAdapter().load(p)
    assert "HEAD" in traj.frames[0].markers
    assert "HIP" not in traj.frames[0].markers

    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames\tUnits",
            "100.0\t100.0\t3\tmm",
            "Frame#\tTime\tHEAD\tHIP",
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
            "1\t0.0\t1000\t2000\t3000\t100\tinvalid\t300",
        ]
    )
    p = tmp_path / "invalid4.trc"
    p.write_text(content)
    traj = TRCAdapter().load(p)
    assert "HEAD" in traj.frames[0].markers
    assert "HIP" not in traj.frames[0].markers

    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames\tUnits",
            "100.0\t100.0\t3\tmm",
            "Frame#\tTime\tHEAD",
            "\t\tX1\tY1\tZ1",
        ]
    )
    p = tmp_path / "invalid5.trc"
    p.write_text(content)
    with pytest.raises(ValueError, match="has no data rows"):
        TRCAdapter().load(p)


def test_trc_load_via_rust(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    import numpy as np

    p = tmp_path / "tiny.trc"
    p.write_text(
        "\n".join(
            [
                "PathFileType\t4\t(X/Y/Z)\ttiny.trc",
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
                "100.0\t100.0\t3\t2\tmm\t100.0\t1\t3",
                "Frame#\tTime\tHEAD\t\t\tHIP\t\t\t",
                "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
                "1\t0.0\t1000\t2000\t3000\t100\t200\t300",
            ]
        )
    )

    mock_rust_io = MagicMock()
    mock_rust_io.parse_trc.return_value = {
        "positions": np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32),
        "labels": ["HEAD", "HIP"],
        "n_frames": 1,
        "fps": 100.0,
        "units": "mm",
    }
    with (
        patch(
            "src.shared.python.motion_pipeline.sources.trc_adapter._rust_io",
            mock_rust_io,
        ),
        patch("src.shared.python.motion_pipeline.sources.trc_adapter._HAS_RUST", True),
    ):
        traj = TRCAdapter().load(p)
        assert traj.num_frames == 1
        assert traj.frames[0].markers["HEAD"].x == pytest.approx(1.0)
        assert traj.frames[0].markers["HIP"].x == pytest.approx(0.1)


def test_trc_load_via_rust_edge_cases(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    import numpy as np

    mock_rust_io = MagicMock()
    mock_rust_io.parse_trc.return_value = {
        "positions": np.array(
            [[float("nan"), 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32
        ),
        "labels": ["HEAD", "HIP"],
        "n_frames": 1,
        "fps": 100.0,
        "units": "mm",
    }
    content = "\n".join(
        [
            "PathFileType\t4\t(X/Y/Z)\tfile.trc",
            "DataRate\tCameraRate\tNumFrames\tUnits",
            "100.0\t100.0\t1\tmm",
            "Frame#\tTime\tHEAD\tHIP",
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
        ]
    )
    p = tmp_path / "edge.trc"
    p.write_text(content)
    with (
        patch(
            "src.shared.python.motion_pipeline.sources.trc_adapter._rust_io",
            mock_rust_io,
        ),
        patch("src.shared.python.motion_pipeline.sources.trc_adapter._HAS_RUST", True),
    ):
        traj = TRCAdapter().load(p)
        assert traj.num_frames == 1
        assert "HEAD" not in traj.frames[0].markers
        assert traj.frames[0].markers["HIP"].x == pytest.approx(0.1)
        assert traj.frames[0].frame_index == 0
        assert traj.frames[0].timestamp == 0.0

        mock_rust_io.parse_trc.return_value["n_frames"] = 0
        mock_rust_io.parse_trc.return_value["positions"] = np.empty(
            (0, 6), dtype=np.float32
        )
        with pytest.raises(ValueError, match="has no data rows"):
            TRCAdapter().load(p)
