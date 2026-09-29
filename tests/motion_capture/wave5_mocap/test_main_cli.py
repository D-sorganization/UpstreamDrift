"""Fast tests for motion_capture.freemocap_ingest CLI entrypoints."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from motion_capture.freemocap_ingest import __main__ as cli_main
from motion_capture.freemocap_ingest import launcher as launcher_mod
from motion_capture.freemocap_ingest import output_adapter as adapter_mod

pytestmark = pytest.mark.unit


def _make_csv(path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["frame_num", "timestamp", "nose_x", "nose_y", "nose_z", "nose_conf"]
        )
        for i in range(2):
            w.writerow([i, i * 0.1, 0.0, 0.0, 0.0, 0.9])


# ---------------------------------------------------------------------------
# __main__.main
# ---------------------------------------------------------------------------


def test_main_no_args_errors() -> None:
    with patch.object(sys, "argv", ["prog"]), pytest.raises(SystemExit):
        cli_main.main()


def test_main_parse_mode(tmp_path: Path) -> None:
    out = tmp_path / "outdir"
    out.mkdir()
    _make_csv(out / "freemocap_3d_landmarks_x.csv")
    (out / "camera_calibration.json").write_text('{"a": 1}')

    npy = tmp_path / "out.npy"
    csv_out = tmp_path / "out.csv"
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            str(out),
            "--parse",
            "--export-npy",
            str(npy),
            "--export-csv",
            str(csv_out),
            "-v",
        ],
    ):
        rc = cli_main.main()
    assert rc == 0
    assert npy.exists()
    assert csv_out.exists()


def test_main_dry_run_capture(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    with patch.object(sys, "argv", ["prog", str(session), "--dry-run"]):
        rc = cli_main.main()
    assert rc == 0


def test_main_capture_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path / "session"
    session.mkdir()
    fake_launcher = MagicMock()
    fake_launcher.launch.return_value = launcher_mod.LaunchResult(
        success=True,
        return_code=0,
        output_dir=session / "freemocap_output",
        log_file=session / "logs" / "x.log",
    )
    monkeypatch.setattr(cli_main, "FreeMoCapLauncher", lambda: fake_launcher)

    with patch.object(sys, "argv", ["prog", str(session)]):
        rc = cli_main.main()
    assert rc == 0


def test_main_capture_failure_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    fake_launcher = MagicMock()
    fake_launcher.launch.return_value = launcher_mod.LaunchResult(
        success=False,
        return_code=2,
        output_dir=None,
        log_file=session / "x.log",
        error_message="nope",
    )
    monkeypatch.setattr(cli_main, "FreeMoCapLauncher", lambda: fake_launcher)
    with patch.object(sys, "argv", ["prog", str(session)]):
        rc = cli_main.main()
    assert rc == 1


def test_main_capture_then_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    out = session / "freemocap_output"
    out.mkdir()
    _make_csv(out / "freemocap_3d_landmarks_y.csv")
    fake_launcher = MagicMock()
    fake_launcher.launch.return_value = launcher_mod.LaunchResult(
        success=True,
        return_code=0,
        output_dir=out,
        log_file=session / "x.log",
    )
    monkeypatch.setattr(cli_main, "FreeMoCapLauncher", lambda: fake_launcher)

    npy = tmp_path / "data.npy"
    csv_out = tmp_path / "data.csv"
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            str(session),
            "--parse-output",
            "--export-npy",
            str(npy),
            "--export-csv",
            str(csv_out),
        ],
    ):
        rc = cli_main.main()
    assert rc == 0
    assert npy.exists()
    assert csv_out.exists()


# ---------------------------------------------------------------------------
# launcher.main and output_adapter.main
# ---------------------------------------------------------------------------


def test_launcher_main_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path / "session"
    session.mkdir()
    fake_launcher = MagicMock()
    fake_launcher.launch.return_value = launcher_mod.LaunchResult(
        success=True, return_code=0, output_dir=session, log_file=None
    )
    monkeypatch.setattr(launcher_mod, "FreeMoCapLauncher", lambda: fake_launcher)
    with (
        patch.object(sys, "argv", ["prog", str(session)]),
        pytest.raises(SystemExit) as ei,
    ):
        launcher_mod.main()
    assert ei.value.code == 0


def test_launcher_main_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path / "session"
    session.mkdir()
    fake_launcher = MagicMock()
    fake_launcher.launch.return_value = launcher_mod.LaunchResult(
        success=False,
        return_code=1,
        output_dir=None,
        log_file=session / "x.log",
        error_message="bad",
    )
    monkeypatch.setattr(launcher_mod, "FreeMoCapLauncher", lambda: fake_launcher)
    with (
        patch.object(sys, "argv", ["prog", str(session), "--gui", "-v"]),
        pytest.raises(SystemExit) as ei,
    ):
        launcher_mod.main()
    assert ei.value.code == 1


def test_output_adapter_main(tmp_path: Path) -> None:
    out = tmp_path / "outdir"
    out.mkdir()
    _make_csv(out / "freemocap_3d_landmarks_a.csv")
    npy = tmp_path / "x.npy"
    csv_out = tmp_path / "x.csv"
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            str(out),
            "--export-npy",
            str(npy),
            "--export-csv",
            str(csv_out),
            "-v",
        ],
    ):
        adapter_mod.main()
    assert npy.exists()
    assert csv_out.exists()
