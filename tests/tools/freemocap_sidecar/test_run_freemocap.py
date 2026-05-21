"""Tests for the FreeMoCap sidecar runner.

These tests never touch real freemocap, mediapipe, OpenCV, or video
files. The subprocess invocation is mocked, and stub artifact behavior
is exercised through ``dry_run`` and ``FileNotFoundError`` paths.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.freemocap_sidecar import run_freemocap as mod
from src.tools.freemocap_sidecar.run_freemocap import (
    FreeMoCapResult,
    FreeMoCapSidecarError,
    _validate_output,
    _write_stub_artifacts,
    main,
    run_freemocap_sidecar,
)

# ---------------------------------------------------------------------------
# Helpers / stub artifacts
# ---------------------------------------------------------------------------


class TestStubArtifacts:
    def test_write_stub_artifacts_creates_files(self, tmp_path: Path) -> None:
        landmarks, metadata = _write_stub_artifacts(tmp_path)
        assert landmarks == tmp_path / "landmarks.csv"
        assert metadata == tmp_path / "metadata.json"
        assert landmarks.exists()
        assert metadata.exists()

    def test_write_stub_artifacts_creates_parent_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        landmarks, metadata = _write_stub_artifacts(nested)
        assert landmarks.exists() and metadata.exists()

    def test_landmarks_header_and_row(self, tmp_path: Path) -> None:
        landmarks, _ = _write_stub_artifacts(tmp_path)
        text = landmarks.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        assert lines[0] == "frame,landmark_id,x,y,z"
        assert lines[1] == "0,0,0.0,0.0,0.0"

    def test_metadata_is_valid_json_with_stub_flag(self, tmp_path: Path) -> None:
        _, metadata = _write_stub_artifacts(tmp_path)
        data = json.loads(metadata.read_text(encoding="utf-8"))
        assert data["stub"] is True
        assert data["n_frames"] == 1
        assert data["n_landmarks"] == 1
        assert data["fps"] == 0
        assert data["freemocap_version"] == "stub"


class TestValidateOutput:
    def test_returns_paths_when_both_exist(self, tmp_path: Path) -> None:
        (tmp_path / "landmarks.csv").write_text("x", encoding="utf-8")
        (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
        lm, md = _validate_output(tmp_path)
        assert lm == tmp_path / "landmarks.csv"
        assert md == tmp_path / "metadata.json"

    def test_returns_none_when_missing_landmarks(self, tmp_path: Path) -> None:
        (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
        assert _validate_output(tmp_path) == (None, None)

    def test_returns_none_when_missing_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "landmarks.csv").write_text("x", encoding="utf-8")
        assert _validate_output(tmp_path) == (None, None)

    def test_returns_none_when_both_missing(self, tmp_path: Path) -> None:
        assert _validate_output(tmp_path) == (None, None)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


class TestFreeMoCapResult:
    def test_defaults(self, tmp_path: Path) -> None:
        r = FreeMoCapResult(
            success=True,
            output_dir=tmp_path,
            return_code=0,
            used_real_freemocap=False,
        )
        assert r.landmarks_csv is None
        assert r.metadata_json is None
        assert r.stderr_tail == ""
        assert r.extra == {}

    def test_frozen(self, tmp_path: Path) -> None:
        r = FreeMoCapResult(
            success=True,
            output_dir=tmp_path,
            return_code=0,
            used_real_freemocap=False,
        )
        with pytest.raises(
            (AttributeError, TypeError, dataclasses.FrozenInstanceError)
        ):
            r.success = False  # type: ignore[misc]

    def test_error_class_is_runtimeerror(self) -> None:
        assert issubclass(FreeMoCapSidecarError, RuntimeError)


# ---------------------------------------------------------------------------
# run_freemocap_sidecar — dry-run + subprocess paths
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_writes_stubs_and_success(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = run_freemocap_sidecar(
            input_dir=tmp_path / "in",
            output_dir=out,
            dry_run=True,
        )
        assert result.success is True
        assert result.used_real_freemocap is False
        assert result.return_code == 0
        assert result.landmarks_csv == out / "landmarks.csv"
        assert result.metadata_json == out / "metadata.json"
        assert result.landmarks_csv.exists()
        assert result.metadata_json.exists()

    def test_dry_run_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "out"
        result = run_freemocap_sidecar(input_dir=tmp_path, output_dir=out, dry_run=True)
        assert out.is_dir()
        assert result.output_dir == out


class TestSubprocessPaths:
    def _mock_completed(self, returncode: int = 0, stderr: str = "") -> MagicMock:
        proc = MagicMock()
        proc.returncode = returncode
        proc.stderr = stderr
        proc.stdout = ""
        return proc

    def test_success_when_subprocess_ok_and_files_exist(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()

        def fake_run(cmd, **kwargs):
            # Simulate real freemocap writing the outputs
            (out / "landmarks.csv").write_text("frame,landmark_id,x,y,z\n")
            (out / "metadata.json").write_text("{}")
            return self._mock_completed(0, "ok\n")

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = run_freemocap_sidecar(
                input_dir=tmp_path / "in",
                output_dir=out,
                freemocap_env_python="/fake/python",
            )
        assert result.success is True
        assert result.used_real_freemocap is True
        assert result.return_code == 0
        assert result.landmarks_csv == out / "landmarks.csv"
        assert result.metadata_json == out / "metadata.json"

    def test_success_uses_sys_executable_when_no_env_python(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            (out / "landmarks.csv").write_text("x")
            (out / "metadata.json").write_text("{}")
            return self._mock_completed(0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            run_freemocap_sidecar(input_dir=tmp_path / "in", output_dir=out)

        import sys as _sys

        assert captured["cmd"][0] == _sys.executable
        assert "-m" in captured["cmd"]
        assert "freemocap" in captured["cmd"]

    def test_command_includes_input_and_output(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            (out / "landmarks.csv").write_text("x")
            (out / "metadata.json").write_text("{}")
            return self._mock_completed(0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            run_freemocap_sidecar(input_dir=tmp_path / "session", output_dir=out)

        cmd = captured["cmd"]
        assert "--input" in cmd
        assert "--output" in cmd
        assert str(tmp_path / "session") in cmd
        assert str(out) in cmd

    def test_subprocess_passes_timeout(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            (out / "landmarks.csv").write_text("x")
            (out / "metadata.json").write_text("{}")
            return self._mock_completed(0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            run_freemocap_sidecar(input_dir=tmp_path, output_dir=out, timeout_s=42.0)

        assert captured["timeout"] == 42.0
        assert captured["check"] is False
        assert captured["capture_output"] is True
        assert captured["text"] is True

    def test_filenotfound_falls_back_to_stubs(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        with patch.object(
            subprocess, "run", side_effect=FileNotFoundError("no python")
        ):
            result = run_freemocap_sidecar(
                input_dir=tmp_path / "in",
                output_dir=out,
                freemocap_env_python="/no/such/python",
            )
        assert result.success is False
        assert result.used_real_freemocap is False
        assert result.return_code == 127
        assert result.landmarks_csv == out / "landmarks.csv"
        assert result.metadata_json == out / "metadata.json"
        assert result.landmarks_csv.exists()
        assert "no python" in result.stderr_tail

    def test_timeout_returns_failure(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        exc = subprocess.TimeoutExpired(cmd=["python"], timeout=1.0)
        with patch.object(subprocess, "run", side_effect=exc):
            result = run_freemocap_sidecar(
                input_dir=tmp_path / "in",
                output_dir=out,
                timeout_s=1.0,
            )
        assert result.success is False
        assert result.return_code == -1
        assert result.used_real_freemocap is True
        assert "timeout" in result.stderr_tail.lower()
        # No stub files were written on timeout
        assert result.landmarks_csv is None
        assert result.metadata_json is None

    def test_no_module_named_freemocap_falls_back_to_stubs(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 1
        proc.stderr = "ModuleNotFoundError: No module named 'freemocap'\n"
        with patch.object(subprocess, "run", return_value=proc):
            result = run_freemocap_sidecar(
                input_dir=tmp_path / "in",
                output_dir=out,
            )
        assert result.success is False
        assert result.used_real_freemocap is False
        assert result.return_code == 1
        assert result.landmarks_csv == out / "landmarks.csv"
        assert result.landmarks_csv.exists()
        assert "No module named 'freemocap'" in result.stderr_tail

    def test_nonzero_exit_returns_failure(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 2
        proc.stderr = "some other error\n"
        with patch.object(subprocess, "run", return_value=proc):
            result = run_freemocap_sidecar(
                input_dir=tmp_path / "in",
                output_dir=out,
            )
        assert result.success is False
        assert result.used_real_freemocap is True
        assert result.return_code == 2
        assert result.landmarks_csv is None
        assert result.metadata_json is None
        assert "some other error" in result.stderr_tail

    def test_exit_zero_but_missing_outputs(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        # subprocess "succeeds" but never writes files
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = "warn\n"
        with patch.object(subprocess, "run", return_value=proc):
            result = run_freemocap_sidecar(
                input_dir=tmp_path / "in",
                output_dir=out,
            )
        assert result.success is False
        assert result.return_code == 0
        assert result.used_real_freemocap is True
        assert result.landmarks_csv is None
        assert result.metadata_json is None

    def test_stderr_tail_is_truncated(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        big_stderr = "x" * 10_000
        proc = MagicMock()
        proc.returncode = 5
        proc.stderr = big_stderr
        with patch.object(subprocess, "run", return_value=proc):
            result = run_freemocap_sidecar(
                input_dir=tmp_path / "in",
                output_dir=out,
            )
        assert len(result.stderr_tail) == 4096
        assert result.stderr_tail == big_stderr[-4096:]

    def test_stderr_none_safely_handled(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 3
        proc.stderr = None
        with patch.object(subprocess, "run", return_value=proc):
            result = run_freemocap_sidecar(
                input_dir=tmp_path / "in",
                output_dir=out,
            )
        assert result.success is False
        assert result.stderr_tail == ""


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_dry_run_returns_zero(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        rc = main(
            [
                "--input",
                str(tmp_path / "in"),
                "--output",
                str(out),
                "--dry-run",
            ]
        )
        assert rc == 0
        assert (out / "landmarks.csv").exists()
        assert (out / "metadata.json").exists()

    def test_main_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        rc = main(
            [
                "--input",
                str(tmp_path / "in"),
                "--output",
                str(out),
                "--dry-run",
                "--json",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["success"] is True
        assert payload["return_code"] == 0
        assert payload["used_real_freemocap"] is False
        assert payload["landmarks_csv"].endswith("landmarks.csv")
        assert payload["metadata_json"].endswith("metadata.json")
        # Path objects must have been stringified for JSON
        assert isinstance(payload["output_dir"], str)

    def test_main_failure_returns_nonzero(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 7
        proc.stderr = "bad\n"
        with patch.object(subprocess, "run", return_value=proc):
            rc = main(
                [
                    "--input",
                    str(tmp_path / "in"),
                    "--output",
                    str(out),
                ]
            )
        assert rc == 7

    def test_main_failure_with_returncode_zero_still_nonzero(
        self, tmp_path: Path
    ) -> None:
        # subprocess exits 0 but no output files -> failure, exit must be >= 1
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = ""
        with patch.object(subprocess, "run", return_value=proc):
            rc = main(
                [
                    "--input",
                    str(tmp_path / "in"),
                    "--output",
                    str(out),
                ]
            )
        assert rc >= 1

    def test_main_passes_env_python_and_timeout(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        called = {}

        def fake_runner(**kwargs):
            called.update(kwargs)
            return FreeMoCapResult(
                success=True,
                output_dir=Path(kwargs["output_dir"]),
                return_code=0,
                used_real_freemocap=True,
            )

        with patch.object(mod, "run_freemocap_sidecar", side_effect=fake_runner):
            rc = main(
                [
                    "--input",
                    str(tmp_path / "in"),
                    "--output",
                    str(out),
                    "--env-python",
                    "/custom/python",
                    "--timeout",
                    "12.5",
                ]
            )
        assert rc == 0
        assert called["freemocap_env_python"] == "/custom/python"
        assert called["timeout_s"] == 12.5
        assert called["dry_run"] is False

    def test_main_requires_input_and_output(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_main_configures_logging_when_no_handlers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import logging

        # Strip handlers so the basicConfig branch runs.
        root = logging.getLogger()
        old_handlers = root.handlers[:]
        root.handlers = []
        try:
            rc = main(
                [
                    "--input",
                    str(tmp_path / "in"),
                    "--output",
                    str(tmp_path / "out"),
                    "--dry-run",
                ]
            )
        finally:
            root.handlers = old_handlers
        assert rc == 0
