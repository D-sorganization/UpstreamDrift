"""Tests for the 4D-Humans / HMR2 sidecar runner.

These tests never touch a real 4D-Humans install, SMPL weights, or
video files. The subprocess invocation is mocked, and stub artifact
behavior is exercised through ``dry_run``, unconfigured-command, and
``FileNotFoundError`` paths.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.hmr2_sidecar import run_hmr2 as mod
from src.tools.hmr2_sidecar.run_hmr2 import (
    HMR2_COMMAND_ENV,
    JOINTS3D_COLUMNS,
    NUM_BETAS,
    SMPL_BODY_JOINTS,
    HMR2Result,
    HMR2SidecarError,
    _resolve_command,
    _validate_output,
    _write_stub_artifacts,
    main,
    run_hmr2_sidecar,
)


@pytest.fixture(autouse=True)
def _no_ambient_hmr2_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the ambient environment from configuring a real command."""
    monkeypatch.delenv(HMR2_COMMAND_ENV, raising=False)


# ---------------------------------------------------------------------------
# Joint contract
# ---------------------------------------------------------------------------


class TestJointContract:
    def test_22_smpl_body_joints(self) -> None:
        assert len(SMPL_BODY_JOINTS) == 22
        assert SMPL_BODY_JOINTS[0] == "pelvis"
        assert SMPL_BODY_JOINTS[-1] == "right_wrist"
        assert len(set(SMPL_BODY_JOINTS)) == 22

    def test_joints3d_columns_shape(self) -> None:
        assert JOINTS3D_COLUMNS[:2] == ("frame", "time")
        assert len(JOINTS3D_COLUMNS) == 2 + 3 * 22
        assert JOINTS3D_COLUMNS[2:5] == ("pelvis_x", "pelvis_y", "pelvis_z")


# ---------------------------------------------------------------------------
# Helpers / stub artifacts
# ---------------------------------------------------------------------------


class TestStubArtifacts:
    def test_write_stub_artifacts_creates_files(self, tmp_path: Path) -> None:
        joints3d, betas, metadata = _write_stub_artifacts(tmp_path)
        assert joints3d == tmp_path / "joints3d.csv"
        assert betas == tmp_path / "betas.json"
        assert metadata == tmp_path / "metadata.json"
        assert joints3d.exists()
        assert betas.exists()
        assert metadata.exists()

    def test_write_stub_artifacts_creates_parent_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        joints3d, betas, metadata = _write_stub_artifacts(nested)
        assert joints3d.exists() and betas.exists() and metadata.exists()

    def test_joints3d_header_and_rows(self, tmp_path: Path) -> None:
        joints3d, _, _ = _write_stub_artifacts(tmp_path)
        lines = joints3d.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == ",".join(JOINTS3D_COLUMNS)
        assert len(lines) == 3  # header + 2 stub frames
        first = lines[1].split(",")
        assert first[0] == "0"
        assert float(first[1]) == 0.0
        assert len(first) == len(JOINTS3D_COLUMNS)

    def test_betas_json_schema(self, tmp_path: Path) -> None:
        _, betas, _ = _write_stub_artifacts(tmp_path)
        data = json.loads(betas.read_text(encoding="utf-8"))
        assert data["gender"] == "neutral"
        assert len(data["betas"]) == NUM_BETAS
        assert all(isinstance(b, float) for b in data["betas"])

    def test_metadata_is_valid_json_with_stub_flag(self, tmp_path: Path) -> None:
        _, _, metadata = _write_stub_artifacts(tmp_path, Path("clip.mp4"), fps=25.0)
        data = json.loads(metadata.read_text(encoding="utf-8"))
        assert data["stub"] is True
        assert data["tool"] == "4D-Humans"
        assert data["tool_version"] == "stub"
        assert data["source_video"] == "clip.mp4"
        assert data["fps"] == 25.0
        assert data["joint_names"] == list(SMPL_BODY_JOINTS)


class TestValidateOutput:
    def _touch_all(self, d: Path) -> None:
        (d / "joints3d.csv").write_text("x", encoding="utf-8")
        (d / "betas.json").write_text("{}", encoding="utf-8")
        (d / "metadata.json").write_text("{}", encoding="utf-8")

    def test_returns_paths_when_all_exist(self, tmp_path: Path) -> None:
        self._touch_all(tmp_path)
        joints3d, betas, metadata = _validate_output(tmp_path)
        assert joints3d == tmp_path / "joints3d.csv"
        assert betas == tmp_path / "betas.json"
        assert metadata == tmp_path / "metadata.json"

    @pytest.mark.parametrize("missing", ["joints3d.csv", "betas.json", "metadata.json"])
    def test_returns_none_when_any_missing(self, tmp_path: Path, missing: str) -> None:
        self._touch_all(tmp_path)
        (tmp_path / missing).unlink()
        assert _validate_output(tmp_path) == (None, None, None)

    def test_returns_none_when_all_missing(self, tmp_path: Path) -> None:
        assert _validate_output(tmp_path) == (None, None, None)


class TestResolveCommand:
    def test_explicit_string_is_shlex_split(self) -> None:
        assert _resolve_command("python demo.py --flag") == [
            "python",
            "demo.py",
            "--flag",
        ]

    def test_explicit_sequence_used_verbatim(self) -> None:
        assert _resolve_command(["python", "demo.py"]) == ["python", "demo.py"]

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(HMR2_COMMAND_ENV, "/opt/hmr2/run.sh --fast")
        assert _resolve_command(None) == ["/opt/hmr2/run.sh", "--fast"]

    def test_explicit_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(HMR2_COMMAND_ENV, "env-command")
        assert _resolve_command("explicit-command") == ["explicit-command"]

    def test_unconfigured_returns_none(self) -> None:
        assert _resolve_command(None) is None

    def test_blank_env_is_unconfigured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(HMR2_COMMAND_ENV, "   ")
        assert _resolve_command(None) is None


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


class TestHMR2Result:
    def test_defaults(self, tmp_path: Path) -> None:
        r = HMR2Result(
            success=True,
            output_dir=tmp_path,
            return_code=0,
            used_real_hmr2=False,
        )
        assert r.joints3d_csv is None
        assert r.betas_json is None
        assert r.metadata_json is None
        assert r.stderr_tail == ""
        assert r.extra == {}

    def test_frozen(self, tmp_path: Path) -> None:
        r = HMR2Result(
            success=True,
            output_dir=tmp_path,
            return_code=0,
            used_real_hmr2=False,
        )
        with pytest.raises(
            (AttributeError, TypeError, dataclasses.FrozenInstanceError)
        ):
            r.success = False  # type: ignore[misc]

    def test_error_class_is_runtimeerror(self) -> None:
        assert issubclass(HMR2SidecarError, RuntimeError)


# ---------------------------------------------------------------------------
# run_hmr2_sidecar — stub + subprocess paths
# ---------------------------------------------------------------------------


class TestStubModes:
    def test_dry_run_writes_stubs_and_success(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = run_hmr2_sidecar(
            video_path=tmp_path / "clip.mp4",
            output_dir=out,
            dry_run=True,
        )
        assert result.success is True
        assert result.used_real_hmr2 is False
        assert result.return_code == 0
        assert result.joints3d_csv == out / "joints3d.csv"
        assert result.betas_json == out / "betas.json"
        assert result.metadata_json == out / "metadata.json"
        assert result.joints3d_csv.exists()
        assert result.extra == {"mode": "dry-run"}

    def test_unconfigured_command_writes_stubs(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = run_hmr2_sidecar(video_path=tmp_path / "clip.mp4", output_dir=out)
        assert result.success is True
        assert result.used_real_hmr2 is False
        assert result.joints3d_csv.exists()
        assert result.betas_json.exists()
        assert result.metadata_json.exists()
        assert result.extra == {"mode": "unconfigured"}

    def test_stub_mode_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "out"
        result = run_hmr2_sidecar(
            video_path=tmp_path / "clip.mp4", output_dir=out, dry_run=True
        )
        assert out.is_dir()
        assert result.output_dir == out


class TestSubprocessPaths:
    def _mock_completed(self, returncode: int = 0, stderr: str = "") -> MagicMock:
        proc = MagicMock()
        proc.returncode = returncode
        proc.stderr = stderr
        proc.stdout = ""
        return proc

    def _write_outputs(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "joints3d.csv").write_text(",".join(JOINTS3D_COLUMNS) + "\n")
        (out / "betas.json").write_text("{}")
        (out / "metadata.json").write_text("{}")

    def test_success_when_subprocess_ok_and_files_exist(self, tmp_path: Path) -> None:
        out = tmp_path / "out"

        def fake_run(cmd, **kwargs):
            self._write_outputs(out)
            return self._mock_completed(0, "ok\n")

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = run_hmr2_sidecar(
                video_path=tmp_path / "clip.mp4",
                output_dir=out,
                hmr2_command="python demo.py",
            )
        assert result.success is True
        assert result.used_real_hmr2 is True
        assert result.return_code == 0
        assert result.joints3d_csv == out / "joints3d.csv"
        assert result.betas_json == out / "betas.json"
        assert result.metadata_json == out / "metadata.json"

    def test_env_var_command_is_invoked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "out"
        monkeypatch.setenv(HMR2_COMMAND_ENV, "python demo.py --batch 1")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            self._write_outputs(out)
            return self._mock_completed(0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = run_hmr2_sidecar(video_path=tmp_path / "clip.mp4", output_dir=out)

        assert result.used_real_hmr2 is True
        assert captured["cmd"][:4] == ["python", "demo.py", "--batch", "1"]

    def test_command_includes_video_and_out_folder(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        video = tmp_path / "swing.mp4"
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            self._write_outputs(out)
            return self._mock_completed(0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            run_hmr2_sidecar(
                video_path=video, output_dir=out, hmr2_command=["hmr2-demo"]
            )

        cmd = captured["cmd"]
        assert cmd[0] == "hmr2-demo"
        assert "--video" in cmd
        assert "--out_folder" in cmd
        assert str(video) in cmd
        assert str(out) in cmd

    def test_subprocess_passes_timeout(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            self._write_outputs(out)
            return self._mock_completed(0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            run_hmr2_sidecar(
                video_path=tmp_path / "clip.mp4",
                output_dir=out,
                hmr2_command="hmr2-demo",
                timeout_s=42.0,
            )

        assert captured["timeout"] == 42.0
        assert captured["check"] is False
        assert captured["capture_output"] is True
        assert captured["text"] is True

    def test_filenotfound_falls_back_to_stubs(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        with patch.object(
            subprocess, "run", side_effect=FileNotFoundError("no such tool")
        ):
            result = run_hmr2_sidecar(
                video_path=tmp_path / "clip.mp4",
                output_dir=out,
                hmr2_command="/no/such/hmr2",
            )
        assert result.success is False
        assert result.used_real_hmr2 is False
        assert result.return_code == 127
        assert result.joints3d_csv == out / "joints3d.csv"
        assert result.joints3d_csv.exists()
        assert result.betas_json.exists()
        assert "no such tool" in result.stderr_tail

    def test_timeout_returns_failure(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        exc = subprocess.TimeoutExpired(cmd=["hmr2-demo"], timeout=1.0)
        with patch.object(subprocess, "run", side_effect=exc):
            result = run_hmr2_sidecar(
                video_path=tmp_path / "clip.mp4",
                output_dir=out,
                hmr2_command="hmr2-demo",
                timeout_s=1.0,
            )
        assert result.success is False
        assert result.return_code == -1
        assert result.used_real_hmr2 is True
        assert "timeout" in result.stderr_tail.lower()
        # No stub files were written on timeout
        assert result.joints3d_csv is None
        assert result.betas_json is None
        assert result.metadata_json is None

    def test_nonzero_exit_returns_failure(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 2
        proc.stderr = "CUDA out of memory\n"
        with patch.object(subprocess, "run", return_value=proc):
            result = run_hmr2_sidecar(
                video_path=tmp_path / "clip.mp4",
                output_dir=out,
                hmr2_command="hmr2-demo",
            )
        assert result.success is False
        assert result.used_real_hmr2 is True
        assert result.return_code == 2
        assert result.joints3d_csv is None
        assert "CUDA out of memory" in result.stderr_tail

    def test_exit_zero_but_missing_outputs(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = "warn\n"
        with patch.object(subprocess, "run", return_value=proc):
            result = run_hmr2_sidecar(
                video_path=tmp_path / "clip.mp4",
                output_dir=out,
                hmr2_command="hmr2-demo",
            )
        assert result.success is False
        assert result.return_code == 0
        assert result.used_real_hmr2 is True
        assert result.joints3d_csv is None

    def test_stderr_tail_is_truncated(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        big_stderr = "x" * 10_000
        proc = MagicMock()
        proc.returncode = 5
        proc.stderr = big_stderr
        with patch.object(subprocess, "run", return_value=proc):
            result = run_hmr2_sidecar(
                video_path=tmp_path / "clip.mp4",
                output_dir=out,
                hmr2_command="hmr2-demo",
            )
        assert len(result.stderr_tail) == 4096
        assert result.stderr_tail == big_stderr[-4096:]

    def test_stderr_none_safely_handled(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 3
        proc.stderr = None
        with patch.object(subprocess, "run", return_value=proc):
            result = run_hmr2_sidecar(
                video_path=tmp_path / "clip.mp4",
                output_dir=out,
                hmr2_command="hmr2-demo",
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
                "--video",
                str(tmp_path / "clip.mp4"),
                "--output",
                str(out),
                "--dry-run",
            ]
        )
        assert rc == 0
        assert (out / "joints3d.csv").exists()
        assert (out / "betas.json").exists()
        assert (out / "metadata.json").exists()

    def test_main_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        rc = main(
            [
                "--video",
                str(tmp_path / "clip.mp4"),
                "--output",
                str(out),
                "--dry-run",
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is True
        assert payload["return_code"] == 0
        assert payload["used_real_hmr2"] is False
        assert payload["joints3d_csv"].endswith("joints3d.csv")
        assert payload["betas_json"].endswith("betas.json")
        assert payload["metadata_json"].endswith("metadata.json")
        assert isinstance(payload["output_dir"], str)

    def test_main_failure_returns_nonzero(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        proc = MagicMock()
        proc.returncode = 7
        proc.stderr = "bad\n"
        with patch.object(subprocess, "run", return_value=proc):
            rc = main(
                [
                    "--video",
                    str(tmp_path / "clip.mp4"),
                    "--output",
                    str(out),
                    "--command",
                    "hmr2-demo",
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
                    "--video",
                    str(tmp_path / "clip.mp4"),
                    "--output",
                    str(out),
                    "--command",
                    "hmr2-demo",
                ]
            )
        assert rc >= 1

    def test_main_passes_command_and_timeout(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        called = {}

        def fake_runner(**kwargs):
            called.update(kwargs)
            return HMR2Result(
                success=True,
                output_dir=Path(kwargs["output_dir"]),
                return_code=0,
                used_real_hmr2=True,
            )

        with patch.object(mod, "run_hmr2_sidecar", side_effect=fake_runner):
            rc = main(
                [
                    "--video",
                    str(tmp_path / "clip.mp4"),
                    "--output",
                    str(out),
                    "--command",
                    "/custom/hmr2 --demo",
                    "--timeout",
                    "12.5",
                ]
            )
        assert rc == 0
        assert called["hmr2_command"] == "/custom/hmr2 --demo"
        assert called["timeout_s"] == 12.5
        assert called["dry_run"] is False

    def test_main_requires_video_and_output(self) -> None:
        with pytest.raises(SystemExit):
            main([])
