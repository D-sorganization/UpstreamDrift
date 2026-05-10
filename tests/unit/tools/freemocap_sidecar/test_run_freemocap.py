"""Unit tests for the freemocap sidecar.

The freemocap library is AGPL and not a dependency of UpstreamDrift, so
these tests never import it. Coverage is via:

- ``dry_run=True`` for the happy path (writes stub artifacts).
- A non-existent interpreter path for the FileNotFoundError branch.
- A monkey-patched ``subprocess.run`` for the rest of the matrix.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.freemocap_sidecar import (
    FreeMoCapResult,
    run_freemocap_sidecar,
)
from src.tools.freemocap_sidecar.run_freemocap import main

# ---------------------------------------------------------------------------
# Dry-run / stub path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dry_run_writes_stub_artifacts(tmp_path: Path) -> None:
    """Dry-run produces a successful result with stub landmarks + metadata."""
    out = tmp_path / "out"
    result = run_freemocap_sidecar(
        input_dir=tmp_path / "in",
        output_dir=out,
        dry_run=True,
    )
    assert result.success is True
    assert result.used_real_freemocap is False
    assert result.return_code == 0
    assert result.landmarks_csv is not None and result.landmarks_csv.exists()
    assert result.metadata_json is not None and result.metadata_json.exists()

    csv_text = result.landmarks_csv.read_text(encoding="utf-8")
    assert csv_text.startswith("frame,landmark_id,x,y,z\n")

    meta = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert meta["stub"] is True
    assert meta["freemocap_version"] == "stub"


@pytest.mark.unit
def test_dry_run_creates_output_dir_if_missing(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "out"
    assert not out.exists()
    result = run_freemocap_sidecar(
        input_dir=tmp_path / "in", output_dir=out, dry_run=True
    )
    assert result.success
    assert out.exists()


# ---------------------------------------------------------------------------
# Real-subprocess paths (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_interpreter_falls_back_to_stub(tmp_path: Path) -> None:
    """A non-existent freemocap_env_python path falls back gracefully."""
    out = tmp_path / "out"
    result = run_freemocap_sidecar(
        input_dir=tmp_path / "in",
        output_dir=out,
        freemocap_env_python="/no/such/python/interpreter_xyz",
    )
    # success=False because the user asked for real, but stub output exists.
    assert result.success is False
    assert result.used_real_freemocap is False
    assert result.return_code == 127
    assert result.landmarks_csv is not None and result.landmarks_csv.exists()


@pytest.mark.unit
def test_missing_freemocap_module_falls_back_to_stub(tmp_path: Path) -> None:
    """If subprocess exits non-zero with `No module named 'freemocap'`, stub."""
    out = tmp_path / "out"
    fake_proc = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="",
        stderr="ModuleNotFoundError: No module named 'freemocap'\n",
    )
    with patch("subprocess.run", return_value=fake_proc):
        result = run_freemocap_sidecar(
            input_dir=tmp_path / "in",
            output_dir=out,
            freemocap_env_python="/usr/bin/python3",  # exists-shaped path
        )
    assert result.success is False
    assert result.used_real_freemocap is False
    assert result.landmarks_csv is not None and result.landmarks_csv.exists()
    assert "freemocap" in result.stderr_tail


@pytest.mark.unit
def test_real_subprocess_failure_returns_failure_no_stub(tmp_path: Path) -> None:
    """A genuine non-zero exit (not a missing-module) is reported as failure."""
    out = tmp_path / "out"
    fake_proc = subprocess.CompletedProcess(
        args=[],
        returncode=2,
        stdout="",
        stderr="some other freemocap-internal error",
    )
    with patch("subprocess.run", return_value=fake_proc):
        result = run_freemocap_sidecar(
            input_dir=tmp_path / "in",
            output_dir=out,
            freemocap_env_python="/usr/bin/python3",
        )
    assert result.success is False
    assert result.used_real_freemocap is True
    assert result.return_code == 2
    assert "freemocap-internal" in result.stderr_tail


@pytest.mark.unit
def test_real_subprocess_success_with_outputs(tmp_path: Path) -> None:
    """If subprocess exits 0 AND outputs exist, success=True."""
    out = tmp_path / "out"
    out.mkdir()
    # Pre-write the expected output files (simulates freemocap doing it)
    (out / "landmarks.csv").write_text(
        "frame,landmark_id,x,y,z\n0,0,1,2,3\n", encoding="utf-8"
    )
    (out / "metadata.json").write_text(
        json.dumps({"freemocap_version": "1.6.0", "n_frames": 100}),
        encoding="utf-8",
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_proc):
        result = run_freemocap_sidecar(
            input_dir=tmp_path / "in",
            output_dir=out,
            freemocap_env_python="/usr/bin/python3",
        )
    assert result.success is True
    assert result.used_real_freemocap is True
    assert result.return_code == 0


@pytest.mark.unit
def test_real_subprocess_success_but_missing_outputs(tmp_path: Path) -> None:
    """Exit 0 but no output files = failure (broken freemocap install)."""
    out = tmp_path / "out"
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_proc):
        result = run_freemocap_sidecar(
            input_dir=tmp_path / "in",
            output_dir=out,
            freemocap_env_python="/usr/bin/python3",
        )
    assert result.success is False
    assert result.return_code == 0
    assert result.used_real_freemocap is True


@pytest.mark.unit
def test_subprocess_timeout_reports_failure(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="freemocap", timeout=0.1),
    ):
        result = run_freemocap_sidecar(
            input_dir=tmp_path / "in",
            output_dir=out,
            freemocap_env_python="/usr/bin/python3",
            timeout_s=0.1,
        )
    assert result.success is False
    assert result.return_code == -1
    assert "timeout" in result.stderr_tail.lower()


# ---------------------------------------------------------------------------
# Result dataclass invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_result_is_frozen() -> None:
    r = FreeMoCapResult(
        success=True, output_dir=Path("/tmp"), return_code=0, used_real_freemocap=False
    )
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError is a subclass
        r.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_dry_run_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["success"] is True
    assert payload["used_real_freemocap"] is False


@pytest.mark.unit
def test_cli_real_failure_returns_nonzero(tmp_path: Path) -> None:
    rc = main(
        [
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(tmp_path / "out"),
            "--env-python",
            "/no/such/python/interpreter_xyz",
        ]
    )
    assert rc != 0
