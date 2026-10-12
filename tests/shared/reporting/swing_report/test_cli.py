"""Tests for the swing report package CLI (#12327)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from src.shared.python.reporting.swing_report.cli import main
from src.shared.python.reporting.swing_report.exporter import load_package

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "swing_report" / "valid_full.json"
HASH = "a" * 64
TIME = [0.0, 0.01, 0.02]


def _results_doc(visibility: str = "public") -> dict[str, Any]:
    capture: dict[str, Any] = {"capture_id": "cap-1", "visibility": visibility}
    if visibility == "public":
        capture["capture_sha256"] = HASH

    def run(engine: str, values: list[float | None]) -> dict[str, Any]:
        return {
            "engine": engine,
            "run_id": f"run-{engine}",
            "engine_version": "1.0.0",
            "model_file_hash": HASH,
            "run_class": "forward_dynamics",
            "quantities": {
                "clubhead_speed": {
                    "units": "m/s",
                    "frame": "world_Zup",
                    "time_s": TIME,
                    "values": values,
                }
            },
        }

    return {
        "scope": "full",
        "capture": capture,
        "results": [run("mujoco", [1.0, None, 3.0]), run("drake", [1.5, 2.5, 3.5])],
    }


def _write(path: Path, doc: Any) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_export_from_report_then_verify(tmp_path: Path, capsys) -> None:
    out = tmp_path / "pkg"
    assert main(["export", "--report", str(FIXTURE), "--out", str(out)]) == 0
    assert (out / "report.json").exists()
    assert main(["verify", str(out)]) == 0
    assert "OK" in capsys.readouterr().out
    assert load_package(out).report_id == "swing-report-fixture-full"


def test_export_from_results_is_deterministic(tmp_path: Path) -> None:
    src = _write(tmp_path / "results.json", _results_doc())
    for name in ("a", "b"):
        argv = ["export", "--results", str(src), "--out", str(tmp_path / name)]
        assert main([*argv, "--timestamp-utc", "2026-10-11T12:00:00Z"]) == 0
    a = (tmp_path / "a" / "MANIFEST.json").read_bytes()
    assert a == (tmp_path / "b" / "MANIFEST.json").read_bytes()
    report = load_package(tmp_path / "a")
    assert report.generated_utc == "2026-10-11T12:00:00Z"


def test_private_results_export_has_no_capture_hash(tmp_path: Path) -> None:
    src = _write(tmp_path / "results.json", _results_doc("private"))
    out = tmp_path / "pkg"
    assert main(["export", "--results", str(src), "--out", str(out)]) == 0
    for path in out.rglob("*"):
        if path.is_file():
            assert b"capture_sha256" not in path.read_bytes()


def test_private_capture_with_hash_is_rejected(tmp_path: Path, capsys) -> None:
    doc = _results_doc("private")
    doc["capture"]["capture_sha256"] = HASH
    src = _write(tmp_path / "results.json", doc)
    code = main(["export", "--results", str(src), "--out", str(tmp_path / "pkg")])
    assert code == 1
    assert "capture_sha256" in capsys.readouterr().err
    assert not (tmp_path / "pkg").exists()


def test_invalid_report_is_rejected_without_output(tmp_path: Path, capsys) -> None:
    doc = json.loads(FIXTURE.read_text("utf-8"))
    doc["scope"] = "bogus"
    src = _write(tmp_path / "bad.json", doc)
    code = main(["export", "--report", str(src), "--out", str(tmp_path / "pkg")])
    assert code == 1
    assert "violates contract" in capsys.readouterr().err
    assert not (tmp_path / "pkg").exists()


def test_missing_input_is_usage_error(tmp_path: Path, capsys) -> None:
    code = main(["export", "--report", str(tmp_path / "nope.json"), "--out", "x"])
    assert code == 2
    assert capsys.readouterr().err


def test_malformed_results_is_usage_error(tmp_path: Path) -> None:
    src = _write(tmp_path / "results.json", {"scope": "full"})
    assert main(["export", "--results", str(src), "--out", str(tmp_path / "p")]) == 2


def test_verify_reports_tampering(tmp_path: Path, capsys) -> None:
    out = tmp_path / "pkg"
    main(["export", "--report", str(FIXTURE), "--out", str(out)])
    (out / "summary.md").write_text("tampered")
    assert main(["verify", str(out)]) == 1
    assert "sha256" in capsys.readouterr().err


def test_export_requires_exactly_one_source(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["export", "--out", str(tmp_path / "pkg")])
    assert exc.value.code == 2


def test_module_entry_point(tmp_path: Path) -> None:
    out = tmp_path / "pkg"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.shared.python.reporting.swing_report",
            "export",
            "--report",
            str(FIXTURE),
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "MANIFEST.json").exists()
