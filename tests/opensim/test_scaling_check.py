"""Contract tests for the OpenSim scaling-check command (MATCH #14444, #13991 metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import opensim_scaling_check as cli
from src.engines.physics_engines.opensim.python.tour_matching import scaling_check
from src.engines.physics_engines.opensim.python.tour_matching.scale import (
    estimate_segment_scales,
)
from src.engines.physics_engines.opensim.python.tour_matching.scaling_check import (
    ACCEPTANCE_REFERENCE,
    ScalingCheckMetrics,
    run_scaling_check,
    write_scaling_check,
)
from src.sim_run.manifest import validate_run_manifest
from tests.opensim.test_segment_scale import _synthetic_capture

pytestmark = pytest.mark.unit

CAPTURE_SHA = "a" * 64
UD_COMMIT = "1" * 40
CREATED_AT = "2026-11-04T04:00:00Z"


@pytest.fixture(scope="module")
def nominal() -> dict[str, float]:
    """Nominal lengths equal to the unit-scale synthetic capture, so factors equal the scale."""
    return estimate_segment_scales(_synthetic_capture()).measured_lengths_m


def _check(scale: float, nominal: dict[str, float]) -> ScalingCheckMetrics:
    return run_scaling_check(
        _synthetic_capture(scale=scale),
        capture_sha256=CAPTURE_SHA,
        nominal_lengths_m=nominal,
    )


def test_plausible_scales_pass(nominal: dict[str, float]) -> None:
    metrics = _check(1.05, nominal)
    assert metrics.verdict == "pass"
    assert metrics.failed_segments == []
    assert metrics.acceptance_reference == ACCEPTANCE_REFERENCE == "#13991"
    assert set(metrics.segments) == set(nominal)
    for segment in metrics.segments.values():
        assert segment.plausible
        assert segment.scale_factor == pytest.approx(1.05)


def test_os9_humerus_scale_fails(nominal: dict[str, float]) -> None:
    """The known OS-9 flaw (humerus factor 1.46) must fail the length plausibility gate."""
    metrics = _check(1.46, nominal)
    assert metrics.verdict == "fail"
    assert "humerus_r" in metrics.failed_segments
    assert "scale_factor" in metrics.segments["humerus_r"].reasons[0]


def test_non_rigid_segment_fails(nominal: dict[str, float]) -> None:
    capture = _synthetic_capture(noise_std=0.02)
    metrics = run_scaling_check(
        capture, capture_sha256=CAPTURE_SHA, nominal_lengths_m=nominal
    )
    assert metrics.verdict == "fail"
    assert any(
        "rigid_residual" in reason
        for segment in metrics.segments.values()
        for reason in segment.reasons
    )


def test_rejects_bad_capture_digest(nominal: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="capture_sha256"):
        run_scaling_check(
            _synthetic_capture(),
            capture_sha256="not-a-digest",
            nominal_lengths_m=nominal,
        )


def test_metrics_schema_is_strict(nominal: dict[str, float]) -> None:
    payload = _check(1.0, nominal).model_dump(mode="json")
    assert ScalingCheckMetrics.model_validate(payload).verdict == "pass"
    payload["unexpected"] = True
    with pytest.raises(ValueError):
        ScalingCheckMetrics.model_validate(payload)
    schema = ScalingCheckMetrics.model_json_schema()
    assert set(schema["required"]) >= {
        "schema_version",
        "acceptance_reference",
        "capture_sha256",
        "bounds",
        "segments",
        "verdict",
        "failed_segments",
    }


def test_write_emits_metrics_and_valid_run_manifest(
    tmp_path: Path, nominal: dict[str, float]
) -> None:
    metrics = _check(1.0, nominal)
    metrics_path, manifest_path = write_scaling_check(
        metrics, tmp_path, ud_commit=UD_COMMIT, created_at=CREATED_AT
    )
    written = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert ScalingCheckMetrics.model_validate(written) == metrics
    manifest = validate_run_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    assert manifest.input_data.sha256 == CAPTURE_SHA
    assert manifest.ud_commit == UD_COMMIT
    assert manifest.model.revision.startswith("sha256:")
    assert manifest.parameters["acceptance_reference"] == "#13991"
    assert manifest.parameters["metrics_sha256"] == scaling_check.sha256_file(
        metrics_path
    )


def test_cli_writes_outputs_and_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nominal: dict[str, float]
) -> None:
    c3d = tmp_path / "capture.c3d"
    c3d.write_bytes(b"synthetic")
    monkeypatch.setattr(
        cli, "load_capture", lambda path: _synthetic_capture(scale=1.46)
    )
    monkeypatch.setattr(cli, "DEFAULT_NOMINAL_LENGTHS_M", nominal)
    out = tmp_path / "out"
    code = cli.main([str(c3d), "--out", str(out), "--ud-commit", UD_COMMIT])
    assert code == 1
    metrics = json.loads((out / scaling_check.METRICS_NAME).read_text("utf-8"))
    assert metrics["capture_sha256"] == scaling_check.sha256_file(c3d)
    assert metrics["verdict"] == "fail"
    validate_run_manifest(
        json.loads((out / scaling_check.MANIFEST_NAME).read_text("utf-8"))
    )


def test_cli_appends_dated_trend_record_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nominal: dict[str, float]
) -> None:
    """#14446: a scaling-check run passed --trend appends one dated record."""
    c3d = tmp_path / "capture.c3d"
    c3d.write_bytes(b"synthetic")
    monkeypatch.setattr(cli, "load_capture", lambda path: _synthetic_capture())
    monkeypatch.setattr(cli, "DEFAULT_NOMINAL_LENGTHS_M", nominal)
    trend = tmp_path / "trend.jsonl"
    args = [str(c3d), "--ud-commit", UD_COMMIT, "--trend", str(trend)]
    assert cli.main([*args, "--out", str(tmp_path / "a"), "--date", "2026-11-05"]) == 0
    assert cli.main([*args, "--out", str(tmp_path / "b"), "--date", "2026-11-06"]) == 0
    dates = [json.loads(line)["date"] for line in trend.read_text("utf-8").splitlines()]
    assert dates == ["2026-11-05", "2026-11-06"]


def test_cli_leaves_trend_alone_without_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nominal: dict[str, float]
) -> None:
    c3d = tmp_path / "capture.c3d"
    c3d.write_bytes(b"synthetic")
    monkeypatch.setattr(cli, "load_capture", lambda path: _synthetic_capture())
    monkeypatch.setattr(cli, "DEFAULT_NOMINAL_LENGTHS_M", nominal)
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "trend_main", calls.append)
    out = tmp_path / "out"
    assert cli.main([str(c3d), "--out", str(out), "--ud-commit", UD_COMMIT]) == 0
    assert calls == []
