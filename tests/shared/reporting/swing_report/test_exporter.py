"""Tests for the swing report package exporter (#12327, epic #11419, #10382)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.data_io.provenance import ProvenanceInfo
from src.shared.python.reporting.swing_report.adapters_base import (
    EngineResult,
    QuantityData,
)
from src.shared.python.reporting.swing_report.builder import build_swing_report
from src.shared.python.reporting.swing_report.contract import (
    Capture,
    SwingReport,
    report_from_dict,
)
from src.shared.python.reporting.swing_report.exporter import (
    MANIFEST_NAME,
    PackageError,
    export_package,
    load_package,
    verify_package,
)
from src.shared.python.reporting.swing_report.validator import (
    SwingReportContractError,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

HASH = "a" * 64
TIME = (0.0, 0.01, 0.02, 0.03)
SENTINEL = 987.654321  # value of every withheld quantity an engine "supplied"
PUBLIC = Capture(capture_id="cap-1", visibility="public", capture_sha256=HASH)
PRIVATE = Capture(capture_id="cap-2", visibility="private")
INFO = ProvenanceInfo(
    timestamp_utc="2026-10-11T12:00:00Z",
    timestamp_local="2026-10-11T08:00:00-04:00",
    git_commit_sha=None,
)


def _data(values: tuple[float | None, ...], channel: str | None = None) -> QuantityData:
    return QuantityData(
        units="m/s",
        frame="world_Zup",
        time_s=TIME,
        values=values,
        channel=channel,
    )


def _result(engine: str, values: tuple[float | None, ...]) -> EngineResult:
    leaked = QuantityData("N", "world_Zup", TIME, (SENTINEL,) * len(TIME))
    return EngineResult(
        engine=engine,
        run_id=f"run-{engine}",
        engine_version="1.0.0",
        model_file_hash=HASH,
        run_class="forward_dynamics",
        quantities={
            "clubhead_speed": _data(values),
            "joint_load": leaked,
            "ground_reaction_force": leaked,
        },
    )


def _report(
    capture: Capture = PUBLIC,
    mujoco: tuple[float | None, ...] = (10.0, 20.0, None, 40.0),
) -> SwingReport:
    results = [
        _result("mujoco", mujoco),
        _result("drake", (10.5, 19.5, 30.0, 41.0)),
    ]
    return build_swing_report(results, capture, "full", provenance=INFO)


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in _files(root)}


def test_package_layout(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    names = {
        p.relative_to(tmp_path / "pkg").as_posix() for p in _files(tmp_path / "pkg")
    }
    assert {"report.json", MANIFEST_NAME, "summary.md"} <= names
    assert "series/clubhead_speed__mujoco.csv" in names
    assert "series/clubhead_speed__drake.csv" in names
    assert "plots/clubhead_speed_plot.png" in names


def test_round_trip_equals_source(tmp_path: Path) -> None:
    source = _report()
    export_package(source, tmp_path / "pkg")
    loaded = load_package(tmp_path / "pkg")
    assert loaded == source
    assert loaded.to_dict() == source.to_dict()


def test_round_trip_private_capture(tmp_path: Path) -> None:
    source = _report(PRIVATE)
    export_package(source, tmp_path / "pkg")
    assert load_package(tmp_path / "pkg") == source


def test_report_json_is_the_contract_document(tmp_path: Path) -> None:
    source = _report()
    export_package(source, tmp_path / "pkg")
    doc = json.loads((tmp_path / "pkg" / "report.json").read_text(encoding="utf-8"))
    assert doc == source.to_dict()
    assert report_from_dict(doc) == source


def test_deterministic_bytes(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "a")
    export_package(_report(), tmp_path / "b")
    assert _tree_bytes(tmp_path / "a") == _tree_bytes(tmp_path / "b")


def test_missing_samples_are_empty_cells_never_zero(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    text = (tmp_path / "pkg/series/clubhead_speed__mujoco.csv").read_text("utf-8")
    rows = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert rows[0].startswith("time_s,value")
    missing = [r for r in rows[1:] if r.startswith("0.02,")]
    assert len(missing) == 1
    cells = missing[0].split(",")
    assert cells[1] == ""
    assert "0.0" not in cells[1:2]
    assert (tmp_path / "pkg/series/clubhead_speed__mujoco.csv").read_bytes().count(
        b"\r"
    ) == 0


def test_csv_carries_provenance_header(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    text = (tmp_path / "pkg/series/clubhead_speed__drake.csv").read_text("utf-8")
    head = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert any("Generated: 2026-10-11T12:00:00Z" in ln for ln in head)
    assert any("Run ID: run-drake" in ln for ln in head)
    assert any("Engine: drake" in ln for ln in head)


def test_withheld_values_appear_in_no_file(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    needles = [str(SENTINEL).encode(), b"987.65"]
    for path in _files(tmp_path / "pkg"):
        data = path.read_bytes()
        assert not any(n in data for n in needles), path
    names = [p.name for p in _files(tmp_path / "pkg")]
    assert not any("joint_load" in n or "ground_reaction" in n for n in names)


def test_withheld_listed_with_reason_only(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    summary = (tmp_path / "pkg" / "summary.md").read_text("utf-8")
    assert "## Withheld Quantities" in summary
    assert "Joint Load" in summary
    assert "not validated" in summary
    assert "# Swing Report" in summary


def test_plots_only_for_value_renders(tmp_path: Path) -> None:
    source = _report()
    export_package(source, tmp_path / "pkg")
    expected = {f"{p.plot_id}.png" for p in source.plots if p.render == "values"}
    present = {p.name for p in (tmp_path / "pkg" / "plots").glob("*.png")}
    assert present == expected
    assert all(
        (tmp_path / "pkg/plots" / n).read_bytes().startswith(b"\x89PNG")
        for n in present
    )


def test_manifest_hashes_verify(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    manifest = json.loads((tmp_path / "pkg" / MANIFEST_NAME).read_text("utf-8"))
    listed = {f["path"]: f["sha256"] for f in manifest["files"]}
    assert MANIFEST_NAME not in listed
    for rel, digest in listed.items():
        actual = hashlib.sha256((tmp_path / "pkg" / rel).read_bytes()).hexdigest()
        assert actual == digest
    assert set(listed) == {
        p.relative_to(tmp_path / "pkg").as_posix()
        for p in _files(tmp_path / "pkg")
        if p.name != MANIFEST_NAME
    }
    verify_package(tmp_path / "pkg")


def test_private_package_has_no_capture_sha256(tmp_path: Path) -> None:
    export_package(_report(PRIVATE), tmp_path / "pkg")
    for path in _files(tmp_path / "pkg"):
        assert b"capture_sha256" not in path.read_bytes(), path


@pytest.mark.parametrize("victim", ["report.json", "summary.md"])
def test_tampering_is_detected(tmp_path: Path, victim: str) -> None:
    export_package(_report(), tmp_path / "pkg")
    target = tmp_path / "pkg" / victim
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(PackageError, match="sha256"):
        verify_package(tmp_path / "pkg")
    with pytest.raises(PackageError):
        load_package(tmp_path / "pkg")


def test_unlisted_and_missing_files_are_detected(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    (tmp_path / "pkg" / "extra.txt").write_text("x")
    with pytest.raises(PackageError, match="unlisted"):
        verify_package(tmp_path / "pkg")
    (tmp_path / "pkg" / "extra.txt").unlink()
    (tmp_path / "pkg/series/clubhead_speed__drake.csv").unlink()
    with pytest.raises(PackageError, match="missing"):
        verify_package(tmp_path / "pkg")


def _rewrite_manifest(root: Path) -> None:
    """Re-sign a tampered package so only the semantic checks can catch it."""
    manifest = json.loads((root / MANIFEST_NAME).read_text("utf-8"))
    for entry in manifest["files"]:
        entry["sha256"] = hashlib.sha256(
            (root / entry["path"]).read_bytes()
        ).hexdigest()
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_resigned_report_violating_contract_fails_closed(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    path = tmp_path / "pkg" / "report.json"
    doc: dict[str, Any] = json.loads(path.read_text("utf-8"))
    for quantity in doc["quantities"]:
        if quantity["quantity_id"] == "joint_load":
            quantity["values"] = [SENTINEL]
    path.write_text(json.dumps(doc), encoding="utf-8")
    _rewrite_manifest(tmp_path / "pkg")
    with pytest.raises(SwingReportContractError):
        load_package(tmp_path / "pkg")


def test_resigned_csv_disagreeing_with_report_is_rejected(tmp_path: Path) -> None:
    export_package(_report(), tmp_path / "pkg")
    csv_path = tmp_path / "pkg/series/clubhead_speed__mujoco.csv"
    csv_path.write_text(csv_path.read_text("utf-8").replace("20.0", "21.0"), "utf-8")
    _rewrite_manifest(tmp_path / "pkg")
    with pytest.raises(PackageError, match="differs from report"):
        load_package(tmp_path / "pkg")


def test_export_refuses_non_empty_directory(tmp_path: Path) -> None:
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(PackageError, match="not empty"):
        export_package(_report(), out)
    assert (out / "keep.txt").read_text() == "x"


def test_export_accepts_existing_empty_directory(tmp_path: Path) -> None:
    out = tmp_path / "pkg"
    out.mkdir()
    export_package(_report(), out)
    assert (out / "report.json").exists()


def test_export_rejects_contract_violating_report(tmp_path: Path) -> None:
    good = _report()
    bad = report_from_dict(good.to_dict())
    object.__setattr__(bad, "scope", "bogus")
    with pytest.raises(SwingReportContractError):
        export_package(bad, tmp_path / "pkg")
    assert not (tmp_path / "pkg").exists()


def test_unavailable_only_report_has_no_series_or_plots(tmp_path: Path) -> None:
    lone = EngineResult(
        engine="mujoco",
        run_id="run-mujoco",
        engine_version="1.0.0",
        model_file_hash=HASH,
        run_class="forward_dynamics",
        quantities={"clubhead_speed": _data((1.0, 2.0, 3.0, 4.0))},
    )
    report = build_swing_report([lone], PUBLIC, "full", provenance=INFO)
    export_package(report, tmp_path / "pkg")
    assert not list((tmp_path / "pkg").glob("series/*"))
    assert not list((tmp_path / "pkg").glob("plots/*"))
    assert load_package(tmp_path / "pkg") == report


def test_awkward_channel_names_get_safe_unique_filenames(tmp_path: Path) -> None:
    doc = _report().to_dict()
    quantity = next(
        q for q in doc["quantities"] if q["quantity_id"] == "clubhead_speed"
    )
    for series, channel in zip(quantity["series"], ("a/b c", "a_b_c"), strict=True):
        series["channel"] = channel
    # same engine twice is not allowed to collide: distinct engines, tricky channels
    report = report_from_dict(doc)
    export_package(report, tmp_path / "pkg")
    names = [p.name for p in (tmp_path / "pkg" / "series").glob("*.csv")]
    assert len(names) == 2
    assert all("/" not in n and " " not in n for n in names)
    assert load_package(tmp_path / "pkg") == report
