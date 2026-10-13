"""CI figure restoration never overwrites article source or trusts extra ZIP members."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.ci import restore_shot_pattern_illustrated_figures as restore

pytestmark = pytest.mark.unit

CURRENT_PDFS = (
    "summary.pdf",
    "driver_baseline_density.pdf",
    "driver_baseline_flight_paths.pdf",
    "seven_iron_baseline_density.pdf",
    "seven_iron_baseline_flight_paths.pdf",
    "pitching_wedge_baseline_density.pdf",
    "pitching_wedge_baseline_flight_paths.pdf",
    "driver_shaft_rotation_sd_density.pdf",
    "driver_curve_strength_density.pdf",
    "seven_iron_shaft_rotation_sd_density.pdf",
    "seven_iron_curve_strength_density.pdf",
    "pitching_wedge_shaft_rotation_sd_density.pdf",
    "pitching_wedge_curve_strength_density.pdf",
    "driver_across_lean_flight_paths.pdf",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _current_manifest(report: Path) -> Path:
    assets = report / "phase2/presentation_v2"
    assets.mkdir(parents=True)
    hashes = {}
    for name in CURRENT_PDFS:
        content = f"current:{name}".encode()
        (assets / name).write_bytes(content)
        hashes[f"phase2/presentation_v2/{name}"] = _sha(content)
    manifest = assets / "figure_assets.json"
    manifest.write_text(json.dumps({"files_sha256": hashes}), encoding="utf-8")
    return manifest


def _legacy_selection(report: Path, path: str, content: bytes) -> None:
    selection = (
        report / "phase2/validation/illustrated_article_v1/figure_selection.json"
    )
    selection.parent.mkdir(parents=True)
    selection.write_text(
        json.dumps({"figures": [{"path": path, "sha256": _sha(content)}]}),
        encoding="utf-8",
    )


def _archive_bytes(members: dict[str, bytes]) -> bytes:
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return handle.getvalue()


def test_current_manifest_verifies_all_assets_without_legacy_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report"
    manifest = _current_manifest(report)
    article = report / "phase2/presentation_v2/article.tex"
    article.write_bytes(b"reviewed 39-page source")
    _legacy_selection(report, "phase2/final_figures_v1/details/old.pdf", b"old")
    monkeypatch.setattr(restore, "_report_root", lambda: report)
    monkeypatch.setattr(
        restore, "_download_archive", lambda _: pytest.fail("legacy download attempted")
    )
    assert restore.main() == 0
    assert article.read_bytes() == b"reviewed 39-page source"
    assert manifest.is_file()


@pytest.mark.parametrize("defect", ["missing", "mismatch", "traversal"])
def test_current_manifest_fails_closed_without_legacy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    report = tmp_path / "report"
    manifest = _current_manifest(report)
    article = report / "phase2/presentation_v2/article.tex"
    article.write_bytes(b"reviewed 39-page source")
    asset = report / "phase2/presentation_v2/summary.pdf"
    if defect == "missing":
        asset.unlink()
    elif defect == "mismatch":
        asset.write_bytes(b"tampered")
    else:
        data = json.loads(manifest.read_text())
        data["files_sha256"]["phase2/presentation_v2/../article.tex"] = _sha(
            article.read_bytes()
        )
        manifest.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(restore, "_report_root", lambda: report)
    monkeypatch.setattr(
        restore, "_download_archive", lambda _: pytest.fail("legacy download attempted")
    )
    assert restore.main() == 1
    assert article.read_bytes() == b"reviewed 39-page source"


def test_legacy_archive_restores_only_allowlisted_figure_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report"
    report.mkdir()
    path = "phase2/final_figures_v1/details/approved.pdf"
    _legacy_selection(report, path, b"approved figure")
    article = report / "phase2/presentation_v2/article.tex"
    article.parent.mkdir(parents=True)
    article.write_bytes(b"reviewed 39-page source")
    canonical = report / "phase2/validation/illustrated_article_v1/canonical.tex"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"reviewed canonical source")
    archive = _archive_bytes(
        {
            path: b"approved figure",
            "phase2/presentation_v2/article.tex": b"obsolete 33-page source",
            "phase2/validation/illustrated_article_v1/canonical.tex": b"obsolete canonical",
            "../escaped.tex": b"escape",
            "phase2/final_figures_v1/details/extra.pdf": b"extra",
        }
    )
    monkeypatch.setattr(restore, "_report_root", lambda: report)
    monkeypatch.setattr(restore, "ARCHIVE_SHA256", _sha(archive))
    monkeypatch.setattr(
        restore, "_download_archive", lambda dest: dest.write_bytes(archive)
    )
    assert restore.main() == 0
    assert (report / path).read_bytes() == b"approved figure"
    assert article.read_bytes() == b"reviewed 39-page source"
    assert canonical.read_bytes() == b"reviewed canonical source"
    assert not (tmp_path / "escaped.tex").exists()
    assert not (report / "phase2/final_figures_v1/details/extra.pdf").exists()


def test_legacy_selection_rejects_traversal_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report"
    report.mkdir()
    _legacy_selection(report, "../escaped.pdf", b"escape")
    monkeypatch.setattr(restore, "_report_root", lambda: report)
    monkeypatch.setattr(
        restore, "_download_archive", lambda _: pytest.fail("download attempted")
    )
    assert restore.main() == 1
    assert not (tmp_path / "escaped.pdf").exists()


def test_legacy_hash_mismatch_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report"
    report.mkdir()
    path = "phase2/final_figures_v1/details/approved.pdf"
    _legacy_selection(report, path, b"approved figure")
    archive = _archive_bytes({path: b"different bytes"})
    monkeypatch.setattr(restore, "_report_root", lambda: report)
    monkeypatch.setattr(restore, "ARCHIVE_SHA256", _sha(archive))
    monkeypatch.setattr(
        restore, "_download_archive", lambda dest: dest.write_bytes(archive)
    )
    assert restore.main() == 1
    assert not (report / path).exists()
