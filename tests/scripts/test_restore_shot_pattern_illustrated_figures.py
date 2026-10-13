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


# --- Adversarial hardening (#12304) -----------------------------------------

APPROVED = "phase2/final_figures_v1/details/approved.pdf"
ARTICLE_BYTES = b"reviewed 39-page source"


def _legacy_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive: bytes,
    *,
    content: bytes = b"approved figure",
) -> tuple[Path, Path]:
    """Build a legacy-mode report root with a reviewed article and an archive."""
    report = tmp_path / "report"
    report.mkdir()
    _legacy_selection(report, APPROVED, content)
    article = report / "phase2/presentation_v2/article.tex"
    article.parent.mkdir(parents=True)
    article.write_bytes(ARTICLE_BYTES)
    monkeypatch.setattr(restore, "_report_root", lambda: report)
    monkeypatch.setattr(restore, "ARCHIVE_SHA256", _sha(archive))
    monkeypatch.setattr(
        restore, "_download_archive", lambda dest: dest.write_bytes(archive)
    )
    return report, article


def _forbid_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        restore, "_download_archive", lambda _: pytest.fail("download attempted")
    )


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escaped.pdf",
        "/etc/escaped.pdf",
        "//server/share/escaped.pdf",
        "C:/escaped.pdf",
        "C:escaped.pdf",
        "phase2/final_figures_v1/details/C:escaped.pdf",
        "phase2/final_figures_v1/details/approved.pdf:stream.pdf",
        "phase2/final_figures_v1/details/..\\..\\escaped.pdf",
        "phase2/final_figures_v1/details/../../presentation_v2/article.pdf",
        "phase2/final_figures_v1/details/./approved.pdf",
        "phase2/final_figures_v1/details//approved.pdf",
        "phase2/final_figures_v1/details/sub /approved.pdf",
        "phase2/final_figures_v1/details/sub./approved.pdf",
        "phase2/final_figures_v1/details/approved\x00.pdf",
        "phase2/final_figures_v1/details/approved.pdf.",
        "phase2/final_figures_v1/details/approved.PDF",
        "phase2/final_figures_v1/details/article.tex",
        "phase2/presentation_v2/article.tex",
        "phase2/presentation_v2/summary.pdf",
    ],
)
def test_legacy_selection_rejects_unsafe_paths_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe: str
) -> None:
    report = tmp_path / "report"
    report.mkdir()
    _legacy_selection(report, unsafe, b"escape")
    monkeypatch.setattr(restore, "_report_root", lambda: report)
    _forbid_download(monkeypatch)
    assert restore.main() == 1
    assert not (tmp_path / "escaped.pdf").exists()


def test_legacy_selection_rejects_case_insensitive_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report"
    selection = (
        report / "phase2/validation/illustrated_article_v1/figure_selection.json"
    )
    selection.parent.mkdir(parents=True)
    figures = [
        {"path": APPROVED, "sha256": _sha(b"one")},
        {"path": APPROVED.replace("approved", "Approved"), "sha256": _sha(b"two")},
    ]
    selection.write_text(json.dumps({"figures": figures}), encoding="utf-8")
    monkeypatch.setattr(restore, "_report_root", lambda: report)
    _forbid_download(monkeypatch)
    assert restore.main() == 1


def test_legacy_archive_duplicate_member_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = io.BytesIO()
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(handle, "w") as bundle,
    ):
        bundle.writestr(APPROVED, b"approved figure")
        bundle.writestr(APPROVED, b"evil twin")
    report, article = _legacy_setup(tmp_path, monkeypatch, handle.getvalue())
    assert restore.main() == 1
    assert not (report / APPROVED).exists()
    assert article.read_bytes() == ARTICLE_BYTES


@pytest.mark.parametrize(
    "alias",
    [
        APPROVED.replace("/", "\\"),
        APPROVED.replace("approved", "APPROVED"),
    ],
)
def test_legacy_archive_rejects_aliased_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    """A backslash or case alias of an allowlisted member is ambiguous on
    Windows (zipfile and NTFS fold it onto the same path) and must fail closed
    on every platform rather than restore whichever entry happens to win."""
    archive = _archive_bytes({APPROVED: b"approved figure", alias: b"evil twin"})
    report, article = _legacy_setup(tmp_path, monkeypatch, archive)
    assert restore.main() == 1
    assert not (report / APPROVED).exists()
    assert article.read_bytes() == ARTICLE_BYTES


def test_legacy_archive_backslash_only_member_is_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive_bytes({APPROVED.replace("/", "\\"): b"approved figure"})
    report, _ = _legacy_setup(tmp_path, monkeypatch, archive)
    assert restore.main() == 1
    assert not (report / APPROVED).exists()


def test_legacy_archive_rejects_symlink_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w") as bundle:
        info = zipfile.ZipInfo(APPROVED)
        info.create_system = 3  # Unix, so the mode bits are meaningful
        info.external_attr = 0o120777 << 16  # S_IFLNK
        bundle.writestr(info, b"approved figure")
    report, article = _legacy_setup(tmp_path, monkeypatch, handle.getvalue())
    assert restore.main() == 1
    assert not (report / APPROVED).exists()
    assert article.read_bytes() == ARTICLE_BYTES


def _deflated(name: str, content: bytes) -> bytes:
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(name, content)
    return handle.getvalue()


def test_legacy_archive_rejects_oversized_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bomb = b"\0" * 4096
    archive = _deflated(APPROVED, bomb)
    report, _ = _legacy_setup(tmp_path, monkeypatch, archive, content=bomb)
    monkeypatch.setattr(restore, "MAX_MEMBER_BYTES", 1024)
    assert restore.main() == 1
    assert not (report / APPROVED).exists()


def test_legacy_archive_caps_member_that_lies_about_its_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap must hold while streaming even when the central directory
    under-reports ``file_size`` (a classic zip-bomb trick)."""
    bomb = b"\0" * 4096
    archive = _deflated(APPROVED, bomb)
    report, _ = _legacy_setup(tmp_path, monkeypatch, archive, content=bomb)
    monkeypatch.setattr(restore, "MAX_MEMBER_BYTES", 1024)
    real_infolist = zipfile.ZipFile.infolist

    def lying_infolist(self: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        infos = real_infolist(self)
        for info in infos:
            info.file_size = 10
        return infos

    monkeypatch.setattr(zipfile.ZipFile, "infolist", lying_infolist)
    assert restore.main() == 1
    assert not (report / APPROVED).exists()


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_download_is_size_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr(restore, "MAX_ARCHIVE_BYTES", 1024)
    monkeypatch.setattr(
        restore.urllib.request,
        "urlopen",
        lambda *_a, **_k: _FakeResponse(b"x" * 4096),
    )
    with pytest.raises(ValueError, match="exceeds"):
        restore._download_archive(tmp_path / "archive.zip")


@pytest.mark.parametrize(
    "repo", ["evil.example/../x", "owner/repo/extra", "owner", "own er/repo"]
)
def test_download_rejects_malformed_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: str
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", repo)
    monkeypatch.setattr(
        restore.urllib.request,
        "urlopen",
        lambda *_a, **_k: pytest.fail("network attempted"),
    )
    with pytest.raises(ValueError, match="GITHUB_REPOSITORY"):
        restore._download_archive(tmp_path / "archive.zip")


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:  # Windows without privilege
        pytest.skip(f"symlinks unavailable: {exc}")


def test_legacy_restore_refuses_symlinked_target_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted ``approved.pdf -> article.tex`` link stays inside the report
    root, so a containment check alone would let the restore overwrite TeX."""
    archive = _archive_bytes({APPROVED: b"approved figure"})
    report, article = _legacy_setup(tmp_path, monkeypatch, archive)
    link = report / APPROVED
    link.parent.mkdir(parents=True)
    _symlink_or_skip(link, article)
    assert restore.main() == 1
    assert article.read_bytes() == ARTICLE_BYTES


def test_legacy_restore_refuses_symlinked_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive_bytes({APPROVED: b"approved figure"})
    report, article = _legacy_setup(tmp_path, monkeypatch, archive)
    details = report / "phase2/final_figures_v1/details"
    details.parent.mkdir(parents=True)
    _symlink_or_skip(details, article.parent, directory=True)
    assert restore.main() == 1
    assert not (article.parent / "approved.pdf").exists()
    assert article.read_bytes() == ARTICLE_BYTES
