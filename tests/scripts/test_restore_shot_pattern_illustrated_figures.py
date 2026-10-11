"""Security contract for the CI illustrated-figure restoration script (#12304)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/ci/restore_shot_pattern_illustrated_figures.py"
)
_FIG = "phase2/final_figures_v1/details/a/density.pdf"
_ARTICLE = "phase2/presentation_v2/article.tex"
_SELECTION = "phase2/validation/illustrated_article_v1/figure_selection.json"
_MANIFEST = "phase2/presentation_v2/figure_assets.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def script(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("restore_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["restore_under_test"] = module
    spec.loader.exec_module(module)
    root = tmp_path / "report"
    root.mkdir()
    monkeypatch.setattr(module, "_report_root", lambda: root)
    module.ROOT = root
    return module


def _write(root: Path, rel: str, data: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _select(root: Path, figures: list[tuple[str, bytes]]) -> None:
    payload = {"figures": [{"path": p, "sha256": _sha(d)} for p, d in figures]}
    _write(root, _SELECTION, json.dumps(payload).encode())


def _install_archive(script, monkeypatch, members: dict[str, bytes], symlinks=()):
    calls: list[int] = []

    def fake_download(dest: Path) -> None:
        calls.append(1)
        with zipfile.ZipFile(dest, "w") as bundle:
            for name, data in members.items():
                bundle.writestr(name, data)
            for name in symlinks:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(info, "/etc/passwd")

    monkeypatch.setattr(script, "_download_archive", fake_download)
    monkeypatch.setattr(script, "_verify_archive_digest", lambda path: True)
    return calls


def test_current_manifest_verifies_all_assets_without_legacy_download(
    script, monkeypatch
):
    data = b"figure"
    _write(script.ROOT, _FIG, data)
    _write(script.ROOT, _ARTICLE, b"canonical")
    manifest = {"assets": [{"path": _FIG, "sha256": _sha(data)}]}
    _write(script.ROOT, _MANIFEST, json.dumps(manifest).encode())
    calls = _install_archive(script, monkeypatch, {_ARTICLE: b"obsolete"})
    assert script.main() == 0
    assert calls == []
    assert (script.ROOT / _ARTICLE).read_bytes() == b"canonical"


@pytest.mark.parametrize("defect", ["missing", "mismatch", "traversal", "absolute"])
def test_current_manifest_fails_closed_without_legacy_fallback(
    script, monkeypatch, defect
):
    path = {"traversal": "../escape.pdf", "absolute": "/etc/passwd"}.get(defect, _FIG)
    if defect == "mismatch":
        _write(script.ROOT, _FIG, b"tampered")
    manifest = {"assets": [{"path": path, "sha256": _sha(b"expected")}]}
    _write(script.ROOT, _MANIFEST, json.dumps(manifest).encode())
    _write(script.ROOT, _ARTICLE, b"canonical")
    calls = _install_archive(script, monkeypatch, {_FIG: b"expected"})
    assert script.main() == 1
    assert calls == []
    assert not (script.ROOT / _FIG).exists() or defect == "mismatch"
    assert (script.ROOT / _ARTICLE).read_bytes() == b"canonical"


def test_malformed_manifest_fails_closed(script, monkeypatch):
    _write(script.ROOT, _MANIFEST, b"{not json")
    calls = _install_archive(script, monkeypatch, {})
    assert script.main() == 1
    assert calls == []


def test_legacy_archive_restores_only_allowlisted_figure_and_preserves_source(
    script, monkeypatch, tmp_path
):
    data = b"legacy-figure"
    _select(script.ROOT, [(_FIG, data)])
    _write(script.ROOT, _ARTICLE, b"canonical")
    calls = _install_archive(
        script,
        monkeypatch,
        {_FIG: data, _ARTICLE: b"obsolete", "extra/file.txt": b"x", "../esc.txt": b"x"},
    )
    assert script.main() == 0
    assert calls == [1]
    assert (script.ROOT / _FIG).read_bytes() == data
    assert (script.ROOT / _ARTICLE).read_bytes() == b"canonical"
    assert not (script.ROOT / "extra").exists()
    assert not (tmp_path / "esc.txt").exists()


@pytest.mark.parametrize(
    "bad", ["../escape.pdf", "/abs/x.pdf", "a/../../x.pdf", "a\\..\\x.pdf", "x.tex"]
)
def test_legacy_selection_rejects_traversal_before_download(script, monkeypatch, bad):
    payload = {"figures": [{"path": bad, "sha256": _sha(b"x")}]}
    _write(script.ROOT, _SELECTION, json.dumps(payload).encode())
    calls = _install_archive(script, monkeypatch, {bad: b"x"})
    assert script.main() == 1
    assert calls == []


def test_legacy_hash_mismatch_writes_nothing(script, monkeypatch):
    good, bad = "phase2/figs/a.pdf", "phase2/figs/b.pdf"
    _select(script.ROOT, [(good, b"good"), (bad, b"expected")])
    _install_archive(script, monkeypatch, {good: b"good", bad: b"corrupt"})
    assert script.main() == 1
    assert not (script.ROOT / good).exists()
    assert not (script.ROOT / bad).exists()


def test_legacy_symlink_member_is_rejected(script, monkeypatch):
    _select(script.ROOT, [(_FIG, b"/etc/passwd")])
    _install_archive(script, monkeypatch, {}, symlinks=(_FIG,))
    assert script.main() == 1
    assert not (script.ROOT / _FIG).exists()


def test_legacy_refuses_to_write_through_symlinked_parent(
    script, monkeypatch, tmp_path
):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = script.ROOT / "phase2"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this host")
    _select(script.ROOT, [(_FIG, b"data")])
    _install_archive(script, monkeypatch, {_FIG: b"data"})
    assert script.main() == 1
    assert list(outside.rglob("*.pdf")) == []
