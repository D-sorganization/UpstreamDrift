"""Historical citations require exact row and independently pinned file evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import check_spec_paths as checker

pytestmark = pytest.mark.unit


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def history(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    provider = root / "vendor/ud-tools"
    provider.mkdir(parents=True)
    for repo in (root, provider):
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "spec-test@example.invalid")
        _git(repo, "config", "user.name", "Spec Test")
    old = "src/shared/python/signal_toolkit/fitting.py"
    current = "src/shared/python/signal_toolkit/fitting.py"
    payload = b"original = True\n"
    source = root / old
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    _git(root, "add", old)
    _git(root, "commit", "-qm", "Historical implementation")
    original = _git(root, "rev-parse", "HEAD")
    source.unlink()
    target = provider / current
    target.parent.mkdir(parents=True)
    target.write_bytes(b"provider = True\n")
    _git(provider, "add", current)
    _git(provider, "commit", "-qm", "Published provider")
    pin = _git(provider, "rev-parse", "HEAD")
    _git(root, "add", "-u")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{pin},vendor/ud-tools")
    _git(root, "commit", "-qm", "Retire child copy")
    row = f"| 2026-05-21 | n/a | Historical `{old}` optimization. |"
    record = {
        "row_sha256": hashlib.sha256(row.encode()).hexdigest(),
        "historical_commit": original,
        "historical_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_commit": pin,
        "provider_path": current,
        "provider_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    return root, old, row, {old: record}, target


def _resolve(history, *, row=None, in_history=True, path=None):
    root, old, original_row, records, _ = history
    return checker._historical_path_is_verified(
        root, path or old, row or original_row, in_history, records
    )


def test_exact_historical_row_resolves_to_independently_pinned_provider(history):
    assert _resolve(history)


def test_current_prose_cannot_borrow_historical_resolution(history):
    assert not _resolve(history, in_history=False)


def test_modified_or_new_historical_row_is_rejected(history):
    assert not _resolve(history, row=history[2] + " Added claim.")


def test_unknown_missing_path_still_fails(history):
    assert not _resolve(history, path="src/missing.py")


@pytest.mark.parametrize(
    "field,value",
    [
        ("historical_commit", "0" * 40),
        ("historical_commit", "HEAD"),
        ("historical_sha256", "0" * 64),
        ("provider_commit", "0" * 40),
        ("provider_sha256", "0" * 64),
        ("provider_path", "../../outside.py"),
    ],
)
def test_wrong_or_missing_provenance_fails_closed(history, field, value):
    history[3][history[1]][field] = value
    assert not _resolve(history)


def test_missing_provider_file_is_rejected(history):
    history[4].unlink()
    assert not _resolve(history)


def test_modified_provider_worktree_is_rejected(history):
    history[4].write_text("different = True\n")
    assert not _resolve(history)


@pytest.mark.parametrize(
    "template,expected",
    [
        ("## 12. Change Log\n\n{row}\n", 0),
        ("## Current Contract\n\n{row}\n", 1),
        ("## 12. Change Log\n\n## Current Contract\n\n{row}\n", 1),
        ("{row}\n\n## 12. Change Log\n", 1),
    ],
)
def test_cli_enforces_actual_change_log_section(
    history, monkeypatch, template, expected
):
    root, _, row, records, _ = history
    spec = root / "SPEC.md"
    spec.write_text(template.format(row=row))
    registry = root / "history.json"
    registry.write_text(json.dumps({"paths": records}))
    monkeypatch.setattr(checker, "ROOT", root)
    monkeypatch.setattr(checker, "SPEC", spec)
    monkeypatch.setattr(checker, "HISTORICAL_PATHS", registry)
    assert checker.main() == expected


def test_check_doc_paths_flags_missing_bare_dirs_and_non_py_files(tmp_path: Path):
    """Curated doc path check catches missing bare dirs and non-Python files."""
    existing_py = tmp_path / "src/shared/python/contracts.py"
    existing_py.parent.mkdir(parents=True)
    existing_py.write_text("# ok\n", encoding="utf-8")
    existing_dir = tmp_path / "tests/unit/api"
    existing_dir.mkdir(parents=True)

    doc = tmp_path / "docs/testing/guide.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "\n".join(
            [
                "Valid `src/shared/python/contracts.py` and `tests/unit/api`.",
                "Missing dir `tests/e2e`.",
                "Missing workflow `.github/workflows/mesh-backends-test.yml`.",
                "Missing asset `tests/assets/pro_driver.c3d`.",
                "Missing py `src/shared/python/missing.py`.",
            ]
        ),
        encoding="utf-8",
    )

    bad = checker.check_doc_paths(tmp_path, [doc])
    assert bad == [
        "  docs/testing/guide.md:2: missing path 'tests/e2e'",
        "  docs/testing/guide.md:3: missing path '.github/workflows/mesh-backends-test.yml'",
        "  docs/testing/guide.md:4: missing path 'tests/assets/pro_driver.c3d'",
        "  docs/testing/guide.md:5: missing path 'src/shared/python/missing.py'",
    ]


def test_check_doc_paths_ignores_fenced_code_blocks_and_module_constants(
    tmp_path: Path,
):
    """Fenced code snippets and module constant qualifiers are ignored."""
    doc = tmp_path / "docs/README.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "\n".join(
            [
                "Constant `src/shared/python/engine_availability.PINOCCHIO_AVAILABLE`.",
                "```bash",
                "pytest tests/nonexistent_in_fence/",
                "```",
                "Inline after fence `src/missing_after_fence.py`.",
            ]
        ),
        encoding="utf-8",
    )

    bad = checker.check_doc_paths(tmp_path, [doc])
    assert bad == [
        "  docs/README.md:5: missing path 'src/missing_after_fence.py'",
    ]


def test_cli_fails_when_curated_docs_cite_missing_repo_path(
    history, monkeypatch: pytest.MonkeyPatch
):
    """CLI main() exits non-zero when a curated doc references a missing repo path."""
    root, _, row, records, _ = history
    spec = root / "SPEC.md"
    spec.write_text(f"## 12. Change Log\n\n{row}\n", encoding="utf-8")
    registry = root / "history.json"
    registry.write_text(json.dumps({"paths": records}), encoding="utf-8")
    readme = root / "docs/README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("Broken `src/missing_module.py`.\n", encoding="utf-8")

    monkeypatch.setattr(checker, "ROOT", root)
    monkeypatch.setattr(checker, "SPEC", spec)
    monkeypatch.setattr(checker, "HISTORICAL_PATHS", registry)
    assert checker.main() == 1


def test_repository_curated_docs_have_no_broken_repo_paths():
    """All curated testing, architecture, and manual docs resolve on disk."""
    assert checker.check_doc_paths(checker.ROOT) == []
