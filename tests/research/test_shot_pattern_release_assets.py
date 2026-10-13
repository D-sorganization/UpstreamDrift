"""Binary study artefacts live in the research release, not in git history."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.research.shot_pattern_analysis import fetch_release_assets as module

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs/research/shot_pattern_analysis/phase2/RELEASE_ASSETS.json"
BINARY_SUFFIXES = (".zip", ".jpg", ".jpeg")


def _tracked(pattern: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", pattern],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split()


def test_index_is_well_formed_and_names_are_unique():
    index = module.load_index(INDEX)
    assets = [entry.asset for entry in index]
    assert len(assets) == len(set(assets))
    assert all(
        entry.path.startswith("docs/research/shot_pattern_analysis/") for entry in index
    )


def test_indexed_binaries_are_not_committed():
    tracked = set(_tracked("docs/research/shot_pattern_analysis"))
    indexed = {entry.path for entry in module.load_index(INDEX)}
    assert not tracked & indexed


def test_no_new_binary_archives_under_shot_pattern_docs():
    offenders = [
        path
        for path in _tracked("docs/research/shot_pattern_analysis")
        if path.lower().endswith(BINARY_SUFFIXES)
    ]
    assert offenders == [], "publish binaries to the research release instead"


def _write_index(tmp_path, payload: bytes, name="a.zip"):
    entry = {
        "asset": "phase2__" + name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    index = tmp_path / "RELEASE_ASSETS.json"
    index.write_text(
        json.dumps(
            {
                "schema": "shot-pattern-release-assets/1",
                "release_tag": "tag",
                "files": {f"docs/research/shot_pattern_analysis/phase2/{name}": entry},
            }
        )
    )
    return index


def test_fetch_verifies_digest_before_writing(tmp_path):
    payload = b"archive bytes"
    index = _write_index(tmp_path, payload)
    written = module.fetch(
        index, tmp_path / "repo", download=lambda tag, asset: payload
    )
    target = tmp_path / "repo/docs/research/shot_pattern_analysis/phase2/a.zip"
    assert written == [target]
    assert target.read_bytes() == payload


def test_fetch_rejects_tampered_download_and_writes_nothing(tmp_path):
    index = _write_index(tmp_path, b"expected")
    with pytest.raises(ValueError, match="sha256"):
        module.fetch(index, tmp_path / "repo", download=lambda tag, asset: b"evil!!!!")
    assert not (tmp_path / "repo").exists()


def test_fetch_skips_files_already_present_and_verified(tmp_path):
    payload = b"present"
    index = _write_index(tmp_path, payload)
    target = tmp_path / "repo/docs/research/shot_pattern_analysis/phase2/a.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    def never(tag, asset):
        raise AssertionError("verified files must not be downloaded again")

    assert module.fetch(index, tmp_path / "repo", download=never) == []


@pytest.mark.parametrize(
    "path",
    [
        "../escape.zip",
        "docs/research/shot_pattern_analysis/../../x.zip",
        "/abs.zip",
        "docs\\research\\x.zip",
    ],
)
def test_index_rejects_unsafe_paths(tmp_path, path):
    index = tmp_path / "RELEASE_ASSETS.json"
    index.write_text(
        json.dumps(
            {
                "schema": "shot-pattern-release-assets/1",
                "release_tag": "tag",
                "files": {path: {"asset": "a", "bytes": 1, "sha256": "0" * 64}},
            }
        )
    )
    with pytest.raises(ValueError):
        module.load_index(index)


def test_index_rejects_wrong_schema(tmp_path):
    index = tmp_path / "RELEASE_ASSETS.json"
    index.write_text(json.dumps({"schema": "other", "release_tag": "t", "files": {}}))
    with pytest.raises(ValueError, match="schema"):
        module.load_index(index)
