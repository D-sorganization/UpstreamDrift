"""Tests for the optional model-submodule hint in scripts/ci/verify_installation.py (#9415)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import verify_installation as mod

pytestmark = pytest.mark.unit


def test_model_submodules_match_gitmodules() -> None:
    """Every declared model submodule path must be a real .gitmodules entry."""
    gitmodules = (Path(mod._REPO_ROOT) / ".gitmodules").read_text(encoding="utf-8")
    declared = {
        line.split("=", 1)[1].strip()
        for line in gitmodules.splitlines()
        if line.strip().startswith("path =")
    }
    for rel_path, _label in mod.MODEL_SUBMODULES:
        assert rel_path in declared, rel_path
    assert "vendor/ud-tools" not in {p for p, _ in mod.MODEL_SUBMODULES}


def test_check_model_submodules_reports_hint_when_missing(tmp_path: Path) -> None:
    results = mod.check_model_submodules(tmp_path)
    assert [rel for rel, _ok, _msg in results] == [
        rel for rel, _label in mod.MODEL_SUBMODULES
    ]
    for rel_path, present, message in results:
        assert not present
        assert f"git submodule update --init {rel_path}" in message
        assert "optional" in message


def test_check_model_submodules_detects_materialised_checkout(
    tmp_path: Path,
) -> None:
    first, _label = mod.MODEL_SUBMODULES[0]
    target = tmp_path / first
    target.mkdir(parents=True)
    (target / "README.md").write_text("model\n", encoding="utf-8")
    # An empty directory (gitlink not initialised) still counts as missing.
    second, _label = mod.MODEL_SUBMODULES[1]
    (tmp_path / second).mkdir(parents=True)

    results = {
        rel: present for rel, present, _msg in mod.check_model_submodules(tmp_path)
    }

    assert results[first] is True
    assert results[second] is False


def test_blob_copies_of_model_submodules_are_gone() -> None:
    """The plain-file copies removed in #9415 must not reappear under src/."""
    root = Path(mod._REPO_ROOT)
    for stale in (
        "src/shared/models/opensim/opensim-models",
        "src/shared/models/myosuite/myo_sim",
    ):
        path = root / stale
        assert not path.is_dir() or not any(path.iterdir()), stale
