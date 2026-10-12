"""Shared shape code is bound to the same canonical numerical/video producer."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "relative",
    [
        "src/shared/python/body_part_viz/renderers/projective_renderer.py",
        "src/shared/python/body_part_viz/overlay_options.py",
        "src/shared/python/body_part_viz/shapes/_transform.py",
    ],
)
def test_shared_shape_change_invalidates_execution_stamp_with_fixed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    from src.shared.python.workspace import necromatcher_fit_jobs as module

    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"# Reviewed fixture\n")
    monkeypatch.setattr(module, "get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "read_git_commit", lambda root: "a" * 40)
    monkeypatch.setattr(module, "version", lambda package: "fixture")
    before = module.fit_execution_stamp()
    assert relative in before["source_files"]
    path.write_bytes(b"# Changed shared behavior\n")
    after = module.fit_execution_stamp()
    assert after["source_commit"] == before["source_commit"] == "a" * 40
    assert after["runtime_sha256"] == before["runtime_sha256"]
    assert after["source_files"][relative] != before["source_files"][relative]
    assert after["source_sha256"] != before["source_sha256"]
