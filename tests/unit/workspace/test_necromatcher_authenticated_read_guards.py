"""Invalid input, detached identity, cold imports and fingerprint regressions."""

from dataclasses import replace
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from hypothesis_fixture import imported_capture
from src.shared.python.workspace.necromatcher_capture_identity import capture_identity

pytestmark = pytest.mark.unit


def test_identity_copies_caller_containers(fit_case: Any, tmp_path: Path) -> None:
    library, _, _ = fit_case
    asset = imported_capture(library, tmp_path / "source")
    identity = capture_identity(library, asset.dataset_id)
    frames, pngs = list(identity.frames), list(identity.png_sha256)
    detached = replace(identity, frames=frames, png_sha256=pngs)
    frames.clear()
    pngs.clear()
    assert detached == identity


@pytest.mark.parametrize("field", ["capture_id", "capture_hash", "source_clock_sha256"])
def test_identity_rejects_mutable_scalar_identity(
    fit_case: Any, tmp_path: Path, field: str
) -> None:
    library, _, _ = fit_case
    asset = imported_capture(library, tmp_path / "source")
    identity = capture_identity(library, asset.dataset_id)
    with pytest.raises(ValueError):
        replace(identity, **{field: ["mutable"]})


def test_invalid_initial_pixels_do_not_populate_reuse(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    import cv2

    library, _, _ = fit_case
    asset = imported_capture(library, tmp_path / "source")
    original = cv2.imdecode
    monkeypatch.setattr(cv2, "imdecode", lambda *args: None)
    with pytest.raises(ValueError, match="dimensions"):
        with library.authenticated_read():
            capture_identity(library, asset.dataset_id)
    monkeypatch.setattr(cv2, "imdecode", original)
    with library.authenticated_read():
        assert len(capture_identity(library, asset.dataset_id).frames) == 3


def test_reparse_parent_is_rejected_before_reuse(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    library, _, _ = fit_case
    asset = imported_capture(library, tmp_path / "source")
    original = Path.lstat
    flagged = False

    def lstat(path: Path, *args: Any, **kwargs: Any) -> Any:
        info = original(path, *args, **kwargs)
        if flagged and path == library.root:
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
            )
        return info

    with pytest.raises(ValueError, match="reparse"):
        with library.authenticated_read():
            capture_identity(library, asset.dataset_id)
            monkeypatch.setattr(Path, "lstat", lstat)
            flagged = True
            capture_identity(library, asset.dataset_id)


def test_actual_capture_symlink_rejected(fit_case: Any, tmp_path: Path) -> None:
    library, _, _ = fit_case
    asset = imported_capture(library, tmp_path / "source")
    path = Path(asset.path)
    target = tmp_path / "copied.zip"
    target.write_bytes(path.read_bytes())
    link = tmp_path / "probe-link"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Owned symlink fixture unavailable: {error}")
    link.unlink()
    with pytest.raises(ValueError, match="link|reparse"):
        with library.authenticated_read():
            capture_identity(library, asset.dataset_id)
            path.unlink()
            path.symlink_to(target)
            capture_identity(library, asset.dataset_id)


def test_cold_curated_facade_does_not_import_native_sdk() -> None:
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mujoco', 'pydrake', 'pinocchio', 'opensim', 'PyQt6'):
            raise AssertionError('native SDK imported: ' + fullname)
sys.meta_path.insert(0, Block())
from src.shared.python.workspace import AuthenticatedRead, authenticated_read
assert AuthenticatedRead and authenticated_read
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_new_owner_changes_canonical_execution_fingerprint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_fit_jobs as jobs

    owner = tmp_path / "src/shared/python/workspace/necromatcher_authenticated_read.py"
    owner.parent.mkdir(parents=True)
    owner.write_text("# initial", encoding="utf-8")
    monkeypatch.setattr(jobs, "get_repo_root", lambda: tmp_path)
    first = jobs.fit_execution_stamp()
    owner.write_text("# changed", encoding="utf-8")
    second = jobs.fit_execution_stamp()
    assert owner.relative_to(tmp_path).as_posix() in first["source_files"]
    assert first["source_sha256"] != second["source_sha256"]
