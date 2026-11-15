"""ML-WS-5 receipt recorder CLI must not recursively delete --output-dir (#15075)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.neural_motion.episodes.mujoco_corpus import (
    CORPUS_MANIFEST_FILENAME,
)
from src.shared.python.neural_motion.episodes.mujoco_corpus_run import (
    CORPUS_RUN_RECEIPT_FILENAME,
)

pytestmark = pytest.mark.unit

_MODULE = "scripts.neural_motion.ml_ws5_record_corpus_receipt"


def _load_script_module():
    return importlib.import_module(_MODULE)


def _seed_prior_tool_output(out_dir: Path, *, unrelated: Path | None = None) -> None:
    mod = _load_script_module()
    mod.write_output_marker(out_dir)
    (out_dir / CORPUS_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    (out_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (out_dir / CORPUS_RUN_RECEIPT_FILENAME).write_text("{}", encoding="utf-8")
    shard_root = out_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    (shard_root / "deadbeef.h5").write_bytes(b"stub")
    if unrelated is not None:
        unrelated.write_text("keep", encoding="utf-8")


def _fake_generation(_request: Any) -> tuple[Any, Any, Any]:
    class _Digest:
        content_digest = "a" * 64

    return _Digest(), _Digest(), _Digest()


def test_refuse_non_empty_output_dir_without_overwrite(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "notes.txt").write_text("unrelated", encoding="utf-8")
    mod = _load_script_module()
    with pytest.raises(SystemExit, match="refusing non-empty output directory"):
        mod.prepare_output_dir(out_dir, overwrite=False)


def test_refuse_overwrite_when_marker_missing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "notes.txt").write_text("unrelated", encoding="utf-8")
    mod = _load_script_module()
    with pytest.raises(SystemExit, match="missing .ml_ws5_record_corpus_receipt.json"):
        mod.prepare_output_dir(out_dir, overwrite=True)


def test_overwrite_clears_prior_tool_output_only(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    unrelated = out_dir / "notes.txt"
    _seed_prior_tool_output(out_dir, unrelated=unrelated)
    mod = _load_script_module()
    mod.prepare_output_dir(out_dir, overwrite=True)
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not (out_dir / mod._OUTPUT_MARKER_FILENAME).exists()
    assert not (out_dir / CORPUS_MANIFEST_FILENAME).exists()
    assert not (out_dir / "shards").exists()


def test_main_overwrite_preserves_unrelated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    unrelated = out_dir / "notes.txt"
    _seed_prior_tool_output(out_dir, unrelated=unrelated)
    mod = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ml_ws5_record_corpus_receipt.py",
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--episode-count",
            "1",
        ],
    )
    monkeypatch.setattr(mod, "run_mujoco_corpus_generation", _fake_generation)
    mod.main()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert (out_dir / mod._OUTPUT_MARKER_FILENAME).is_file()
