"""Historical verifier execution must be bound to published source bytes."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_REVIEW = _ROOT / "docs/research/shot_pattern_analysis/phase2/review_followups"
_BASELINE = "4fc22c694404217208534b78308d31ea8985e7d3"
_HISTORICAL_CANDIDATE = "ecc82796cec7e9119e2dd2fb0e8bb2582851761f"


def _plot_verifier() -> ModuleType:
    script = _REVIEW / "check_plot_source_equivalence.py"
    spec = importlib.util.spec_from_file_location("plot_source_verifier", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_course_verifier_rejects_tampered_git_bytes_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "executed"
    monkeypatch.chdir(_ROOT)

    def tampered_git(command: list[str], **kwargs: object) -> bytes:
        if command[:2] == ["git", "show"]:
            return f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode()
        raise AssertionError(f"Unexpected subprocess command: {command}")

    monkeypatch.setattr(subprocess, "check_output", tampered_git)
    with pytest.raises(ValueError, match="historical source digest"):
        runpy.run_path(str(_REVIEW / "check_course_scoring_refactor.py"))
    assert not marker.exists()


@pytest.mark.unit
def test_plot_loader_rejects_unapproved_bytes_before_execution(
    tmp_path: Path,
) -> None:
    module = _plot_verifier()
    marker = tmp_path / "executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode()
    with pytest.raises(ValueError, match="density source digest"):
        module._module("_density_original", source)
    assert not marker.exists()


@pytest.mark.unit
def test_plot_verifier_reproduces_exact_historical_comparisons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "historical.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_plot_source_equivalence.py",
            "--baseline",
            _BASELINE,
            "--candidate",
            _HISTORICAL_CANDIDATE,
            "--output",
            str(output),
        ],
    )
    _plot_verifier().main()
    receipt = json.loads(output.read_text())
    assert receipt["density_comparisons_exact"] == 84
    assert receipt["candidate_commit"] == _HISTORICAL_CANDIDATE
    assert receipt["pass"] is True


@pytest.mark.unit
def test_plot_verifier_rejects_stale_current_density(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "current.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_plot_source_equivalence.py",
            "--baseline",
            _BASELINE,
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Unexpected semantic source change"):
        _plot_verifier().main()
    assert not output.exists()


@pytest.mark.unit
def test_plot_verifier_rejects_unknown_candidate_before_git_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_git(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("Git was called before candidate authentication")

    monkeypatch.setattr(subprocess, "check_output", unexpected_git)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_plot_source_equivalence.py",
            "--baseline",
            _BASELINE,
            "--candidate",
            "HEAD",
            "--output",
            str(tmp_path / "unapproved.json"),
        ],
    )
    with pytest.raises(ValueError, match="Unapproved historical candidate"):
        _plot_verifier().main()


@pytest.mark.unit
def test_qualified_historical_script_is_lossless_compressed_evidence() -> None:
    bundle = _REVIEW / "qualified_source_2e0a74f8ca"
    manifest = json.loads((bundle / "BUNDLE_MANIFEST.json").read_text())
    entry = next(
        item
        for item in manifest["copied_evidence"]
        if item["original_relative_path"].endswith("/check_plot_source_equivalence.py")
    )
    assert entry["encoding"] == "gzip-lossless"
    assert entry["stored_file"] == "check_plot_source_equivalence.py.gz"
    stored = (bundle / entry["stored_file"]).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == entry["stored_sha256"]
    original = gzip.decompress(stored)
    assert (
        hashlib.sha256(original).hexdigest()
        == entry["original_sha256"]
        == ("dfcf5bb441911dcb854f84a360ff5f3c78348f56aad2d9c00b51c5973c730c62")
    )
    assert gzip.compress(original, mtime=0) == stored
    assert not (bundle / "check_plot_source_equivalence.py").exists()
