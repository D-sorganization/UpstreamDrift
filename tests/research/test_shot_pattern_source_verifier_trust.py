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

_PLOT_SOURCE_FOLDER = _ROOT / "tests/research/fixtures/shot_pattern_verifier_sources"
_PLOT_SOURCE_RECORDS = {
    "4fc22c694404217208534b78308d31ea8985e7d3:src/tools/shot_pattern_analysis/density.py": {
        "filename": "baseline-density.py.gz",
        "sha256": "127d1d249203efbddb3152e668e270dc54be5857e97377a344edc9d543c2d3ce",
        "stored_sha256": "75101a1fbdd75acf11e8a08e60f33e854500b446a754c47025c75e369197bfd8",
    },
    "4fc22c694404217208534b78308d31ea8985e7d3:src/tools/shot_pattern_analysis/academic_plots.py": {
        "filename": "baseline-academic_plots.py.gz",
        "sha256": "5822f2c2dd701582908f0157f98c093d1164c5ebe68b865e24bd94fe796f889c",
        "stored_sha256": "b9968925a4e8f4fb12d9376848233acce935f027b59b09913f77752b250fb00f",
    },
    "4fc22c694404217208534b78308d31ea8985e7d3:src/tools/shot_pattern_analysis/matched_report_plots.py": {
        "filename": "baseline-matched_report_plots.py.gz",
        "sha256": "99753f1624d3b33716a4bebaf44517281f3ac662cc1fca12176abbc75752a377",
        "stored_sha256": "9a81830d01cd51aa39f7ab3b8196802690514cf05ea0bc33ee3f9fcce9f97317",
    },
    "ecc82796cec7e9119e2dd2fb0e8bb2582851761f:src/tools/shot_pattern_analysis/density.py": {
        "filename": "candidate-density.py.gz",
        "sha256": "d444779f6ba6179e0fd87348c9b60f901a853e4558afa4cae3e732ff675640ad",
        "stored_sha256": "e89c2386d4d424034bc90813b5ed30cd4847fcb26121180397f6798905d2582b",
    },
    "ecc82796cec7e9119e2dd2fb0e8bb2582851761f:src/tools/shot_pattern_analysis/academic_plots.py": {
        "filename": "candidate-academic_plots.py.gz",
        "sha256": "246b891faaa6e3512ecb9685d6cc6a84e2bea3ccadd1f0030a7afa5671978944",
        "stored_sha256": "0fc5fc8c56402350351231dc0173625a7afd7452ec78cd4e6b42902854196c99",
    },
    "ecc82796cec7e9119e2dd2fb0e8bb2582851761f:src/tools/shot_pattern_analysis/matched_report_plots.py": {
        "filename": "candidate-matched_report_plots.py.gz",
        "sha256": "72515aef38a1c72b740f0c70a770a4c7db1124a62b4514bb424af7124d6dd8fc",
        "stored_sha256": "2cb122cb44db9ac56c9f7022eb94b2cc0686a0c7db91cba8de547fbc3c0529f3",
    },
}


def _archived_plot_git(command: list[str], *, cwd: Path) -> bytes:
    """Transport exact historical source for unit tests in shallow checkouts."""
    if (
        len(command) != 3
        or command[:2] != ["git", "show"]
        or Path(cwd).resolve() != _ROOT.resolve()
        or command[2] not in _PLOT_SOURCE_RECORDS
    ):
        raise ValueError("Unapproved archived Git request")
    record = _PLOT_SOURCE_RECORDS[command[2]]
    stored = (_PLOT_SOURCE_FOLDER / record["filename"]).read_bytes()
    if hashlib.sha256(stored).hexdigest() != record["stored_sha256"]:
        raise ValueError("Historical fixture stored digest differs")
    source = gzip.decompress(stored)
    if hashlib.sha256(source).hexdigest() != record["sha256"]:
        raise ValueError("Historical fixture source digest differs")
    return source


@pytest.fixture
def historical_plot_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", _archived_plot_git)


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, historical_plot_git: None
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, historical_plot_git: None
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


@pytest.mark.unit
@pytest.mark.parametrize("change", ["commit", "path", "cwd", "command"])
def test_archived_git_transport_rejects_unapproved_requests(change: str) -> None:
    command = ["git", "show", f"{_BASELINE}:src/tools/shot_pattern_analysis/density.py"]
    cwd = _ROOT
    if change == "commit":
        command[2] = command[2].replace(_BASELINE, "HEAD")
    elif change == "path":
        command[2] = command[2].replace("density.py", "unapproved.py")
    elif change == "cwd":
        cwd = _ROOT.parent
    else:
        command.append("--extra")
    with pytest.raises(ValueError, match="Unapproved archived Git request"):
        _archived_plot_git(command, cwd=cwd)


@pytest.mark.unit
@pytest.mark.parametrize("layer", ["stored", "source"])
def test_archived_git_transport_rejects_corrupted_source_before_returning(
    monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    key = f"{_BASELINE}:src/tools/shot_pattern_analysis/density.py"
    record = dict(_PLOT_SOURCE_RECORDS[key])
    tampered = gzip.compress(b"raise AssertionError('must never execute')\n", mtime=0)
    if layer == "source":
        record["stored_sha256"] = hashlib.sha256(tampered).hexdigest()
    monkeypatch.setitem(_PLOT_SOURCE_RECORDS, key, record)
    original_read = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        return tampered if path.name == record["filename"] else original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(ValueError, match=f"Historical fixture {layer} digest"):
        _archived_plot_git(["git", "show", key], cwd=_ROOT)
