"""Refitting is required after the reconstructed input changes."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.motion_capture.provenance import input_record
from src.motion_capture.rig.documents import write_document
from src.tools.capture_rig.result_evidence import model_revision_problem
from tests.tools.capture_rig.test_wizard_evidence import _capture

pytestmark = pytest.mark.unit


def test_model_fit_keeps_its_original_reconstruction_revision(tmp_path: Path) -> None:
    root, _, media = _capture(tmp_path)
    source = root / "reconstruct" / "session_reconstruction.json"
    source.parent.mkdir()
    write_document(source, {"camera_source_sha256": "a" * 64})
    report = {
        "provenance": {
            "inputs": [input_record(source, root)],
            "parameters": {"source": {"kind": "triangulate"}},
        }
    }
    media = replace(media, model_fit=report)
    assert model_revision_problem(media) is None
    write_document(source, {"camera_source_sha256": "b" * 64})
    problem = model_revision_problem(media)
    assert problem is not None and "fit again" in problem.lower()
    assert media.model_fit == report


def test_legacy_model_without_input_lineage_requires_review(tmp_path: Path) -> None:
    _, _, media = _capture(tmp_path)
    media = replace(media, model_fit={})
    problem = model_revision_problem(media)
    assert problem is not None and "unverified" in problem.lower()
