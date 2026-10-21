"""Contract tests for the vendored Repository_Management CI tier classifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = REPO_ROOT / "scripts/ci/classify_event_tier.py"
PROVENANCE_PATH = REPO_ROOT / "scripts/config/ci_event_classifier_provenance.json"
EXPECTED_PROVIDER_SHA256 = (
    "ad5fbc0ddfd52d6131cb5906f68cc327134db1b35d3305a9f98012c14fa56685"
)
EXPECTED_PROVIDER_COMMIT = "52f1d5267f6bef67fd750657882ea0b0b767a458"


def _load_classifier():
    spec = importlib.util.spec_from_file_location(
        "classify_event_tier", CLASSIFIER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_event_tiers_and_docs_fast_path_contract() -> None:
    classifier = _load_classifier()
    assert classifier.classify("pull_request", ["src/core.py"]) == {
        "tier": "fast",
        "code_changed": "true",
    }
    assert classifier.classify("push", ["src/core.py"]) == {
        "tier": "fast",
        "code_changed": "true",
    }
    for event in ("merge_group", "schedule", "workflow_dispatch", "unknown"):
        assert classifier.classify(event, ["docs/guide.md"]) == {
            "tier": "full",
            "code_changed": "true",
        }
    assert classifier.classify("pull_request", ["docs/guide.md"]) == {
        "tier": "fast",
        "code_changed": "false",
    }


def test_empty_or_unreadable_diff_never_marks_docs_only() -> None:
    classifier = _load_classifier()
    assert classifier.classify("pull_request", []) == {
        "tier": "fast",
        "code_changed": "true",
    }
    assert classifier.classify("pull_request", ["", "  "]) == {
        "tier": "fast",
        "code_changed": "true",
    }
    assert classifier.classify("merge_group", []) == {
        "tier": "full",
        "code_changed": "true",
    }


def test_source_and_workflow_paths_remain_code_changes() -> None:
    classifier = _load_classifier()
    for path in (
        "src/core.py",
        ".github/workflows/ci.yml",
        "src/shared/python/physics/core.py",
    ):
        assert classifier.classify("pull_request", [path])["code_changed"] == "true"


def test_local_bytes_and_provenance_match_pinned_provider_receipt() -> None:
    receipt = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    local_sha256 = hashlib.sha256(CLASSIFIER_PATH.read_bytes()).hexdigest()
    assert receipt == {
        "repository": "D-sorganization/Repository_Management",
        "ref": EXPECTED_PROVIDER_COMMIT,
        "source_path": "scripts/classify_ci_changes.py",
        "sha256": EXPECTED_PROVIDER_SHA256,
    }
    assert local_sha256 == EXPECTED_PROVIDER_SHA256
