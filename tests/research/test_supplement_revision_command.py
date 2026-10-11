"""CLI pins are explicit and rendering cannot impersonate independent approval."""

from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path):
    from tests.research.frozen_source_fixture import create_frozen_study_package

    root = tmp_path / "frozen"
    package = create_frozen_study_package(root)
    (package / "matched_supplement_pipeline.py").write_text(
        "def _report_source_hashes():\n    return {}\n"
    )
    (package / "matched_export.py").write_text(
        "def matched_source_snapshot():\n    return {'native':'exact'}\n"
    )
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "pin",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps({"native": "exact"}))
    return root, commit, execution


@pytest.mark.parametrize(
    "mutation",
    ["commit", "dirty_source", "untracked_source", "runtime", "execution_hash"],
)
def test_pin_generation_rejects_unqualified_environment(frozen, tmp_path, mutation):
    from scripts.research.shot_pattern_analysis.supplement_revision_pin import (
        create_parent_pin,
    )

    root, commit, execution = frozen
    digest = sha(execution)
    if mutation == "commit":
        commit = "0" * 40
    elif mutation == "dirty_source":
        (root / "src/tools/shot_pattern_analysis/matched_export.py").write_text(
            "different"
        )
    elif mutation == "untracked_source":
        (root / "src/tools/shot_pattern_analysis/unknown.py").write_text("unknown")
    elif mutation == "runtime":
        execution.write_text('{"native":"changed"}')
        digest = sha(execution)
    else:
        digest = "0" * 64
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        create_parent_pin(
            root, Path(sys.executable), commit, execution, digest, tmp_path / "pin.json"
        )
    assert not (tmp_path / "pin.json").exists()


def test_pin_generation_probes_exact_source_and_runtime(frozen, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_revision_pin import (
        create_parent_pin,
        load_parent_pin,
    )

    root, commit, execution = frozen
    output = tmp_path / "pin.json"
    create_parent_pin(
        root, Path(sys.executable), commit, execution, sha(execution), output
    )
    pin = load_parent_pin(output, sha(output), base=tmp_path / "child")
    assert pin["source_commit"] == commit and pin["execution"] == {"native": "exact"}
    assert (tmp_path / "child" / pin["source_root"]).resolve() == root


def test_cli_has_no_automatic_review_acceptance_flag():
    from scripts.research.shot_pattern_analysis.revise_supplement_figures import parser

    with pytest.raises(SystemExit):
        parser().parse_args(["finalize", "child", "--accept"])


def test_origin_pin_mutation_rejected_after_rebasing(frozen, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_revision_pin import (
        create_parent_pin,
        load_parent_pin,
    )
    from scripts.research.shot_pattern_analysis.supplement_parent_verifier import (
        check_verifier_pin,
    )

    root, commit, execution = frozen
    output = tmp_path / "pin.json"
    child = tmp_path / "child"
    create_parent_pin(
        root, Path(sys.executable), commit, execution, sha(execution), output
    )
    pin = load_parent_pin(output, sha(output), base=child)
    output.write_text("{}")
    with pytest.raises(ValueError, match="origin pin"):
        check_verifier_pin(pin, base=child)
