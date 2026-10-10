"""A density-only publication cannot change numerical or reused evidence."""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_reviewed_revision(parent, child, **kwargs):
    """Synthetic independent-review fixture; never used by production code."""
    from scripts.research.shot_pattern_analysis import (
        supplement_publication as publication,
    )

    publication.create_density_revision(parent, child, **kwargs)
    candidate_path = child / publication.CANDIDATE_RECEIPT
    candidate = json.loads(candidate_path.read_text())
    evidence = child.parent / (child.name + "_synthetic_review")
    evidence.mkdir()
    (evidence / "visual.txt").write_text(
        "Synthetic fixture: no actual visual acceptance"
    )
    review = {
        "schema": "matched-supplement-publication-review/1",
        "accepted": True,
        "review_scope": "visual_and_contract",
        "reviewer": "synthetic test reviewer",
        "candidate_sha256": sha(candidate_path),
        "source_delta": publication.approved_source_delta(
            kwargs["parent_verifier"](parent), candidate
        ),
        "revised_asset_sha256": {
            n: r["child_sha256"]
            for n, r in candidate["origins"].items()
            if r["operation"] == "rerendered"
        },
        "evidence_files": {"visual.txt": sha(evidence / "visual.txt")},
    }
    path = evidence / "review.json"
    path.write_text(json.dumps(review))
    return publication.finalize_density_revision(
        child,
        review_path=path,
        review_sha256=sha(path),
        parent_verifier=kwargs["parent_verifier"],
    )


@pytest.fixture
def parent(tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_publication import ARTIFACTS

    root = tmp_path / "parent"
    root.mkdir()
    figures = root / "figures"
    figures.mkdir()
    records = []
    for name in ["density.png", "density.pdf", "density.svg", "overhead.png"]:
        path = figures / name
        path.write_text(name)
        records.append(
            {
                "relative_path": name,
                "sha256": sha(path),
                "presentation": "academic",
                "figure_kind": name.replace(".", "_"),
                "source_kind": "six_population_common_index_comparison",
                "comparison_kind": "across_lean",
                "first_cell": "a",
                "second_cell": "b",
                "first_protocol": "one",
                "second_protocol": "two",
                "target_x_m": 100.0,
                "common_supported_n": 9,
                "total_n_per_population": 10,
            }
        )
    ledger = {"schema": "matched-supplement-figure-ledger/1", "figures": records}
    for key, name in ARTIFACTS.items():
        (root / name).write_text(json.dumps(ledger if key == "figure_ledger" else {}))
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        _report_source_hashes,
    )

    receipt = {
        "postprocessing_source_sha256": _report_source_hashes(),
        "schema": "matched-supplement-pipeline-receipt/1",
        "stage": "final",
        "artifact_sha256": {k: sha(root / n) for k, n in ARTIFACTS.items()},
    }
    (root / "REPORT_SUPPLEMENT_PIPELINE.json").write_text(json.dumps(receipt))
    loaded = {
        "report": {},
        "comparisons": {},
        "figure_ledger": ledger,
        "receipt": receipt,
    }
    return root, loaded


def renderer(records, output):
    result = []
    for record in records:
        path = output / "figures" / record["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new " + record["relative_path"])
        result.append({**record, "sha256": sha(path)})
    return result


def test_child_preserves_parent_and_reuses_exact_bytes(parent, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_publication import (
        create_density_revision,
        verify_density_revision,
    )

    root, data = parent
    child = tmp_path / "child"

    def verifier(path):
        return copy.deepcopy(data)

    create_reviewed_revision(
        root, child, parent_verifier=verifier, render_density=renderer
    )
    result = verify_density_revision(child, parent_verifier=verifier)
    assert result["publication_identity"]["kind"] == "academic_pair_density_row_gutter"
    assert (root / "figures/overhead.png").read_bytes() == (
        child / "figures/overhead.png"
    ).read_bytes()
    assert len(result["figure_ledger"]["figures"]) == 4
    assert result["receipt"] == data["receipt"]


@pytest.mark.parametrize(
    "mutation", ["numeric", "reused", "extra", "identity", "duplicate", "source"]
)
def test_tampering_rejected(parent, tmp_path, mutation):
    from scripts.research.shot_pattern_analysis.supplement_publication import (
        create_density_revision,
        verify_density_revision,
    )

    root, data = parent
    child = tmp_path / "child"

    def verifier(path):
        return copy.deepcopy(data)

    create_reviewed_revision(
        root, child, parent_verifier=verifier, render_density=renderer
    )
    receipt_path = child / "REPORT_SUPPLEMENT_PUBLICATION_REVISION.json"
    receipt = json.loads(receipt_path.read_text())
    if mutation == "numeric":
        (child / "REPORT_SUPPLEMENT.json").write_text('{"changed":1}')
    elif mutation == "reused":
        (child / "figures/overhead.png").write_text("changed")
    elif mutation == "extra":
        (child / "figures/extra.png").write_text("extra")
    elif mutation == "source":
        receipt["revision_source_sha256"]["supplement_publication.py"] = "0" * 64
    else:
        ledger_path = child / "REPORT_SUPPLEMENT_FIGURES.json"
        ledger = json.loads(ledger_path.read_text())
        if mutation == "identity":
            ledger["figures"][0]["target_x_m"] = 101
        else:
            ledger["figures"].append(ledger["figures"][0])
        ledger_path.write_text(json.dumps(ledger))
        receipt["child_artifact_sha256"]["figure_ledger"] = sha(ledger_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_density_revision(child, parent_verifier=verifier)


def test_parent_mutation_during_render_rejected_without_receipt(parent, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_publication import (
        create_density_revision,
    )

    root, data = parent
    child = tmp_path / "child"

    def mutate(records, output):
        out = renderer(records, output)
        (root / "REPORT_SUPPLEMENT.json").write_text("changed")
        return out

    with pytest.raises(ValueError):
        create_reviewed_revision(
            root,
            child,
            parent_verifier=lambda p: copy.deepcopy(data),
            render_density=mutate,
        )
    assert not (child / "REPORT_SUPPLEMENT_PUBLICATION_REVISION.json").exists()


def test_wrong_renderer_scope_rejected(parent, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_publication import (
        create_density_revision,
    )

    root, data = parent

    def wrong(records, output):
        out = renderer(records, output)
        out[0]["first_protocol"] = "wrong"
        return out

    with pytest.raises(ValueError):
        create_reviewed_revision(
            root,
            tmp_path / "child",
            parent_verifier=lambda p: copy.deepcopy(data),
            render_density=wrong,
        )


def test_ordinary_dispatch_retains_public_parent_verifier(parent, monkeypatch):
    from scripts.research.shot_pattern_analysis.supplement_publication import (
        load_verified_supplement_publication,
    )
    from src.tools.shot_pattern_analysis import matched_supplement_pipeline as pipeline

    root, data = parent
    monkeypatch.setattr(
        pipeline, "load_verified_supplement_artifacts", lambda p: copy.deepcopy(data)
    )
    out = load_verified_supplement_publication(root)
    assert out["publication_identity"]["kind"] == "pipeline"
    monkeypatch.setattr(
        pipeline,
        "load_verified_supplement_artifacts",
        lambda p: (_ for _ in ()).throw(ValueError("source changed")),
    )
    with pytest.raises(ValueError, match="source changed"):
        load_verified_supplement_publication(root)


def test_parent_executable_pin_rejects_tamper(tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_parent_verifier import (
        check_verifier_pin,
    )

    executable = tmp_path / "python"
    executable.write_text("not python")
    with pytest.raises(ValueError, match="executable"):
        check_verifier_pin(
            {
                "source_root": ".",
                "python_executable": str(executable),
                "python_sha256": "0" * 64,
            },
            base=tmp_path,
        )


@pytest.fixture
def verifier_pin(tmp_path):
    import subprocess
    import sys

    root = tmp_path / "frozen"
    package = root / "src/tools/shot_pattern_analysis"
    package.mkdir(parents=True)
    (package / "matched_supplement_pipeline.py").write_text(
        "def load_verified_supplement_artifacts(path):\n    return {'verified': True}\n"
    )
    (package / "matched_export.py").write_text(
        "def matched_source_snapshot():\n    return {'native': 'pinned'}\n"
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
            "fixture",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return root, {
        "source_root": str(root),
        "source_commit": commit,
        "python_executable": sys.executable,
        "python_sha256": sha(Path(sys.executable)),
        "source_sha256": {
            str(p.relative_to(root)): sha(p) for p in package.glob("*.py")
        },
        "execution": {"native": "pinned"},
    }


@pytest.mark.parametrize("mutation", ["runtime", "source", "commit", "omitted_source"])
def test_frozen_subprocess_rejects_drift(verifier_pin, tmp_path, mutation):
    import subprocess
    from scripts.research.shot_pattern_analysis.supplement_parent_verifier import (
        verify_parent_in_frozen_source,
    )

    root, pin = verifier_pin
    if mutation == "runtime":
        pin["execution"]["native"] = "changed"
    elif mutation == "source":
        (root / "src/tools/shot_pattern_analysis/matched_export.py").write_text(
            "changed"
        )
    elif mutation == "commit":
        pin["source_commit"] = "0" * 40
    else:
        pin["source_sha256"].pop("src/tools/shot_pattern_analysis/matched_export.py")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        verify_parent_in_frozen_source(tmp_path, pin, base=tmp_path)


def test_frozen_subprocess_origin_and_runtime(verifier_pin, tmp_path, monkeypatch):
    from scripts.research.shot_pattern_analysis.supplement_parent_verifier import (
        verify_parent_in_frozen_source,
    )

    root, pin = verifier_pin
    monkeypatch.setenv("PYTHONPATH", "/untrusted/import/root")
    assert verify_parent_in_frozen_source(tmp_path, pin, base=tmp_path) == {
        "verified": True
    }


def test_child_mutation_during_second_parent_validation_rejected(parent, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_publication import (
        create_density_revision,
        verify_density_revision,
    )

    root, data = parent
    child = tmp_path / "child"
    create_reviewed_revision(
        root,
        child,
        parent_verifier=lambda p: copy.deepcopy(data),
        render_density=renderer,
    )
    calls = 0

    def verifier(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            (child / "figures/density.png").write_text("changed while validating")
        return copy.deepcopy(data)

    with pytest.raises(ValueError):
        verify_density_revision(child, parent_verifier=verifier)


def test_report_load_uses_child_receipt_identity(parent, tmp_path, monkeypatch):
    from scripts.research.shot_pattern_analysis.supplement_publication import (
        create_density_revision,
    )
    from scripts.research.shot_pattern_analysis import (
        supplement_parent_verifier,
        build_final_report,
    )

    root, data = parent
    child = tmp_path / "child"
    create_reviewed_revision(
        root,
        child,
        parent_verifier=lambda p: copy.deepcopy(data),
        render_density=renderer,
        verifier_binding={"test": "binding"},
    )
    monkeypatch.setattr(
        supplement_parent_verifier,
        "verify_parent_in_frozen_source",
        lambda *a, **k: copy.deepcopy(data),
    )
    loaded, digest = build_final_report.load_supplement_at_stable_receipt(child)
    assert digest == sha(child / "REPORT_SUPPLEMENT_PUBLICATION_REVISION.json")
    assert loaded["publication_identity"]["parent_receipt_sha256"] == sha(
        root / "REPORT_SUPPLEMENT_PIPELINE.json"
    )
    assert not (child / "REPORT_SUPPLEMENT_PIPELINE.json").exists()


def test_final_receipt_retains_parent_and_child_identity(tmp_path):
    from scripts.research.shot_pattern_analysis.build_final_report import (
        final_publication_receipt,
    )

    paths = tuple(tmp_path / name for name in ("report.md", "cells.csv", "lean.csv"))
    for path in paths:
        path.write_text("output")
    identity = {
        "kind": "academic_pair_density_row_gutter",
        "receipt_filename": "REPORT_SUPPLEMENT_PUBLICATION_REVISION.json",
        "receipt_sha256": "child",
        "parent_receipt_sha256": "parent",
    }
    receipt = final_publication_receipt(
        output_paths=paths,
        source_sha256={},
        primary_plan_sha256="a",
        control_plan_sha256="b",
        figure_manifest_sha256={},
        supplement_receipt_sha256="child",
        supplement_publication_identity=identity,
    )
    assert receipt["supplement_publication_identity"] == identity


def test_coherently_rehashed_protected_source_is_still_rejected(
    parent, tmp_path, monkeypatch
):
    from scripts.research.shot_pattern_analysis import (
        supplement_publication as publication,
    )

    root, data = parent
    child = tmp_path / "child"
    create_reviewed_revision(
        root,
        child,
        parent_verifier=lambda p: copy.deepcopy(data),
        render_density=renderer,
    )
    path = child / publication.REVISION_RECEIPT
    receipt = json.loads(path.read_text())
    altered = {**receipt["revision_source_sha256"], "density.py": "0" * 64}
    receipt["revision_source_sha256"] = altered
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(publication, "revision_source_hashes", lambda: altered)
    with pytest.raises(ValueError, match="protected reporting source"):
        publication.verify_density_revision(
            child, parent_verifier=lambda p: copy.deepcopy(data)
        )


def test_rendering_alone_cannot_publish_an_accepted_child(parent, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_publication import (
        create_density_revision,
        REVISION_RECEIPT,
    )

    root, data = parent
    child = tmp_path / "candidate_only"
    path = create_density_revision(
        root,
        child,
        parent_verifier=lambda p: copy.deepcopy(data),
        render_density=renderer,
    )
    assert path.name == "REPORT_SUPPLEMENT_PUBLICATION_CANDIDATE.json"
    assert not (child / REVISION_RECEIPT).exists()


@pytest.mark.parametrize(
    "mutation", ["accepted", "candidate", "source_delta", "assets", "evidence"]
)
def test_independent_qualification_cannot_be_forged_by_rehashing(
    parent, tmp_path, mutation
):
    from scripts.research.shot_pattern_analysis import (
        supplement_publication as publication,
    )

    root, data = parent
    child = tmp_path / "child"
    create_reviewed_revision(
        root,
        child,
        parent_verifier=lambda p: copy.deepcopy(data),
        render_density=renderer,
    )
    path = child / publication.REVISION_RECEIPT
    receipt = json.loads(path.read_text())
    review_path = child / "review/independent_review.json"
    review = json.loads(review_path.read_text())
    if mutation == "accepted":
        review["accepted"] = False
    elif mutation == "candidate":
        review["candidate_sha256"] = "0" * 64
    elif mutation == "source_delta":
        review["source_delta"] = {"invented": {}}
    elif mutation == "assets":
        review["revised_asset_sha256"] = {}
    else:
        (child / "review/visual.txt").write_text("changed review evidence")
    review_path.write_text(json.dumps(review))
    receipt["qualification"]["review_sha256"] = sha(review_path)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        publication.verify_density_revision(
            child, parent_verifier=lambda p: copy.deepcopy(data)
        )
