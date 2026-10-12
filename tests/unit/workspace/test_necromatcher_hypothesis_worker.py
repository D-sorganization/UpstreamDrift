"""Native hypotheses use an actual compiled fixture and never run an optimizer."""

import pytest


pytestmark = pytest.mark.unit


def test_actual_native_seed_rebinds_exact_start_and_is_recallable(
    hypothesis_case, monkeypatch
) -> None:
    from src.shared.python.workspace.necromatcher_hypothesis import (
        author_native_hypothesis,
    )
    from src.shared.python.workspace import necromatcher_hypothesis_worker as worker
    from src.shared.python.workspace import necromatcher_native_worker as transport
    from src.shared.python.workspace import necromatcher_native as native
    import json

    library, request, parent = hypothesis_case
    compilations = []
    original = native.get_plant

    def compile_model(*args):
        compilations.append(args)
        return original(*args)

    def execute(path, budget, cancelled, *, operation):
        assert operation == "hypothesis"
        return {
            "fit": worker.compute_native_hypothesis(
                json.loads(path.read_text(encoding="utf-8"))
            )
        }

    monkeypatch.setattr(native, "get_plant", compile_model)
    monkeypatch.setattr(transport, "execute_native_research_worker", execute)
    saved = author_native_hypothesis(library, "parent", "hypothesis-seed", request)
    recalled = library.load_fit(saved.dataset_id)
    assert len(compilations) == 1
    assert recalled["q"] == parent["q"]
    assert recalled["frames"] == parent["frames"]
    assert recalled["model_id"] == "candidate"
    original_fit = recalled["evidence"]["original_fit"]
    assert original_fit["camera"] == request.to_record()["camera"]
    assert original_fit["camera"] != parent["evidence"]["original_fit"]["camera"]
    assert original_fit["optimizer_ran"] is False
    assert original_fit["converged"] is False
    assert original_fit["initialization"] is None
    assert original_fit["initial_spline"] == original_fit["spline_start"]
    assert "native_hypothesis" in recalled["provenance"]


def test_worker_rejects_native_unknown_marker_before_persistence(
    hypothesis_case,
) -> None:
    from src.shared.python.workspace.necromatcher_hypothesis import (
        author_native_hypothesis,
    )
    from src.shared.python.workspace.necromatcher_hypothesis_contracts import (
        NativeHypothesisRequest,
    )

    library, request, _ = hypothesis_case
    record = request.to_record()
    record["model"]["attachments"]["origin"][0] = "missing_native_body"
    before = list(library.assets("practice"))
    with pytest.raises((ValueError, RuntimeError), match="frame|body|worker"):
        author_native_hypothesis(
            library, "parent", "bad-seed", NativeHypothesisRequest.from_record(record)
        )
    assert library.assets("practice") == before


def test_clean_worker_persists_fixture_seed(
    hypothesis_case,
) -> None:
    from src.shared.python.workspace import author_native_hypothesis

    library, request, parent = hypothesis_case
    saved = author_native_hypothesis(library, "parent", "clean-seed", request)
    assert library.load_fit(saved.dataset_id)["q"] == parent["q"]


@pytest.mark.parametrize(
    "mutation",
    [
        "lineage",
        "worker_stamp",
        "camera",
        "optimizer",
        "coefficients",
        "config",
        "scales",
        "training",
    ],
)
def test_worker_response_tamper_cannot_publish_fixture_seed(
    hypothesis_case, monkeypatch, mutation: str
) -> None:
    from src.shared.python.workspace import author_native_hypothesis
    from src.shared.python.workspace import necromatcher_hypothesis_worker as worker
    from src.shared.python.workspace import necromatcher_native_worker as transport
    import json

    library, request, _ = hypothesis_case

    def execute(path, budget, cancelled, *, operation):
        fit = worker.compute_native_hypothesis(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if mutation == "lineage":
            del fit["provenance"]["native_hypothesis"]
        elif mutation == "worker_stamp":
            fit["provenance"]["worker_stamp"]["source_sha256"] = "sha256:" + "0" * 64
        elif mutation == "camera":
            fit["evidence"]["original_fit"]["camera"]["translation"][0] += 1
        elif mutation == "optimizer":
            fit["evidence"]["original_fit"]["optimizer_ran"] = True
        elif mutation == "config":
            fit["evidence"]["original_fit"]["config"]["smoothness_weight"] += 1
            fit["provenance"]["request_options"]["config"]["smoothness_weight"] += 1
        elif mutation == "scales":
            fit["provenance"]["request_options"]["coordinate_scales"][0] *= 2
        elif mutation == "training":
            fit["evidence"]["original_fit"]["frame_indices"] = [0, 1, 2]
        else:
            fit["evidence"]["original_fit"]["spline_coefficients"][0] += 1
        return {"fit": fit}

    monkeypatch.setattr(transport, "execute_native_research_worker", execute)
    before = list(library.assets("practice"))
    with pytest.raises(ValueError):
        author_native_hypothesis(library, "parent", "tampered-seed", request)
    assert library.assets("practice") == before
