"""Exact scoped publication domains and complete contact recipes."""

from dataclasses import asdict
from types import SimpleNamespace
from copy import deepcopy
import pytest
from test_necromatcher_scope_integration import scoped
from src.shared.python.workspace import necromatcher_fit as fit
from src.shared.python.workspace import necromatcher_fit_jobs as jobs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("where", ["request", "payload"])
def test_selected_domain_binding_rejects_bool_integer_equality(
    tmp_path, monkeypatch, where
) -> None:
    identity, scope = scoped(tmp_path)
    source = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {},
    }
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    bound = fit.admit_refit_scope(None, source, (0, 2), requested=scope)
    binding = fit.scope_binding_record(bound, (0, 2))
    binding["frame_indices"][0] = False
    if where == "request":
        library = SimpleNamespace(
            load_fit=lambda _: source, load_source_scope_review=lambda _: scope
        )
        request = {
            "source_fit_id": "old",
            "source_scope": scope.to_record(),
            "source_scope_binding": binding,
            "options": asdict(jobs.NativeRefitOptions((0, 2), 2, (1.0,))),
        }
        with pytest.raises(ValueError, match="binding"):
            jobs._verify_scope_request(library, request)
    else:
        payload = {
            **source,
            "frame_indices": [0, 1, 2],
            "frames": [f.to_dict() for f in identity.frames[:3]],
            "provenance": {
                "source_fit_scope": scope.to_record(),
                "source_fit_scope_binding": binding,
            },
            "evidence": {
                "original_fit": {
                    "config": asdict(jobs.NativeRefitOptions((0, 2), 2, (1.0,)))[
                        "config"
                    ],
                    "frame_indices": [0, 2],
                    "source_times": [
                        float(identity.frames[i].presentation_time) for i in (0, 2)
                    ],
                }
            },
        }
        payload["provenance"]["request_options"] = asdict(
            jobs.NativeRefitOptions((0, 2), 2, (1.0,))
        )
        with pytest.raises(ValueError, match="binding"):
            fit.validate_scope_payload(None, payload)


def test_worker_request_cannot_delete_inherited_scope(tmp_path, monkeypatch) -> None:
    identity, scope = scoped(tmp_path)
    source = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {"source_fit_scope": scope.to_record()},
    }
    library = SimpleNamespace(
        load_fit=lambda _: source, load_source_scope_review=lambda _: scope
    )
    with pytest.raises(ValueError, match="inherited|scope"):
        jobs._verify_scope_request(library, {"source_fit_id": "parent"})


def scoped_payload(tmp_path, monkeypatch, training=(0, 2)) -> tuple:
    identity, scope = scoped(tmp_path, end=4)
    source = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {},
    }
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    library = SimpleNamespace(
        load_fit=lambda _: source, load_source_scope_review=lambda _: scope
    )
    options = asdict(jobs.NativeRefitOptions(training, 2, (1.0,)))
    bound = fit.admit_refit_scope(library, source, training, requested=scope)
    binding = fit.scope_binding_record(bound, training)
    request = {
        "source_fit_id": "old",
        "source_scope": scope.to_record(),
        "source_scope_binding": binding,
        "options": options,
    }
    payload = {
        **source,
        "frame_indices": list(range(training[0], training[-1] + 1)),
        "frames": [f.to_dict() for f in identity.frames[: training[-1] + 1]],
        "provenance": {
            "source_fit_scope": scope.to_record(),
            "source_fit_scope_binding": deepcopy(binding),
            "request_options": deepcopy(options),
        },
        "evidence": {
            "original_fit": {
                "frame_indices": list(training),
                "source_times": [
                    float(identity.frames[i].presentation_time) for i in training
                ],
                "config": deepcopy(options["config"]),
            }
        },
    }
    return library, request, payload


def test_publication_rejects_independently_valid_changed_selected_domain(
    tmp_path, monkeypatch
) -> None:
    library, request, _ = scoped_payload(tmp_path, monkeypatch)
    _, _, changed = scoped_payload(tmp_path, monkeypatch, (0, 3))
    fit.validate_scope_payload(library, changed)
    with pytest.raises(ValueError, match="admitted|selected|binding"):
        jobs._verify_scope_request(library, request, changed)


@pytest.mark.parametrize(
    "mutation",
    ["missing_request", "different_request", "missing_original", "empty_both"],
)
def test_scoped_recall_requires_matching_complete_config(
    tmp_path, monkeypatch, mutation
) -> None:
    library, _, payload = scoped_payload(tmp_path, monkeypatch)
    if mutation == "missing_request":
        payload["provenance"]["request_options"].pop("config")
    elif mutation == "missing_original":
        payload["evidence"]["original_fit"].pop("config")
    elif mutation == "empty_both":
        payload["provenance"]["request_options"]["config"] = {}
        payload["evidence"]["original_fit"]["config"] = {}
    else:
        payload["provenance"]["request_options"]["config"]["max_iterations"] += 1
    with pytest.raises(ValueError, match="config|recipe"):
        fit.validate_scope_payload(library, payload)


@pytest.mark.parametrize("mutation", ["config", "bool_indices"])
def test_publication_rejects_consistent_but_changed_request_options(
    tmp_path, monkeypatch, mutation
) -> None:
    library, request, payload = scoped_payload(tmp_path, monkeypatch)
    if mutation == "config":
        for record in (
            payload["provenance"]["request_options"]["config"],
            payload["evidence"]["original_fit"]["config"],
        ):
            record["max_iterations"] += 1
    else:
        payload["provenance"]["request_options"]["frame_indices"] = [False, 2]
    fit.validate_scope_payload(library, payload)
    with pytest.raises(ValueError, match="admitted|options"):
        jobs._verify_scope_request(library, request, payload)
