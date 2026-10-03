"""Source scopes stop excluded providers and preserve legacy queue identities."""

from dataclasses import replace, asdict
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
from test_scope_fixtures import fixture_capture, fixture_review_artifact
from src.shared.python.workspace import necromatcher_fit as fit
from src.shared.python.workspace import necromatcher_fit_jobs as jobs

pytestmark = pytest.mark.unit


def scoped(tmp_path, source=None, end=3) -> tuple:
    from src.shared.python.workspace.necromatcher_source_scope import (
        SourceFitScope,
        SourceScopeReview,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    identity = fixture_capture()
    if source is not None:
        identity = replace(
            identity,
            capture_id=source["capture_id"],
            capture_hash=source["capture_hash"],
        )
    artifact = fixture_review_artifact(tmp_path, identity, 0, end)
    scope = SourceFitScope(
        identity.capture_id,
        identity.capture_hash,
        identity.source_clock_sha256,
        0,
        end,
        SourceScopeReview(
            artifact,
            __import__("pathlib").Path(artifact.path).stat().st_size,
            identity.frames[0],
            identity.frames[end],
            "Synthetic conservative window",
            "exclude transition",
        ),
    )
    return identity, scope


def test_scoped_queue_rejects_body_before_scheduler_or_run_directory(
    fit_case, tmp_path, monkeypatch
) -> None:
    library, path, source = fit_case
    library.add_fit("old", "practice", path)
    identity, scope = scoped(tmp_path, source, end=2)
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    monkeypatch.setattr(
        library, "load_source_scope_review", lambda _: scope, raising=False
    )
    called = []
    service = SimpleNamespace(start=lambda *a, **k: called.append("scheduler"))
    with pytest.raises(ValueError, match="scope|outside"):
        jobs.start_native_refit(
            library,
            "old",
            "new",
            jobs.NativeRefitOptions((0, 2), 2, (1.0,)),
            service,
            source_scope=scope,
        )
    assert called == []
    assert not (library.root / "runs").exists()


def test_queue_scope_is_hashed_and_legacy_absence_unchanged(
    fit_case, tmp_path, monkeypatch
) -> None:
    library, path, source = fit_case
    library.add_fit("old", "practice", path)
    identity, scope = scoped(tmp_path, source)
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    monkeypatch.setattr(
        library, "load_source_scope_review", lambda _: scope, raising=False
    )
    monkeypatch.setattr(
        jobs,
        "fit_execution_stamp",
        lambda: {"source_sha256": "source", "runtime_sha256": "runtime"},
    )
    specs = []
    service = SimpleNamespace(start=lambda spec, **kw: specs.append(spec) or "handle")
    options = jobs.NativeRefitOptions((0, 2), 2, (1.0,))
    _, legacy = jobs.start_native_refit(library, "old", "legacy", options, service)
    _, root = jobs.start_native_refit(
        library, "old", "scoped", options, service, source_scope=scope
    )
    old = json.loads((legacy / "request.json").read_text())
    new = json.loads((root / "request.json").read_text())
    assert "source_scope" not in old
    assert specs[0].hashes.controller_hash == jobs._digest(asdict(options))
    assert new["source_scope"] == scope.to_record()
    assert specs[1].hashes.controller_hash != specs[0].hashes.controller_hash


def test_inherited_scope_cannot_be_widened_or_erased(tmp_path, monkeypatch) -> None:
    identity, scope = scoped(tmp_path)
    source = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {"source_fit_scope": scope.to_record()},
    }
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    bound = fit.admit_refit_scope(None, source, (0, 2), None, None, None)
    assert bound.scope == scope
    _, wide = scoped(tmp_path / "wide", end=4)
    with pytest.raises(ValueError, match="widen|scope"):
        fit.admit_refit_scope(None, source, (0, 2), None, None, wide)


def test_schedule_outside_selected_domain_is_rejected(tmp_path, monkeypatch) -> None:
    from src.shared.python.motion_matching.historical_fit import ImageFitConfig
    from src.shared.python.motion_matching.historical_fit.contact_schedule import (
        ContactPinPhase,
        ContactPinSchedule,
        ScheduledConstraintOptions,
    )
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane

    identity, scope = scoped(tmp_path)
    source = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {},
    }
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    # Exact clock fixture starts at 1 second, advances by 1/30.
    first = identity.frames[0]
    last = identity.frames[4]

    def pts(f):
        return (f.pts_ticks * f.timebase_numerator, f.timebase_denominator)

    phase = ContactPinPhase(pts(first), pts(last), (), ("sha256:" + last.frame_sha256,))
    config = ImageFitConfig(
        constraint_options=ScheduledConstraintOptions(
            ConstraintOptions(
                GroundPlane((0.0, 0.0, 1.0), 0.0), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ),
            ContactPinSchedule(identity.capture_id, identity.capture_hash, (phase,)),
        )
    )
    with pytest.raises(ValueError, match="contact|schedule|domain"):
        fit.admit_refit_scope(None, source, (0, 2), config, None, scope)


def test_worker_scope_rejects_before_native_binding(tmp_path, monkeypatch) -> None:
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    identity, scope = scoped(tmp_path, end=2)
    source = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {},
    }
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    calls = []
    monkeypatch.setattr(
        worker, "load_native_fit_binding", lambda *a: calls.append("native")
    )
    monkeypatch.setattr(
        worker,
        "fit_execution_stamp",
        lambda: {"source_sha256": "s", "runtime_sha256": "r"},
    )
    library = SimpleNamespace(
        load_asset=lambda *_: SimpleNamespace(metadata={"hash": "h"}),
        load_fit=lambda *_: source,
        load_source_scope_review=lambda _: scope,
    )
    monkeypatch.setattr(worker, "NecromatcherLibrary", lambda *_: library)
    options = jobs.NativeRefitOptions((0, 2), 2, (1.0,))
    request = {
        "library_root": "unused",
        "source_fit_id": "old",
        "source_fit_hash": "h",
        "execution_stamp": {"source_sha256": "s", "runtime_sha256": "r"},
        "options": asdict(options),
        "source_scope": scope.to_record(),
    }
    with pytest.raises(ValueError, match="scope|outside"):
        worker.compute_native_refit(request)
    assert calls == []


def test_builder_inherits_scope_for_model_camera_seed(tmp_path) -> None:
    from test_necromatcher_fit_records import specimen
    from src.shared.python.workspace.necromatcher_fit_records import (
        build_native_fit_payload,
    )

    request, source, result, dense, stamp, expansion = specimen()
    _, scope = scoped(tmp_path)
    source["provenance"]["source_fit_scope"] = scope.to_record()
    identity = fixture_capture()
    source["provenance"]["source_fit_scope_binding"] = {
        "frame_indices": [0, 2],
        "first_pts": [100, 30],
        "last_pts": [102, 30],
        "source_clock_sha256": identity.source_clock_sha256,
    }
    output = build_native_fit_payload(
        request, source, result, dense, stamp, 0.01, coordinate_expansion=expansion
    )
    assert output["provenance"]["source_fit_scope"] == scope.to_record()
    assert source["provenance"]["source_fit_scope"] == scope.to_record()


@pytest.mark.parametrize("mutation", ["erase", "dense", "training", "clock", "frames"])
def test_scoped_payload_recall_rejects_scope_and_sample_tampering(
    tmp_path, monkeypatch, mutation
) -> None:
    identity, scope = scoped(tmp_path)
    parent = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {"source_fit_scope": scope.to_record()},
    }
    payload = {
        **parent,
        "frame_indices": [0, 1, 2],
        "frames": [f.to_dict() for f in identity.frames[:3]],
        "provenance": deepcopy(parent["provenance"]),
        "evidence": {
            "original_fit": {
                "config": asdict(jobs.NativeRefitOptions((0, 2), 2, (1.0,)))["config"],
                "frame_indices": [0, 2],
                "source_times": [
                    float(identity.frames[i].presentation_time) for i in (0, 2)
                ],
            }
        },
    }
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    bound = fit.admit_refit_scope(None, parent, (0, 2))
    payload["provenance"]["source_fit_scope_binding"] = fit.scope_binding_record(
        bound, (0, 2)
    )
    payload["provenance"]["request_options"] = asdict(
        jobs.NativeRefitOptions((0, 2), 2, (1.0,))
    )
    fit.validate_scope_payload(None, payload, parent)
    if mutation == "erase":
        payload["provenance"].pop("source_fit_scope")
    elif mutation == "dense":
        payload["frame_indices"] = [0, 2]
        payload["frames"] = [payload["frames"][0], payload["frames"][2]]
    elif mutation == "training":
        payload["evidence"]["original_fit"]["frame_indices"] = [0, 3]
    elif mutation == "clock":
        payload["evidence"]["original_fit"]["source_times"][1] += 1
    elif mutation == "frames":
        payload["frames"][1]["pts_ticks"] += 1
    with pytest.raises(ValueError, match="scope|Scoped|source|domain"):
        fit.validate_scope_payload(None, payload, parent)


def test_builder_cannot_erase_scope_with_explicit_null(tmp_path) -> None:
    from test_necromatcher_fit_records import specimen
    from src.shared.python.workspace.necromatcher_fit_records import (
        build_native_fit_payload,
    )

    request, source, result, dense, stamp, expansion = specimen()
    _, scope = scoped(tmp_path)
    source["provenance"]["source_fit_scope"] = scope.to_record()
    request["source_scope"] = None
    with pytest.raises(ValueError, match="erase|scope"):
        build_native_fit_payload(
            request, source, result, dense, stamp, 0.01, coordinate_expansion=expansion
        )


def test_scoped_preserved_restart_rejects_before_scheduler(
    fit_case, tmp_path, monkeypatch
) -> None:
    library, path, source = fit_case
    library.add_fit("old", "practice", path)
    identity, scope = scoped(tmp_path, source)
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    monkeypatch.setattr(
        library, "load_source_scope_review", lambda _: scope, raising=False
    )
    called = []
    service = SimpleNamespace(start=lambda *a, **k: called.append("schedule"))
    with pytest.raises(ValueError, match="preserved|restriction"):
        jobs.start_native_refit(
            library,
            "old",
            "narrow",
            jobs.NativeRefitOptions(
                (0, 2), 2, (1.0,), initialization_source="preserved_spline"
            ),
            service,
            source_scope=scope,
        )
    assert called == []


def test_session_optional_scope_keeps_legacy_call_shape(tmp_path, monkeypatch) -> None:
    from src.shared.python.workspace import necromatcher_refits as refits

    _, scope = scoped(tmp_path)
    library = SimpleNamespace(root=tmp_path)
    session = refits.NativeRefitSession(library)
    calls = []
    root = tmp_path / "run"
    handle = SimpleNamespace(join=lambda **kw: None)
    monkeypatch.setattr(
        refits,
        "start_native_refit",
        lambda *a, **kw: calls.append((a, kw)) or (handle, root),
    )
    monkeypatch.setattr(session, "_view", lambda _: {})
    options = jobs.NativeRefitOptions((0, 2), 2, (1.0,))
    session.submit("old", "new", options)
    assert len(calls[0][0]) == 5 and calls[0][1] == {}
    session.submit("old", "scoped", options, source_scope=scope)
    assert len(calls[1][0]) == 7 and calls[1][0][-2:] == (None, scope)
    session._service.close()


def test_scoped_publication_rejects_candidate_erase_before_add(
    tmp_path, monkeypatch
) -> None:
    identity, scope = scoped(tmp_path)
    source = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {},
    }
    monkeypatch.setattr(fit, "capture_identity", lambda *_: identity)
    library = SimpleNamespace(
        load_fit=lambda _: source, load_source_scope_review=lambda _: scope
    )
    bound = fit.admit_refit_scope(library, source, (0, 2), requested=scope)
    request = {
        "source_fit_id": "old",
        "source_scope": scope.to_record(),
        "source_scope_binding": fit.scope_binding_record(bound, (0, 2)),
        "options": asdict(jobs.NativeRefitOptions((0, 2), 2, (1.0,))),
    }
    called = []
    with pytest.raises(ValueError, match="Candidate"):
        jobs._verify_scope_request(library, request, {"provenance": {}})
        called.append("publish")
    assert called == []
