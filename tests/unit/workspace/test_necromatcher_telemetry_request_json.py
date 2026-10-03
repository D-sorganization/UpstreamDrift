"""Live typed queue recipes bind to their exact stored JSON without coercion."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest

from src.shared.python.motion_matching.historical_fit import ImageFitConfig
from src.shared.python.motion_matching.constraint_kinematics import ConstraintOptions
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.workspace import NativeRefitOptions
from src.shared.python.workspace import necromatcher_fit_telemetry as telemetry


def _queue_request(tmp_path):
    root = tmp_path / "runs" / ("a" * 32)
    root.mkdir(parents=True)
    options = NativeRefitOptions(
        (0, 3, 6),
        2,
        (1.0,),
        ImageFitConfig(
            interior_fractions=(0.25, 0.75),
            coordinate_bounds=(("q", -1.0, 1.0),),
            constraint_options=ConstraintOptions(
                GroundPlane((0.0, 0.0, 1.0), 0.0), 1.0, 1.0, 1.0, 0.01, 0.1, 0.01
            ),
        ),
    )
    request = {
        "library_root": str(tmp_path),
        "source_fit_id": "parent",
        "source_fit_hash": "sha256:" + "b" * 64,
        "execution_stamp": {"source_sha256": "c" * 64, "runtime_sha256": "d" * 64},
        "options": asdict(options),
    }
    path = root / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path, request


def test_tuple_bearing_queue_child_receipt_binds_to_parent_live_request(tmp_path):
    path, live = _queue_request(tmp_path)
    before = path.read_bytes()
    child = json.loads(before)
    assert child != live
    telemetry.write_worker_telemetry(path, child, None, 2.5, "controlled child failure")
    measured = telemetry.read_worker_telemetry(path.parent, live)
    assert measured.worker_elapsed_s == 2.5 and measured.nfev is None
    assert path.read_bytes() == before
    record = json.loads((path.parent / "worker-telemetry.json").read_bytes())
    assert record["binding"]["request_sha256"] == hashlib.sha256(before).hexdigest()


def test_changed_nested_recipe_still_rejects(tmp_path):
    path, live = _queue_request(tmp_path)
    telemetry.write_worker_telemetry(
        path, json.loads(path.read_bytes()), None, 1.0, "failed"
    )
    changed = deepcopy(live)
    changed["options"]["config"]["coordinate_bounds"] = (("q", -1.0, 1.01),)
    with pytest.raises(ValueError, match="binding"):
        telemetry.read_worker_telemetry(path.parent, changed)


@pytest.mark.parametrize("stored, changed", [(True, 1), (False, 0), (1, 1.0)])
def test_json_scalar_types_cannot_impersonate_one_another(tmp_path, stored, changed):
    path, request = _queue_request(tmp_path)
    request = json.loads(path.read_bytes())
    request["options"]["config"]["probe"] = stored
    path.write_text(json.dumps(request), encoding="utf-8")
    telemetry.write_worker_telemetry(path, request, None, 1.0, "failed")
    different = deepcopy(request)
    different["options"]["config"]["probe"] = changed
    with pytest.raises(ValueError, match="binding"):
        telemetry.read_worker_telemetry(path.parent, different)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_request_never_produces_measurement_receipt(tmp_path, value):
    path, request = _queue_request(tmp_path)
    request = json.loads(path.read_bytes())
    request["options"]["config"]["probe"] = value
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        telemetry.write_worker_telemetry(path, request, None, 1.0, "failed")
    assert not (path.parent / "worker-telemetry.json").exists()


def test_nonstring_mapping_key_cannot_match_json_string_key(tmp_path):
    path, request = _queue_request(tmp_path)
    request = json.loads(path.read_bytes())
    request["options"]["config"]["probe"] = {"1": "value"}
    path.write_text(json.dumps(request), encoding="utf-8")
    telemetry.write_worker_telemetry(path, request, None, 1.0, "failed")
    changed = deepcopy(request)
    changed["options"]["config"]["probe"] = {1: "value"}
    with pytest.raises(ValueError, match="binding"):
        telemetry.read_worker_telemetry(path.parent, changed)


def test_parent_worker_result_reaches_rejected_publication_with_live_tuples(
    tmp_path, monkeypatch
):
    from src.shared.python.estimation import SolverTelemetry
    from src.shared.python.workspace import necromatcher_fit_jobs as jobs

    path, request = _queue_request(tmp_path)
    request["new_fit_id"] = "future"
    stamp = request["execution_stamp"]
    monkeypatch.setattr(jobs, "fit_execution_stamp", lambda: stamp)
    measured = SolverTelemetry(
        nfev=3, njev=2, solver_elapsed_s=0.1, termination_reason="gtol"
    )
    fit = {
        "evidence": {
            "original_fit": {
                "optimizer_ran": True,
                "solver_telemetry": measured.to_record(),
            },
            "rejection_reasons": ["research_only"],
        }
    }

    def child(request_path, *_):
        decoded = json.loads(request_path.read_bytes())
        telemetry.write_worker_telemetry(request_path, decoded, fit, 1.0, "gtol")
        return {"fit": fit}

    monkeypatch.setattr(jobs, "_execute_worker", child)
    published = []
    library = SimpleNamespace(
        load_asset=lambda _: SimpleNamespace(
            metadata={"hash": request["source_fit_hash"]}
        ),
        add_fit=lambda *args: published.append(args),
    )
    spec = SimpleNamespace(
        run_root=path.parent, hashes=SimpleNamespace(solver_hash=stamp["source_sha256"])
    )
    work = jobs._refit_work(
        library, SimpleNamespace(budget_wall_s=300), request, spec, "session"
    )
    outcome = work(lambda _: None, lambda: False)
    assert outcome.acceptance.value == "rejected" and published == []
    outcome.publish()
    assert published[0][:2] == ("future", "session")
    assert json.loads((path.parent / "candidate.json").read_bytes()) == fit
    assert telemetry.read_worker_telemetry(path.parent, request).nfev == 3


def test_semantically_equal_request_byte_rewrite_still_breaks_sidecar_hash(tmp_path):
    path, request = _queue_request(tmp_path)
    decoded = json.loads(path.read_bytes())
    telemetry.write_worker_telemetry(path, decoded, None, 1.0, "failed")
    path.write_text(json.dumps(decoded, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        telemetry.read_worker_telemetry(path.parent, request)
