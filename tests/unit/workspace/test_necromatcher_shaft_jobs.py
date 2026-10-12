"""Optional shaft recipes enter job hashes without changing legacy requests."""

from dataclasses import asdict
import importlib
import json
from types import SimpleNamespace
import pytest
from src.shared.python.workspace import NativeRefitOptions
from src.shared.python.workspace import necromatcher_fit_jobs as jobs
from tests.unit.workspace.test_necromatcher_shaft_evidence import (
    binding_inputs as source_binding_inputs,
)  # noqa: F401
from tests.unit.motion_matching.test_shaft_observations import evidence

pytestmark = pytest.mark.unit


class Service:
    def start(self, spec, work):
        self.spec, self.work = spec, work
        return SimpleNamespace()


def test_sdk_free_queue_hash_and_legacy_exact_request(fit_case, monkeypatch):
    library, path, _ = fit_case
    library.add_fit("old", "practice", path)
    options = NativeRefitOptions((0, 2), 2, (1.0,))
    legacy = Service()
    _, root = jobs.start_native_refit(library, "old", "legacy", options, legacy)
    old = json.loads((root / "request.json").read_text(encoding="utf-8"))
    assert "shaft_images" not in old
    assert legacy.spec.hashes.controller_hash == jobs._digest(asdict(options))
    value = evidence()
    bound = SimpleNamespace(
        evidence=value,
        source_clock_sha256="sha256:" + "e" * 64,
        reviewed_frame_count=1,
        observed_segment_count=1,
    )
    calls = []

    def admit(lib, fit, ev):
        calls.append((lib, ev))
        return bound

    monkeypatch.setattr(jobs, "bind_fit_shaft_evidence", admit, raising=False)
    native = importlib.import_module("src.shared.python.workspace.necromatcher_native")
    monkeypatch.setattr(
        native,
        "load_native_fit_binding",
        lambda *args: pytest.fail("SDK compiled in queue"),
    )
    enabled = Service()
    _, root = jobs.start_native_refit(
        library, "old", "enabled", options, enabled, value
    )
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    assert calls == [(library, value)]
    assert request["shaft_images"]["evidence"] == value.to_record()
    assert (
        request["shaft_images"]["binding"]["source_clock_sha256"]
        == bound.source_clock_sha256
    )
    assert enabled.spec.hashes.controller_hash == jobs._digest(
        {"options": asdict(options), "shaft_images": request["shaft_images"]}
    )
    assert enabled.spec.hashes.controller_hash != legacy.spec.hashes.controller_hash


def test_queue_evidence_tamper_rejected_before_run(fit_case, monkeypatch):
    library, path, _ = fit_case
    library.add_fit("old", "practice", path)

    def reject(*args):
        raise ValueError("Original PNG changed")

    monkeypatch.setattr(jobs, "bind_fit_shaft_evidence", reject, raising=False)
    with pytest.raises(ValueError, match="PNG"):
        jobs.start_native_refit(
            library,
            "old",
            "new",
            NativeRefitOptions((0, 2), 2, (1.0,)),
            Service(),
            evidence(),
        )
    assert not (library.root / "runs").exists() or not list(
        (library.root / "runs").iterdir()
    )


def test_recipe_rebind_rejects_post_queue_png_change(source_binding_inputs):  # noqa: F811 -- shared pytest fixture
    library, value, row, asset = source_binding_inputs
    source = {
        "capture_id": value.capture_id,
        "capture_hash": value.capture_sha256,
        "frames": [value.frames[0].frame.to_dict()],
    }
    recipe = jobs._queue_shaft_recipe(library, source, value, 0.5)
    row["frame"]["pts_ticks"] += 1
    with pytest.raises(ValueError):
        jobs._verify_shaft_request(library, source, recipe)


@pytest.mark.parametrize("changed_before_publish", [False, True])
def test_publication_rebind_prevents_changed_png_without_overwriting_parent(
    fit_case, monkeypatch, changed_before_publish
):
    from copy import deepcopy
    from src.shared.python.workspace.necromatcher_shaft_evidence import (
        BoundShaftEvidence,
    )

    library, path, original = fit_case
    library.add_fit("old", "practice", path)
    value = evidence()
    bound = BoundShaftEvidence(value, "sha256:" + "e" * 64, 1, 1)
    state = {"changed": False, "calls": 0}

    def admit(*args):
        state["calls"] += 1
        if state["changed"]:
            raise ValueError("Original PNG changed")
        return bound

    monkeypatch.setattr(jobs, "bind_fit_shaft_evidence", admit)

    def execute(request_path, *args):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = deepcopy(original)
        payload["evidence"]["shaft_axis"] = {"recipe": request["shaft_images"]}
        return {"fit": payload}

    monkeypatch.setattr(jobs, "_execute_worker", execute)
    service = Service()
    _, root = jobs.start_native_refit(
        library, "old", "new", NativeRefitOptions((0, 2), 2, (1.0,)), service, value
    )
    outcome = service.work(lambda event: None, lambda: False)
    assert state["calls"] == 2
    state["changed"] = changed_before_publish
    if changed_before_publish:
        with pytest.raises(ValueError, match="PNG"):
            outcome.publish()
        with pytest.raises(KeyError):
            library.load_fit("new")
    else:
        outcome.publish()
        assert library.load_fit("new")["evidence"]["shaft_axis"][
            "recipe"
        ] == jobs._shaft_record(bound, 0.5)
    assert state["calls"] == 3
    assert library.load_fit("old") == original


@pytest.mark.parametrize("field", ["reviewed_frame_count", "observed_segment_count"])
@pytest.mark.parametrize("count", [True, 1.0])
def test_verify_recipe_rejects_noninteger_counts_before_source_read(
    monkeypatch, field, count
):
    from src.shared.python.workspace.necromatcher_shaft_evidence import (
        BoundShaftEvidence,
    )

    value = evidence()
    recipe = jobs._shaft_record(
        BoundShaftEvidence(value, "sha256:" + "e" * 64, 1, 1), 0.5
    )
    recipe["binding"][field] = count
    monkeypatch.setattr(
        jobs,
        "bind_fit_shaft_evidence",
        lambda *args: pytest.fail(
            "Source read reached before strict binding validation"
        ),
    )
    with pytest.raises(ValueError, match="counts|integer"):
        jobs._verify_shaft_request("library", {}, recipe)
