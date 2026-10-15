"""NSX-40: service composition over a mocked facade, and the result cache (#12572)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pydantic")

from src.api.services import loop_nullspace_service as svc  # noqa: E402
from src.shared.python.loop_nullspace.contracts import (  # noqa: E402
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.facade import (  # noqa: E402
    EngineContext,
    TrajectoryInput,
)

pytestmark = pytest.mark.unit

REPORT: dict[str, Any] = {"schema_version": "1.0.0", "status": "complete"}


class _Source:
    """A source that never builds anything: the facade is mocked."""

    def __init__(self, model_hash: str = "m1") -> None:
        self.model_hash = model_hash
        self.loads = 0

    def describe(self, request):
        return svc.SourceDescription(
            {
                "kind": "stub",
                "source": "stub-trajectory",
                "frames": request.frame_indices,
            },
            {"name": "stub-model", "version": "1", "content_hash": self.model_hash},
        )

    def load(self, request):
        self.loads += 1
        return (
            TrajectoryInput((), (), (), InputProvenance("stub", "stub-trajectory")),
            EngineContext("stub", ModelProvenance("stub-model", "1", self.model_hash)),
        )


@pytest.fixture
def source():
    s = _Source()
    svc.register_source("stub", s)
    svc.RESULT_CACHE.clear()
    yield s
    svc.unregister_source("stub")
    svc.RESULT_CACHE.clear()


@pytest.fixture
def facade_calls(monkeypatch):
    calls: list[tuple] = []

    def fake(trajectory, engine, preset, options, **hooks):
        calls.append((trajectory, engine, preset, options, hooks))
        return dict(REPORT)

    monkeypatch.setattr(svc.facade, "analyze", fake)
    return calls


def _request(**over):
    base = {"source_id": "stub", "preset": "envelope_only", "frame_indices": [0, 3, 6]}
    return svc.LoopNullspaceRequest(**{**base, **over})


def test_run_composes_source_and_facade(source, facade_calls) -> None:
    report = svc.run(_request(options={"metric": "inverse_mass"}))
    ((trajectory, engine, preset, options, _hooks),) = facade_calls
    assert preset == "envelope_only" and options.metric == "inverse_mass"
    assert engine.name == "stub"
    assert report["status"] == "complete"
    assert report["service"]["source_id"] == "stub"
    assert len(report["service"]["cache_key"]) == 64


def test_hooks_are_forwarded_to_the_facade(source, facade_calls) -> None:
    marker = lambda *a: None  # noqa: E731
    svc.run(_request(), on_stage=marker, should_cancel=lambda: False)
    assert facade_calls[0][4]["on_stage"] is marker


def test_identical_inputs_hit_the_cache(source, facade_calls) -> None:
    first = svc.run(_request())
    second = svc.run(_request())
    assert len(facade_calls) == 1 and source.loads == 1
    assert second == first
    assert svc.lookup(_request()) == first


def test_different_options_or_frames_miss(source, facade_calls) -> None:
    svc.run(_request())
    svc.run(_request(options={"include_allocation": True}))
    svc.run(_request(frame_indices=[0, 3, 9]))
    svc.run(_request(preset="capacity_min_effort"))
    assert len(facade_calls) == 4


def test_registry_version_change_misses(source, facade_calls, monkeypatch) -> None:
    svc.run(_request())
    monkeypatch.setattr(svc, "_registry_version", lambda: "9.9.9")
    assert svc.lookup(_request()) is None
    svc.run(_request())
    assert len(facade_calls) == 2


def test_model_provenance_change_misses(source, facade_calls) -> None:
    key = svc.cache_key(_request())
    source.model_hash = "m2"
    assert svc.cache_key(_request()) != key


def test_frame_order_is_part_of_the_key(source) -> None:
    assert svc.cache_key(_request(frame_indices=[0, 3])) != svc.cache_key(
        _request(frame_indices=[3, 0])
    )


def test_unknown_source_is_refused(source) -> None:
    with pytest.raises(svc.RequestRefusedError) as err:
        svc.validate(_request(source_id="missing"))
    assert err.value.code == "unknown_source"


def test_unknown_preset_is_refused(source) -> None:
    with pytest.raises(svc.RequestRefusedError) as err:
        svc.validate(_request(preset="nope"))
    assert err.value.code == "unknown_preset"


def test_request_rejects_bad_frame_lists() -> None:
    with pytest.raises(ValueError):
        _request(frame_indices=[])
    with pytest.raises(ValueError):
        _request(frame_indices=[1, 1])
    with pytest.raises(ValueError):
        _request(frame_indices=[-1])
    with pytest.raises(ValueError):
        _request(frame_indices=list(range(svc.MAX_FRAMES + 1)))


def test_failed_run_is_not_cached(source, monkeypatch) -> None:
    def boom(*a, **k):
        raise ValueError("bad")

    monkeypatch.setattr(svc.facade, "analyze", boom)
    with pytest.raises(ValueError):
        svc.run(_request())
    assert svc.lookup(_request()) is None


def test_cache_is_bounded() -> None:
    cache = svc.ResultCache(max_entries=2)
    for i in range(3):
        cache.put(str(i), {"i": i})
    assert cache.get("0") is None and cache.get("2") == {"i": 2}
