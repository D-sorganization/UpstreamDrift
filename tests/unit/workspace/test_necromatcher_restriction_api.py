"""The app delegates restriction admission to canonical public options/session."""

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.necromatcher import RefitRequest, get_refits, router
from tests.unit.workspace.test_necromatcher_source_scope import make_scope
from src.shared.python.motion_matching.historical_fit import (
    ShaftAxisEvidence,
    ShaftAxisSegment,
    SourceBoundShaftFrame,
)


def recipe() -> dict[str, Any]:
    return {
        "new_fit_id": "restricted-seed",
        "frame_indices": [0, 1, 3],
        "knot_count": 3,
        "coordinate_scales": [1.0],
        "config": {"initialization_policy": "strict"},
        "operation": "restrict_initialization",
        "initialization_source": "restricted_spline",
    }


@pytest.mark.parametrize("with_shaft", [False, True])
def test_restriction_http_forwards_typed_scope_without_optimizer(
    tmp_path: Path,
    with_shaft: bool,
) -> None:
    identity, scope = make_scope(tmp_path)
    evidence = ShaftAxisEvidence(
        identity.capture_id,
        identity.capture_hash,
        identity.source_sha256,
        (32, 24),
        (
            SourceBoundShaftFrame(
                0,
                identity.frames[0],
                identity.png_sha256[0],
                ShaftAxisSegment(
                    "observed",
                    ((1.0, 1.0), (2.0, 2.0)),
                    "reviewer",
                    "synthetic",
                    0.5,
                    None,
                    3.0,
                ),
            ),
        ),
    )
    calls: list[tuple] = []

    class Session:
        def submit(self, *args: Any, **kwargs: Any) -> dict:
            calls.append((args, kwargs))
            return {"status": "pending", "new_fit_id": "restricted-seed"}

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_refits] = lambda: Session()
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/fits/parent/refits",
            json={
                **recipe(),
                "source_scope": scope.to_record(),
                **({"shaft_evidence": evidence.to_record()} if with_shaft else {}),
            },
        )
    assert response.status_code == 202, response.text
    args, kwargs = calls[0]
    assert args[:2] == ("parent", "restricted-seed")
    assert args[2].operation == "restrict_initialization"
    assert args[2].initialization_source == "restricted_spline"
    assert args[2].config.initialization_policy == "strict"
    assert kwargs == {"source_scope": scope}
    assert args[3:] == ((evidence,) if with_shaft else ())


@pytest.mark.parametrize(
    "changes",
    [
        {"operation": "fit"},
        {"initialization_source": "sampled_parent"},
        {"config": {"initialization_policy": "authored_range_project_zero_slopes"}},
    ],
)
def test_invalid_restriction_pair_never_submits(changes: dict) -> None:
    calls: list[tuple] = []

    class Session:
        def submit(self, *args: Any, **kwargs: Any) -> dict:
            calls.append(args)
            return {}

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_refits] = lambda: Session()
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/fits/parent/refits", json={**recipe(), **changes}
        )
    assert response.status_code in {400, 422}
    assert calls == []


def test_legacy_options_remain_exact() -> None:
    record = recipe()
    record.pop("operation")
    record.pop("initialization_source")
    options = RefitRequest.model_validate(record).options()
    assert options.operation == "fit"
    assert options.initialization_source == "sampled_parent"


def test_legacy_http_keeps_three_positional_arguments() -> None:
    calls: list[tuple] = []

    class Session:
        def submit(self, *args: Any, **kwargs: Any) -> dict:
            calls.append((args, kwargs))
            return {"status": "pending"}

    record = recipe()
    record.pop("operation")
    record.pop("initialization_source")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_refits] = lambda: Session()
    with TestClient(app) as client:
        response = client.post("/necromatcher/fits/parent/refits", json=record)
    assert response.status_code == 202
    args, kwargs = calls[0]
    assert len(args) == 3 and kwargs == {}
    assert args[2] == RefitRequest.model_validate(record).options()
