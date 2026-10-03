"""Reviewed source windows cross HTTP without changing disabled call shapes."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.routes.necromatcher import get_library, get_refits, router
from tests.unit.workspace.test_necromatcher_source_scope import make_scope
from tests.unit.motion_matching.test_shaft_observations import evidence

pytestmark = pytest.mark.unit


def test_raw_receipt_upload_preserves_bytes_and_returns_portable_dto(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from src.api.routes import necromatcher as module

    _, scope = make_scope(tmp_path)
    calls: list[Any] = []
    library = SimpleNamespace()
    monkeypatch.setattr(
        module,
        "import_fit_source_scope_review",
        lambda *args: calls.append(args) or scope,
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_library] = lambda: library
    raw = b'{ "schema": "necromatcher/source-fit-scope-review/1" }\n'
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/fits/parent/source-scope-reviews",
            files={"file": ("review.json", raw, "application/json")},
        )
    assert response.status_code == 201, response.text
    assert response.json() == scope.to_record()
    assert calls == [(library, "parent", raw)]


def test_oversized_receipt_upload_never_reaches_library(monkeypatch: Any) -> None:
    from src.api.routes import necromatcher as module

    calls: list[Any] = []
    monkeypatch.setattr(
        module, "import_fit_source_scope_review", lambda *args: calls.append(args)
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_library] = lambda: SimpleNamespace()
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/fits/parent/source-scope-reviews",
            files={
                "file": ("review.json", b"x" * (1024 * 1024 + 1), "application/json")
            },
        )
    assert response.status_code == 413
    assert not calls


def test_explicit_null_scope_keeps_legacy_submit_shape() -> None:
    calls: list[Any] = []
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_refits] = lambda: SimpleNamespace(
        submit=lambda *args, **kwargs: calls.append((args, kwargs)) or {"run_id": "r"}
    )
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/fits/parent/refits",
            json={
                "new_fit_id": "new",
                "frame_indices": [0, 3],
                "knot_count": 2,
                "coordinate_scales": [1.0],
                "source_scope": None,
            },
        )
    assert response.status_code == 202
    assert len(calls[0][0]) == 3 and calls[0][1] == {}


def test_canonical_scope_rejection_has_no_unscoped_retry(tmp_path: Any) -> None:
    _, scope = make_scope(tmp_path)
    calls: list[Any] = []

    def reject(*args: Any, **kwargs: Any) -> dict[str, str]:
        calls.append((args, kwargs))
        raise ValueError("Scope review source clock mismatch")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_refits] = lambda: SimpleNamespace(submit=reject)
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/fits/parent/refits",
            json={
                "new_fit_id": "new",
                "frame_indices": [0, 3],
                "knot_count": 2,
                "coordinate_scales": [1.0],
                "source_scope": scope.to_record(),
            },
        )
    assert response.status_code == 422
    assert len(calls) == 1 and calls[0][1] == {"source_scope": scope}


@pytest.mark.parametrize("shaft", [False, True])
def test_scope_is_typed_before_submission(tmp_path: Any, shaft: bool) -> None:
    _, scope = make_scope(tmp_path)
    calls: list[Any] = []
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_refits] = lambda: SimpleNamespace(
        submit=lambda *args, **kwargs: calls.append((args, kwargs)) or {"run_id": "r"}
    )
    payload = {
        "new_fit_id": "new",
        "frame_indices": [0, 3],
        "knot_count": 2,
        "coordinate_scales": [1.0],
        "source_scope": scope.to_record(),
    }
    if shaft:
        payload["shaft_evidence"] = evidence().to_record()
    with TestClient(app) as client:
        response = client.post("/necromatcher/fits/parent/refits", json=payload)
    assert response.status_code == 202, response.text
    assert len(calls[0][0]) == (4 if shaft else 3)
    assert calls[0][1] == {"source_scope": scope}


@pytest.mark.parametrize("fault", ["boolean", "schema", "contact", "extra"])
def test_bad_scope_rejected_before_queue(tmp_path: Any, fault: str) -> None:
    _, scope = make_scope(tmp_path)
    record = deepcopy(scope.to_record())
    if fault == "boolean":
        record["first_frame"] = True
    elif fault == "schema":
        record["schema"] = "foreign"
    elif fault == "contact":
        record["review"]["contact_calibrated"] = True
    else:
        record["invented"] = 1
    calls: list[Any] = []
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_refits] = lambda: SimpleNamespace(
        submit=lambda *args, **kwargs: calls.append(args)
    )
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/fits/parent/refits",
            json={
                "new_fit_id": "new",
                "frame_indices": [0, 3],
                "knot_count": 2,
                "coordinate_scales": [1.0],
                "source_scope": record,
            },
        )
    assert response.status_code == 422
    assert not calls
