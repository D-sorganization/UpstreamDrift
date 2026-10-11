"""Unit tests for the Frankenstein assembly API (CMB-10, #11661).

The web Model Explorer drives drag-and-drop through these stateless endpoints:
the client keeps an ordered assembly plan and the server replays it through the
headless ``AssemblySession`` (the same rules as the PyQt6 assembly panel).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src.api.route_registry import register_routes
from src.api.routes.model_explorer_assembly import router

pytestmark = pytest.mark.unit

PARTS = "/tools/model-explorer/parts"
ASSEMBLY = "/tools/model-explorer/assembly"
EXPORT = "/tools/model-explorer/assembly/export"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _plan(*steps: tuple[str, str], **extra: Any) -> dict[str, Any]:
    return {
        "base_part_id": "humanoid_torso",
        "steps": [{"part_id": p, "host_port": h} for p, h in steps],
        **extra,
    }


def test_parts_lists_the_bundled_catalog_with_typed_ports(client: TestClient) -> None:
    response = client.get(PARTS)
    assert response.status_code == 200
    data = response.json()
    ids = {part["part_id"] for part in data["parts"]}
    assert {"humanoid_torso", "leg_left", "head"} <= ids
    assert {c["id"] for c in data["categories"]} >= {"limb"}
    torso = next(p for p in data["parts"] if p["part_id"] == "humanoid_torso")
    hip = next(port for port in torso["ports"] if port["name"] == "hip_left")
    assert hip["polarity"] == "socket"
    assert hip["port_type"] == "hip"


def test_parts_filters_by_category_and_query(client: TestClient) -> None:
    data = client.get(PARTS, params={"category": "limb", "query": "leg"}).json()
    assert data["parts"]
    assert all(p["category"] == "limb" for p in data["parts"])
    assert all("leg" in p["name"].lower() for p in data["parts"])


def test_empty_plan_returns_the_base_part(client: TestClient) -> None:
    response = client.post(ASSEMBLY, json=_plan())
    assert response.status_code == 200
    data = response.json()
    assert [p["instance_id"] for p in data["placed"]] == ["humanoid_torso_0"]
    assert data["validation"]["ok"] is True
    assert "hip_left" in {s["name"] for s in data["free_sockets"]}
    assert data["model"]["link_count"] > 0
    assert data["drop"] is None


def test_plan_replays_steps_and_chains_on_placed_part_sockets(
    client: TestClient,
) -> None:
    plan = _plan(("leg_left", "hip_left"), ("shoe_left", "leg_left_1__ankle"))
    data = client.post(ASSEMBLY, json=plan).json()
    placed = data["placed"]
    assert [p["instance_id"] for p in placed] == [
        "humanoid_torso_0",
        "leg_left_1",
        "shoe_left_2",
    ]
    assert placed[2]["host_instance"] == "leg_left_1"
    sockets = {s["name"] for s in data["free_sockets"]}
    assert "hip_left" not in sockets
    assert "leg_left_1__ankle" not in sockets
    link_names = {n["name"] for n in data["model"]["tree"] if n["node_type"] != "joint"}
    assert "leg_left_1__thigh_left" in link_names
    assert data["validation"]["ok"] is True


@pytest.mark.parametrize(
    ("part_id", "port", "fragment"),
    [
        ("leg_right", "hip_left", "side mismatch"),
        ("head", "hip_left", "type mismatch"),
        ("leg_left", "no_such_port", "unknown host port"),
    ],
)
def test_candidate_drop_is_evaluated_without_mutating(
    client: TestClient, part_id: str, port: str, fragment: str
) -> None:
    plan = _plan(candidate={"part_id": part_id, "host_port": port})
    data = client.post(ASSEMBLY, json=plan).json()
    assert data["drop"]["accepted"] is False
    assert fragment in data["drop"]["reason"]
    assert len(data["placed"]) == 1


def test_compatible_candidate_is_accepted(client: TestClient) -> None:
    plan = _plan(candidate={"part_id": "leg_left", "host_port": "hip_left"})
    drop = client.post(ASSEMBLY, json=plan).json()["drop"]
    assert drop == {
        "accepted": True,
        "reason": "compatible",
        "part_id": "leg_left",
        "host_port": "hip_left",
        "findings": drop["findings"],
    }


def test_rejected_plan_step_is_a_422_naming_the_step(client: TestClient) -> None:
    plan = _plan(("leg_left", "hip_left"), ("leg_left", "hip_left"))
    response = client.post(ASSEMBLY, json=plan)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "step 2" in detail and "occupied" in detail


def test_unknown_base_part_is_a_404(client: TestClient) -> None:
    response = client.post(ASSEMBLY, json={"base_part_id": "nope", "steps": []})
    assert response.status_code == 404


def test_export_urdf_embeds_the_assembly_record(client: TestClient) -> None:
    response = client.post(EXPORT, json=_plan(("head", "neck"), format="urdf"))
    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "urdf"
    assert data["content"].lstrip().startswith("<")
    assert "ud_assembly" in data["content"]
    assert "head_1__" in data["content"]
    assert data["validation"]["ok"] is True


def test_export_mjcf_converts_the_assembly(client: TestClient) -> None:
    response = client.post(EXPORT, json=_plan(("head", "neck"), format="mjcf"))
    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "mjcf"
    assert "<mujoco" in data["content"]


def test_export_rejects_unknown_format(client: TestClient) -> None:
    response = client.post(EXPORT, json=_plan(format="sdf"))
    assert response.status_code == 422


def test_parts_path_is_not_shadowed_by_the_model_name_route() -> None:
    """``/tools/model-explorer/{model_name}`` must not swallow ``/parts``."""
    app = FastAPI()
    register_routes(app, prefix="")
    first = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute)
        and "GET" in (route.methods or ())
        and route.path_regex.match(PARTS)
    )
    assert first.endpoint.__module__ == "src.api.routes.model_explorer_assembly"
