"""Unit tests for the ``model_pack/v1`` JSON Schema.

Covers a minimal valid manifest for every supported engine and a
representative set of invalid payloads. See
``docs/adr/0014-shared-biomech-models.md`` (UpstreamDrift#5184).
"""

from __future__ import annotations

import copy
import json

import pytest

from src.shared.python.biomech.schemas import (
    MODEL_PACK_V1_SCHEMA_PATH,
    load_model_pack_v1_schema,
)

pytestmark = pytest.mark.unit


jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(scope="module")
def schema() -> dict:
    return load_model_pack_v1_schema()


@pytest.fixture(scope="module")
def validator(schema: dict) -> object:
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _minimal_model_pack(engine: str, fmt: str) -> dict:
    return {
        "schema": "model_pack/v1",
        "repo": f"{engine.capitalize()}_Models",
        "package": f"{engine}_models",
        "engine": engine,
        "engine_version": ">=1.0,<2",
        "anthropometrics": "winter_2009",
        "format": fmt,
        "models_root": "src/pkg/exercises",
        "exercises": [
            {"id": "squat", "path": "src/pkg/exercises/squat"},
        ],
    }


def _minimal_tool_pack() -> dict:
    return {
        "schema": "tool_pack/v1",
        "repo": "Movement-Optimizer",
        "package": "movement_optimizer",
        "role": "optimizer",
        "formulation": "lagrangian",
        "muscle_model": "hill",
        "plane": "sagittal",
        "links": 3,
        "anthropometrics": "winter_2009",
        "supported_exercises": ["squat", "deadlift"],
        "consumes_models_from": ["mujoco_models", "drake_models"],
        "produces": ["trajectories", "muscle_activations"],
    }


@pytest.mark.parametrize(
    "engine,fmt",
    [
        ("mujoco", "mjcf"),
        ("drake", "sdf"),
        ("pinocchio", "urdf"),
        ("opensim", "osim"),
    ],
)
def test_minimal_model_pack_is_valid(
    engine: str,
    fmt: str,
    validator: object,
) -> None:
    """A minimal model pack for each engine validates clean."""
    payload = _minimal_model_pack(engine, fmt)
    validator.validate(payload)  # type: ignore[attr-defined]


def test_minimal_tool_pack_is_valid(validator: object) -> None:
    """A minimal tool pack validates clean."""
    validator.validate(_minimal_tool_pack())  # type: ignore[attr-defined]


def test_schema_id_and_draft(schema: dict) -> None:
    """The schema declares draft 2020-12 and a stable $id."""
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert "model_pack" in schema.get("$id", "")


def test_model_pack_with_optional_fields_is_valid(validator: object) -> None:
    """Optional ``axis_convention`` and ``addons`` are accepted."""
    payload = _minimal_model_pack("mujoco", "mjcf")
    payload["axis_convention"] = "x_forward_z_up"
    payload["addons"] = ["soft_tissue", "contact_softening"]
    validator.validate(payload)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "mutator,reason",
    [
        (lambda p: p.pop("schema"), "schema missing"),
        (lambda p: p.__setitem__("schema", "model_pack/v2"), "wrong schema enum"),
        (lambda p: p.__setitem__("engine", "physx"), "unknown engine"),
        (lambda p: p.__setitem__("format", "blend"), "unknown format"),
        (lambda p: p.__setitem__("exercises", []), "empty exercises"),
        (
            lambda p: p.__setitem__("exercises", [{"id": "Squat", "path": "a"}]),
            "exercise id pattern",
        ),
        (lambda p: p.__setitem__("package", "Bad-Name"), "package pattern"),
    ],
)
def test_model_pack_rejects_bad_payloads(
    mutator: object,
    reason: str,
    validator: object,
) -> None:
    """The schema rejects the mutated invalid payload."""
    payload = _minimal_model_pack("mujoco", "mjcf")
    mutator(payload)  # type: ignore[operator]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "mutator,reason",
    [
        (lambda p: p.__setitem__("role", "unknown_role"), "unknown role"),
        (lambda p: p.__setitem__("plane", "diagonal"), "unknown plane"),
        (lambda p: p.__setitem__("links", 0), "links min 1"),
        (lambda p: p.__setitem__("supported_exercises", []), "empty exercises"),
        (lambda p: p.__setitem__("produces", []), "empty produces"),
    ],
)
def test_tool_pack_rejects_bad_payloads(
    mutator: object,
    reason: str,
    validator: object,
) -> None:
    payload = _minimal_tool_pack()
    mutator(payload)  # type: ignore[operator]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)  # type: ignore[attr-defined]


def test_schema_file_is_parseable_json() -> None:
    """Direct JSON load of the bundled schema file succeeds."""
    json.loads(MODEL_PACK_V1_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_model_pack_schema_discriminator_blocks_tool_pack_fields(
    validator: object,
) -> None:
    """A ``model_pack/v1`` payload missing engine fields is rejected."""
    payload = copy.deepcopy(_minimal_model_pack("mujoco", "mjcf"))
    payload.pop("engine")
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)  # type: ignore[attr-defined]


def test_tool_pack_schema_discriminator_blocks_model_pack_fields(
    validator: object,
) -> None:
    """A ``tool_pack/v1`` payload missing role is rejected."""
    payload = copy.deepcopy(_minimal_tool_pack())
    payload.pop("role")
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)  # type: ignore[attr-defined]
