"""Characterization tests for model-pack parsing (#12846).

Pin ``ModelPackEntry.from_dict`` validation order/messages and the
``_normalize_model_pack_v1`` exercise expansion so both can be decomposed
into helpers without any behaviour change.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.config.model_pack_manifest import (
    ModelPackEntry,
    _normalize_model_pack_v1,
)


def _base(**extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": " a ",
        "name": " n ",
        "description": " d ",
        "type": " t ",
        "path": " p ",
    }
    data.update(extra)
    return data


@pytest.mark.unit
def test_entry_minimal_defaults_and_strip() -> None:
    e = ModelPackEntry.from_dict(_base())
    assert (e.id, e.name, e.description, e.type, e.path) == ("a", "n", "d", "t", "p")
    assert e.order == 99 and e.hidden is False
    assert e.engine_type is None and e.provider is None
    assert e.exercise is None and e.preset_params == {}
    assert e.exchange_artifacts == () and e.identity is None


@pytest.mark.unit
def test_entry_all_optional_fields_round_trip() -> None:
    e = ModelPackEntry.from_dict(
        _base(
            engine_type=" mujoco ",
            provider=" prov ",
            source_root=" root ",
            working_dir=" wd ",
            order=3,
            hidden=True,
            hidden_reason=" why ",
            hidden_owner=" me ",
            embed_adapter=" ad ",
            exercise=7,
            preset_params={"x": 1},
            capabilities=["a", "b"],
            tags=["t"],
            python_paths=["src"],
        )
    )
    assert e.engine_type == "mujoco" and e.provider == "prov"
    assert e.source_root == "root" and e.working_dir == "wd"
    assert e.order == 3 and e.hidden is True
    assert (e.hidden_reason, e.hidden_owner) == ("why", "me")
    assert e.embed_adapter == "ad" and e.exercise == "7"
    assert e.preset_params == {"x": 1}


@pytest.mark.unit
def test_entry_hidden_false_keeps_stripped_reason() -> None:
    e = ModelPackEntry.from_dict(_base(hidden_reason=" r ", hidden_owner=" o "))
    assert e.hidden is False and e.hidden_reason == "r" and e.hidden_owner == "o"


@pytest.mark.unit
def test_entry_non_dict_preset_params_ignored() -> None:
    assert ModelPackEntry.from_dict(_base(preset_params=[1])).preset_params == {}


@pytest.mark.unit
def test_entry_missing_fields_message() -> None:
    with pytest.raises(ValueError, match=r"missing required fields: \['id', 'name'\]"):
        ModelPackEntry.from_dict({"description": "d", "type": "t", "path": "p"})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"engine_type": ""}, "engine_type must be a non-empty string"),
        ({"provider": 3}, "provider must be a non-empty string"),
        ({"source_root": " "}, "source_root must be a non-empty string"),
        ({"working_dir": ""}, "working_dir must be a non-empty string"),
        ({"order": "1"}, "order must be an integer"),
        ({"order": -1}, "order must be non-negative"),
        ({"hidden": "yes"}, "hidden must be a boolean"),
        ({"hidden": True}, "hidden_reason must be a non-empty string"),
        (
            {"hidden": True, "hidden_reason": "r"},
            "hidden_owner must be a non-empty string",
        ),
        ({"exchange_artifacts": {}}, "exchange_artifacts must be a list"),
        ({"embed_adapter": ""}, "embed_adapter must be a non-empty string"),
    ],
)
def test_entry_validation_errors(extra: dict[str, Any], message: str) -> None:
    with pytest.raises(Exception, match=message):
        ModelPackEntry.from_dict(_base(**extra))


@pytest.mark.unit
def test_entry_validation_order_first_failure_wins() -> None:
    with pytest.raises(Exception, match="engine_type must be"):
        ModelPackEntry.from_dict(_base(engine_type="", provider="", order=-5))
    with pytest.raises(Exception, match="order must be non-negative"):
        ModelPackEntry.from_dict(_base(order=-5, hidden="x"))


@pytest.mark.unit
def test_entry_required_field_type_and_empty() -> None:
    with pytest.raises(Exception, match="must be a string"):
        ModelPackEntry.from_dict(_base(name=5))
    with pytest.raises(Exception, match="must be non-empty"):
        ModelPackEntry.from_dict(_base(path="  "))
    with pytest.raises(Exception, match="model entry must be a mapping"):
        ModelPackEntry.from_dict([])  # type: ignore[arg-type]


@pytest.mark.unit
def test_v1_exercises_expand_with_models_root_and_defaults() -> None:
    out = _normalize_model_pack_v1(
        {
            "repo": "Repo",
            "engine": "mujoco",
            "format": "mjcf",
            "models_root": "models" + chr(92) + "sub/",
            "exercises": [
                {"id": "squat_up", "path": "squat.xml"},
                {"id": "dl", "path": "models/sub/dl.xml", "name": "N", "tags": ["x"]},
                {"id": "c", "path": "/c.xml", "capabilities": ["k"], "type": "T"},
            ],
        }
    )
    assert out["pack_id"] == "Repo" and out["pack_name"] == "Repo"
    assert out["provider"] == "mujoco" and out["manifest_version"] == "1.0.0"
    assert out["source_repo"] == "Repo" and out["description"] == ""
    m0, m1, m2 = out["models"]
    assert m0 == {
        "id": "Repo-squat_up",
        "name": "Squat Up",
        "description": "mujoco squat_up model",
        "type": "mjcf",
        "path": "models/sub/squat.xml",
        "engine_type": "mujoco",
        "capabilities": ["biomechanics", "mujoco", "squat_up"],
    }
    assert m1["path"] == "models/sub/dl.xml" and m1["name"] == "N"
    assert m1["tags"] == ["x"]
    assert m2["path"] == "models/sub/c.xml" and m2["capabilities"] == ["k"]
    assert m2["type"] == "T"


@pytest.mark.unit
def test_v1_models_passthrough_and_provider_fallbacks() -> None:
    out = _normalize_model_pack_v1({"repo": "R", "models": [{"id": "x"}]})
    assert out["models"] == [{"id": "x"}] and out["provider"] == "R"
    out2 = _normalize_model_pack_v1(
        {"provider": "P", "package": "pk", "exercises": [{"id": "e", "path": "p"}]}
    )
    assert out2["models"][0]["id"] == "pk-e"
    assert out2["models"][0]["engine_type"] is None
    assert out2["models"][0]["type"] == "model"
    assert out2["models"][0]["description"] == "P e model"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "missing a provider"),
        ({"repo": "r"}, "must declare either 'models' or 'exercises'"),
        ({"repo": "r", "exercises": {}}, "'exercises' must be a list"),
        ({"repo": "r", "exercises": [1]}, "exercise entries must be mappings"),
        ({"repo": "r", "exercises": [{"id": "a"}]}, "require non-empty 'id'"),
    ],
)
def test_v1_errors(data: dict[str, Any], message: str) -> None:
    with pytest.raises(Exception, match=message):
        _normalize_model_pack_v1(data)
