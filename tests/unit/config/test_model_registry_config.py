"""Tests for src.shared.python.config.model_registry (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.shared.python.config.model_registry import ModelConfig, ModelRegistry

# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_stores_required_fields(self) -> None:
        mc = ModelConfig(
            id="ball",
            name="Golf Ball",
            description="Simple ball model",
            type="mjcf",
            path="models/ball.xml",
        )
        assert mc.id == "ball"
        assert mc.name == "Golf Ball"
        assert mc.type == "mjcf"
        assert mc.path == "models/ball.xml"

    def test_engine_type_defaults_none(self) -> None:
        mc = ModelConfig(
            id="x",
            name="X",
            description="",
            type="urdf",
            path="x.urdf",
        )
        assert mc.engine_type is None

    def test_engine_type_can_be_set(self) -> None:
        mc = ModelConfig(
            id="x",
            name="X",
            description="",
            type="urdf",
            path="x.urdf",
            engine_type="mujoco",
        )
        assert mc.engine_type == "mujoco"


# ---------------------------------------------------------------------------
# ModelRegistry — missing file
# ---------------------------------------------------------------------------


class TestModelRegistryMissingFile:
    def test_missing_file_does_not_raise(self) -> None:
        # A missing config path should log a warning, not raise
        registry = ModelRegistry("/nonexistent/path/models.yaml")
        assert registry.models == {}

    def test_get_model_returns_none_when_empty(self) -> None:
        registry = ModelRegistry("/nonexistent/path/models.yaml")
        assert registry.get_model("anything") is None

    def test_get_all_models_empty(self) -> None:
        registry = ModelRegistry("/nonexistent/path/models.yaml")
        assert registry.get_all_models() == []


# ---------------------------------------------------------------------------
# ModelRegistry — YAML loading
# ---------------------------------------------------------------------------


def _write_registry(models: list[dict], tmp_dir: Path) -> Path:
    path = tmp_dir / "models.yaml"
    with open(path, "w") as f:
        yaml.dump({"models": models}, f)
    return path


class TestModelRegistryLoad:
    def test_loads_single_model(self, tmp_path: Path) -> None:
        path = _write_registry(
            [
                {
                    "id": "ball",
                    "name": "Ball",
                    "description": "ball",
                    "type": "mjcf",
                    "path": "b.xml",
                }
            ],
            tmp_path,
        )
        registry = ModelRegistry(path)
        assert "ball" in registry.models

    def test_get_model_returns_model_config(self, tmp_path: Path) -> None:
        path = _write_registry(
            [
                {
                    "id": "club",
                    "name": "Club",
                    "description": "club",
                    "type": "urdf",
                    "path": "c.urdf",
                }
            ],
            tmp_path,
        )
        registry = ModelRegistry(path)
        mc = registry.get_model("club")
        assert mc is not None
        assert mc.name == "Club"

    def test_get_model_missing_returns_none(self, tmp_path: Path) -> None:
        path = _write_registry(
            [
                {
                    "id": "ball",
                    "name": "Ball",
                    "description": "",
                    "type": "mjcf",
                    "path": "b.xml",
                }
            ],
            tmp_path,
        )
        registry = ModelRegistry(path)
        assert registry.get_model("nonexistent") is None

    def test_get_all_models_returns_list(self, tmp_path: Path) -> None:
        path = _write_registry(
            [
                {
                    "id": "a",
                    "name": "A",
                    "description": "",
                    "type": "mjcf",
                    "path": "a.xml",
                },
                {
                    "id": "b",
                    "name": "B",
                    "description": "",
                    "type": "urdf",
                    "path": "b.urdf",
                },
            ],
            tmp_path,
        )
        registry = ModelRegistry(path)
        models = registry.get_all_models()
        assert len(models) == 2

    def test_get_models_by_type(self, tmp_path: Path) -> None:
        path = _write_registry(
            [
                {
                    "id": "a",
                    "name": "A",
                    "description": "",
                    "type": "mjcf",
                    "path": "a.xml",
                },
                {
                    "id": "b",
                    "name": "B",
                    "description": "",
                    "type": "urdf",
                    "path": "b.urdf",
                },
                {
                    "id": "c",
                    "name": "C",
                    "description": "",
                    "type": "mjcf",
                    "path": "c.xml",
                },
            ],
            tmp_path,
        )
        registry = ModelRegistry(path)
        mjcf_models = registry.get_models_by_type("mjcf")
        assert len(mjcf_models) == 2
        assert all(m.type == "mjcf" for m in mjcf_models)

    def test_empty_yaml_no_crash(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text("")
        registry = ModelRegistry(path)
        assert registry.models == {}

    def test_malformed_model_skipped(self, tmp_path: Path) -> None:
        # Model missing required fields should be skipped, not crash
        path = tmp_path / "models.yaml"
        with open(path, "w") as f:
            yaml.dump(
                {
                    "models": [
                        {"id": "bad"},  # missing name, description, type, path
                        {
                            "id": "good",
                            "name": "G",
                            "description": "",
                            "type": "mjcf",
                            "path": "g.xml",
                        },
                    ]
                },
                f,
            )
        registry = ModelRegistry(path)
        # "good" should still be loaded
        assert registry.get_model("good") is not None

    def test_config_path_stored_as_path(self, tmp_path: Path) -> None:
        path = _write_registry([], tmp_path)
        registry = ModelRegistry(path)
        assert isinstance(registry.config_path, Path)

    def test_engine_type_preserved(self, tmp_path: Path) -> None:
        path = _write_registry(
            [
                {
                    "id": "robot",
                    "name": "Robot",
                    "description": "",
                    "type": "urdf",
                    "path": "r.urdf",
                    "engine_type": "pinocchio",
                }
            ],
            tmp_path,
        )
        registry = ModelRegistry(path)
        mc = registry.get_model("robot")
        assert mc is not None
        assert mc.engine_type == "pinocchio"
