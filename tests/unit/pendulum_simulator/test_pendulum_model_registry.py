"""Tests for src.shared.python.pendulum_simulator.model_registry (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.pendulum_simulator.model_registry import (
    ModelConfig,
    clear_registry,
    get_model,
    list_models,
    register_model,
)


def _make_config(name: str = "TestModel", n_dof: int = 2) -> ModelConfig:
    return ModelConfig(
        name=name,
        n_dof=n_dof,
        state_size=n_dof * 2,
        param_class=object,
        simulation_runner=lambda: None,
        result_class=object,
        description="Test model",
    )


class TestModelConfig:
    def test_pendulum_model_registry_instantiates(self) -> None:
        cfg = _make_config()
        assert cfg is not None

    def test_n_dof_stored(self) -> None:
        cfg = _make_config(n_dof=3)
        assert cfg.n_dof == 3

    def test_state_size_stored(self) -> None:
        cfg = _make_config(n_dof=4)
        assert cfg.state_size == 8

    def test_description_stored(self) -> None:
        cfg = _make_config()
        assert isinstance(cfg.description, str)

    def test_extra_defaults_empty(self) -> None:
        cfg = _make_config()
        assert cfg.extra == {}


class TestRegisterAndGet:
    def setup_method(self) -> None:
        # Save existing state; clear for isolation
        clear_registry()

    def teardown_method(self) -> None:
        clear_registry()

    def test_register_and_retrieve(self) -> None:
        cfg = _make_config("MyModel")
        register_model("MyModel", cfg)
        retrieved = get_model("MyModel")
        assert retrieved is cfg

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_model("nonexistent")

    def test_list_models_sorted(self) -> None:
        register_model("Zebra", _make_config("Zebra"))
        register_model("Alpha", _make_config("Alpha"))
        models = list_models()
        assert models == sorted(models)

    def test_list_models_contains_registered(self) -> None:
        register_model("TestA", _make_config("TestA"))
        assert "TestA" in list_models()

    def test_clear_registry_empties_list(self) -> None:
        register_model("Temp", _make_config("Temp"))
        clear_registry()
        assert list_models() == []

    def test_register_empty_name_raises(self) -> None:
        with pytest.raises(AssertionError):
            register_model("", _make_config())

    def test_register_wrong_type_raises(self) -> None:
        with pytest.raises(AssertionError):
            register_model("bad", {"not": "a ModelConfig"})  # type: ignore[arg-type]

    def test_list_models_returns_list(self) -> None:
        assert isinstance(list_models(), list)
