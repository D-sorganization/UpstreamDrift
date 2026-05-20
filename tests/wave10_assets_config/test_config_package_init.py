"""Tests for src.shared.python.config package __init__ re-exports.

The repo's tests/conftest.py stubs ``src.shared.python.config`` as an empty
namespace before collection (see comments there), so we exercise __init__
by reloading it explicitly and inspecting its re-exports list.
"""

from __future__ import annotations

import importlib


def test_package_imports_cleanly() -> None:
    mod = importlib.import_module("src.shared.python.config")
    assert mod is not None


def _parse_all_from_init() -> list[str]:
    """Statically parse __all__ from the real __init__.py.

    The repo conftest replaces the package with an empty namespace stub, so we
    cannot import it directly to inspect ``__all__``. Parse the source instead.
    """
    import ast
    from pathlib import Path

    init_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "shared"
        / "python"
        / "config"
        / "__init__.py"
    )
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and isinstance(node.value, ast.List)
        ):
            return [
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, ast.List)
                ):
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
    return []


def test_all_public_names_can_be_imported_from_submodules() -> None:
    """Every name in __init__'s __all__ resolves from at least one submodule."""
    submodules = [
        "src.shared.python.config.config_utils",
        "src.shared.python.config.configuration_manager",
        "src.shared.python.config.environment",
        "src.shared.python.config.handedness_support",
        "src.shared.python.config.model_pack_manifest",
        "src.shared.python.config.model_registry",
        "src.shared.python.config.model_source_providers",
        "src.shared.python.config.provider_catalog",
        "src.shared.python.config.standard_models",
    ]
    mods = [importlib.import_module(name) for name in submodules]
    names = _parse_all_from_init()
    assert names, "Failed to parse __all__ from __init__.py"
    for name in names:
        assert any(hasattr(m, name) for m in mods), (
            f"{name} from __all__ is not provided by any submodule"
        )


def test_dataclasses_are_importable() -> None:
    from src.shared.python.config.configuration_manager import (
        ConfigurationManager,
        SimulationConfig,
    )
    from src.shared.python.config.model_registry import ModelConfig, ModelRegistry
    from src.shared.python.config.standard_models import StandardModelManager

    assert ConfigurationManager is not None
    assert SimulationConfig is not None
    assert ModelConfig is not None
    assert ModelRegistry is not None
    assert StandardModelManager is not None


def test_environment_helpers_are_callable() -> None:
    from src.shared.python.config.environment import (
        get_api_host,
        get_api_port,
        get_environment,
        get_log_level,
    )

    assert callable(get_api_host)
    assert callable(get_api_port)
    assert callable(get_environment)
    assert callable(get_log_level)


def test_handedness_helpers_are_importable() -> None:
    from src.shared.python.config.handedness_support import (
        Handedness,
        HandednessConverter,
        MirrorTransform,
        mirror_position,
    )

    assert Handedness is not None
    assert HandednessConverter is not None
    assert MirrorTransform is not None
    assert callable(mirror_position)


def test_provider_catalog_helpers_are_importable() -> None:
    from src.shared.python.config.provider_catalog import (
        ProviderRepoDefinition,
        iter_known_engine_provider_ids,
        iter_known_provider_ids,
    )

    assert ProviderRepoDefinition is not None
    assert callable(iter_known_engine_provider_ids)
    assert callable(iter_known_provider_ids)
