"""TDD tests for correct default path resolution in model registries (issue #2493).

Three bugs:
1. StandardModelManager computes suite_root as src/shared/ then appends "shared/urdf",
   producing src/shared/shared/urdf (double-shared).
2. src/shared/python/config/ModelRegistry defaults to "config/models.yaml" (relative),
   but the real manifest is at src/config/models.yaml.
3. src/launchers/ModelRegistry defaults to "config/models.yaml"; when load() is called
   with the repo root, it looks at root/"config/models.yaml" which doesn't exist.
"""

from __future__ import annotations


class TestStandardModelManagerPaths:
    """StandardModelManager must not produce double-shared path segments."""

    def test_default_models_dir_has_no_double_shared(self) -> None:
        """models_dir must not contain 'shared/shared' when constructed with defaults."""
        from src.shared.python.config.standard_models import StandardModelManager

        mgr = StandardModelManager()
        models_str = str(mgr.models_dir)
        assert (
            "shared" + ("/" if "/" in models_str else "\\") + "shared" not in models_str
        ), f"models_dir has double-shared segment: {mgr.models_dir}"

    def test_default_meshes_dir_has_no_double_shared(self) -> None:
        """meshes_dir must not contain 'shared/shared' when constructed with defaults."""
        from src.shared.python.config.standard_models import StandardModelManager

        mgr = StandardModelManager()
        meshes_str = str(mgr.meshes_dir)
        assert (
            "shared" + ("/" if "/" in meshes_str else "\\") + "shared" not in meshes_str
        ), f"meshes_dir has double-shared segment: {mgr.meshes_dir}"

    def test_default_config_file_under_models_dir(self) -> None:
        """config_file must be under models_dir (standard_models.yaml within the urdf tree)."""
        from src.shared.python.config.standard_models import StandardModelManager

        mgr = StandardModelManager()
        # config_file should be a child of models_dir, not in a double-shared subtree
        assert mgr.config_file.parent == mgr.models_dir, (
            f"config_file parent {mgr.config_file.parent} != models_dir {mgr.models_dir}"
        )


class TestSharedModelRegistryDefaultPath:
    """src/shared/python/config/ModelRegistry default config_path must point to src/config/models.yaml."""

    def test_default_config_path_is_absolute(self) -> None:
        """Default ModelRegistry() config_path must be absolute, not a bare relative string."""
        from src.shared.python.config.model_registry import ModelRegistry

        registry = ModelRegistry()
        assert registry.config_path.is_absolute(), (
            f"Default config_path is relative: {registry.config_path}. "
            "Relative paths break when the process CWD differs from the repo root."
        )

    def test_default_config_path_points_to_src_config_models_yaml(self) -> None:
        """Default ModelRegistry() must target src/config/models.yaml, not config/models.yaml."""
        from src.shared.python.config.model_registry import ModelRegistry

        registry = ModelRegistry()
        # The correct file lives under src/config/
        path_str = str(registry.config_path).replace("\\", "/")
        assert "src/config/models.yaml" in path_str, (
            f"Default config_path does not point to src/config/models.yaml: {registry.config_path}"
        )

    def test_default_config_path_file_exists(self) -> None:
        """Default ModelRegistry() config_path must exist on disk."""
        from src.shared.python.config.model_registry import ModelRegistry

        registry = ModelRegistry()
        assert registry.config_path.exists(), (
            f"Default config_path does not exist: {registry.config_path}"
        )


class TestLaunchersModelRegistryDefaultPath:
    """src/launchers/ModelRegistry default config_path must resolve to src/config/models.yaml."""

    def test_launchers_default_resolves_to_src_config(self) -> None:
        """launchers.ModelRegistry default must reference 'src/config/models.yaml', not 'config/models.yaml'."""
        from src.launchers.model_registry import ModelRegistry

        registry = ModelRegistry()
        config_str = str(registry.config_path).replace("\\", "/")
        # After the fix the path should contain src/config, not just config/models.yaml at root
        assert "src/config/models.yaml" in config_str or not config_str.endswith(
            "config/models.yaml"
        ), (
            f"launchers.ModelRegistry still defaults to bare 'config/models.yaml': {registry.config_path}"
        )

    def test_launchers_default_file_found_from_repo_root(self) -> None:
        """load() with repo root must find the models.yaml without falling back to a warning."""

        from src.launchers.base import REPO_ROOT
        from src.launchers.model_registry import ModelRegistry

        registry = ModelRegistry()
        full_path = REPO_ROOT / registry.config_path
        # If config_path is already absolute, use it directly; otherwise join with repo root
        resolved = (
            registry.config_path if registry.config_path.is_absolute() else full_path
        )
        assert resolved.exists(), (
            f"launchers.ModelRegistry config_path does not resolve to an existing file "
            f"from repo root: {resolved}"
        )
