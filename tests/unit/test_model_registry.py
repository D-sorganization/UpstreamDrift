"""Unit tests for ModelRegistry.

TEST-004: Added @pytest.mark.unit markers for test categorization.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from src.shared.python.config.model_registry import ModelRegistry


@pytest.mark.unit
class TestModelRegistry:
    """Test cases for ModelRegistry."""

    def test_load_valid_registry(self) -> None:
        """Test loading a valid model registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            config_data = {
                "models": [
                    {
                        "id": "test_model",
                        "name": "Test Model",
                        "description": "A test model",
                        "type": "mjcf",
                        "path": "engines/test/model.xml",
                    }
                ]
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            registry = ModelRegistry(config_path)
            assert len(registry.models) == 1

            model = registry.get_model("test_model")
            assert model is not None
            assert model.id == "test_model"
            assert model.name == "Test Model"
            assert model.type == "mjcf"

    def test_load_empty_registry_file(self) -> None:
        """Test loading an empty registry file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            config_path.write_text("", encoding="utf-8")

            registry = ModelRegistry(config_path)
            assert len(registry.models) == 0

    def test_load_missing_registry(self) -> None:
        """Test loading when registry file doesn't exist."""
        # Using a path that definitely doesn't exist
        registry = ModelRegistry(Path("/nonexistent/path/models.yaml"))
        assert len(registry.models) == 0

    def test_load_malformed_yaml(self) -> None:
        """Test loading a malformed YAML file raises yaml.YAMLError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            # Write invalid YAML (tab character instead of spaces)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("models:\n\t- id: test")

            with pytest.raises(yaml.YAMLError):
                ModelRegistry(config_path)

    def test_load_invalid_model_format(self) -> None:
        """Test loading registry with invalid model structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            # Missing required fields like 'name', 'type'
            config_data = {
                "models": [
                    {
                        "id": "bad_model",
                        # Missing name, type, path
                    }
                ]
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            registry = ModelRegistry(config_path)
            # Should skip the bad model
            assert len(registry.models) == 0

    def test_get_all_models(self) -> None:
        """Test retrieving all models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            config_data = {
                "models": [
                    {
                        "id": "m1",
                        "name": "Model 1",
                        "description": "D1",
                        "type": "mjcf",
                        "path": "p1",
                    },
                    {
                        "id": "m2",
                        "name": "Model 2",
                        "description": "D2",
                        "type": "urdf",
                        "path": "p2",
                    },
                ]
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            registry = ModelRegistry(config_path)
            models = registry.get_all_models()
            assert len(models) == 2
            assert {m.id for m in models} == {"m1", "m2"}

    def test_get_models_by_type(self) -> None:
        """Test filtering models by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            config_data = {
                "models": [
                    {
                        "id": "m1",
                        "name": "M1",
                        "description": "D1",
                        "type": "mjcf",
                        "path": "p1",
                    },
                    {
                        "id": "m2",
                        "name": "M2",
                        "description": "D2",
                        "type": "drake",
                        "path": "p2",
                    },
                    {
                        "id": "m3",
                        "name": "M3",
                        "description": "D3",
                        "type": "mjcf",
                        "path": "p3",
                    },
                ]
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            registry = ModelRegistry(config_path)
            mjcf_models = registry.get_models_by_type("mjcf")
            assert len(mjcf_models) == 2
            assert {m.id for m in mjcf_models} == {"m1", "m3"}

            drake_models = registry.get_models_by_type("drake")
            assert len(drake_models) == 1
            assert drake_models[0].id == "m2"

    def test_load_provider_manifest_from_configured_roots(self):
        """Test loading external provider manifests via env-configured roots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("models: []", encoding="utf-8")

            provider_root = root / "providers" / "Drake_Models"
            provider_root.mkdir(parents=True)
            provider_manifest = provider_root / "model_pack.yaml"
            provider_manifest.write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "drake-models",
                        "pack_name": "Drake Models",
                        "provider": "drake_models",
                        "models": [
                            {
                                "id": "external_drake",
                                "name": "External Drake",
                                "description": "Provider model",
                                "type": "drake",
                                "path": "models/external.urdf",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {"UPSTREAM_DRIFT_PROVIDER_ROOTS": str(provider_root)},
                clear=False,
            ):
                registry = ModelRegistry(config_path)

            model = registry.get_model("external_drake")
            assert model is not None
            assert model.provider == "drake_models"
            assert model.source_root == str(provider_root)

    def test_load_provider_manifest_preserves_cross_engine_metadata(self):
        """Provider metadata should survive registry loading unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("models: []", encoding="utf-8")

            provider_root = root / "providers" / "MuJoCo_Models"
            provider_root.mkdir(parents=True)
            provider_manifest = provider_root / "model_pack.yaml"
            provider_manifest.write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "mujoco-models",
                        "pack_name": "MuJoCo Models",
                        "provider": "mujoco_models",
                        "models": [
                            {
                                "id": "external_mujoco",
                                "name": "External MuJoCo",
                                "description": "Provider model",
                                "type": "mjcf",
                                "path": "models/external.xml",
                                "identity": {
                                    "canonical_id": "golf.swing.main",
                                    "motion_family": "Golf Swing",
                                    "exercise": "Driver Full Swing",
                                    "humanoid": "Golf Athlete",
                                },
                                "capabilities": ["inverse-kinematics"],
                                "exchange_artifacts": [
                                    {
                                        "format": "urdf",
                                        "path": "exports/external.urdf",
                                        "role": "derived",
                                    }
                                ],
                                "provenance": {
                                    "source_format": "mjcf",
                                    "source_path": "models/external.xml",
                                    "version": "2026.04",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {"UPSTREAM_DRIFT_PROVIDER_ROOTS": str(provider_root)},
                clear=False,
            ):
                registry = ModelRegistry(config_path)

            model = registry.get_model("external_mujoco")
            assert model is not None
            assert model.identity is not None
            assert model.identity.canonical_id == "golf.swing.main"
            assert model.capabilities == ("ik",)
            assert model.exchange_artifacts[0].format == "urdf"
            assert model.provenance is not None
            assert model.provenance.version == "2026.04"

    def test_load_provider_manifest_preserves_launcher_metadata(self):
        """Provider launcher metadata should survive registry loading unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("models: []", encoding="utf-8")

            provider_root = root / "providers" / "Drake_Models"
            provider_root.mkdir(parents=True)
            provider_manifest = provider_root / "model_pack.yaml"
            provider_manifest.write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "drake-models",
                        "pack_name": "Drake Models",
                        "provider": "drake_models",
                        "models": [
                            {
                                "id": "external_drake",
                                "name": "External Drake",
                                "description": "Provider model",
                                "type": "drake",
                                "path": "models/external.urdf",
                                "launcher": {
                                    "category": "physics_engine",
                                    "logo": "drake.svg",
                                    "status": "provider_ready",
                                    "web_route": "/providers/drake",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {"UPSTREAM_DRIFT_PROVIDER_ROOTS": str(provider_root)},
                clear=False,
            ):
                registry = ModelRegistry(config_path)

            model = registry.get_model("external_drake")
            assert model is not None
            assert model.launcher is not None
            assert model.launcher.logo == "drake.svg"
            assert model.launcher.web_route == "/providers/drake"

    def test_discovers_known_sibling_provider_repos_without_env(self):
        """Known engine-model sibling repos should be discovered without env wiring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            repo_root = workspace_root / "UpstreamDrift"
            config_path = repo_root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("models: []", encoding="utf-8")

            provider_layout = {
                "MuJoCo_Models": (
                    "mujoco_models",
                    "external_mujoco",
                    "mjcf",
                    "mujoco",
                ),
                "Drake_Models": (
                    "drake_models",
                    "external_drake",
                    "urdf",
                    "drake",
                ),
                "Pinocchio_Models": (
                    "pinocchio_models",
                    "external_pinocchio",
                    "urdf",
                    "pinocchio",
                ),
                "OpenSim_Models": (
                    "opensim_models",
                    "external_opensim",
                    "osim",
                    "opensim",
                ),
            }

            for (
                repo_name,
                (provider_id, model_id, model_type, engine_type),
            ) in provider_layout.items():
                provider_root = workspace_root / repo_name
                provider_root.mkdir(parents=True)
                (provider_root / "model_pack.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "manifest_version": "1.0.0",
                            "pack_id": provider_id.replace("_", "-"),
                            "pack_name": repo_name,
                            "provider": provider_id,
                            "models": [
                                {
                                    "id": model_id,
                                    "name": model_id.replace("_", " ").title(),
                                    "description": f"{repo_name} model",
                                    "type": model_type,
                                    "path": f"models/{model_id}.{model_type}",
                                    "engine_type": engine_type,
                                    "capabilities": ["rigid_body"],
                                    "identity": {
                                        "canonical_id": f"demo.{model_id}",
                                        "motion_family": "demo",
                                        "exercise": model_id,
                                        "humanoid": "athlete",
                                    },
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with patch.dict("os.environ", {}, clear=False):
                registry = ModelRegistry(config_path)

            assert registry.get_model("external_mujoco") is not None
            assert registry.get_model("external_drake") is not None
            assert registry.get_model("external_pinocchio") is not None
            assert registry.get_model("external_opensim") is not None

    def test_missing_known_sibling_provider_repos_do_not_break_registry(self):
        """Absent sibling provider repos should not prevent registry startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            repo_root = workspace_root / "UpstreamDrift"
            config_path = repo_root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("models: []", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=False):
                registry = ModelRegistry(config_path)

            assert registry.get_all_models() == []

    def test_discovers_utility_provider_repos_without_env(self):
        """Utility sibling repos should flow through the shared registry too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            repo_root = workspace_root / "UpstreamDrift"
            config_path = repo_root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("models: []", encoding="utf-8")

            tools_root = workspace_root / "Tools"
            tools_root.mkdir(parents=True)
            (tools_root / "model_pack.yaml").write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "tools-pack",
                        "pack_name": "Tools",
                        "provider": "tools",
                        "models": [
                            {
                                "id": "pendulum_suite",
                                "name": "Pendulum Suite",
                                "description": "Pendulum workflows",
                                "type": "special_app",
                                "path": "src/pendulum_launcher.py",
                                "capabilities": ["pendulum", "simulation"],
                                "launcher": {
                                    "category": "tool",
                                    "logo": "golf_logo.svg",
                                    "status": "utility",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            optimizer_root = workspace_root / "Movement-Optimizer"
            optimizer_root.mkdir(parents=True)
            (optimizer_root / "model_pack.yaml").write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "movement-optimizer-pack",
                        "pack_name": "Movement Optimizer",
                        "provider": "movement_optimizer",
                        "models": [
                            {
                                "id": "movement_optimizer_cli",
                                "name": "Movement Optimizer",
                                "description": "Optimization utility",
                                "type": "special_app",
                                "path": "src/optimizer.py",
                                "capabilities": ["optimization", "trajectory"],
                                "launcher": {
                                    "category": "tool",
                                    "logo": "golf_logo.svg",
                                    "status": "utility",
                                    "web_route": "/tools/movement-optimizer",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=False):
                registry = ModelRegistry(config_path)

            pendulum = registry.get_model("pendulum_suite")
            optimizer = registry.get_model("movement_optimizer_cli")
            assert pendulum is not None
            assert pendulum.launcher is not None
            assert pendulum.launcher.category == "tool"
            assert optimizer is not None
            assert optimizer.launcher is not None
            assert optimizer.launcher.web_route == "/tools/movement-optimizer"

    def test_local_only_mode_ignores_provider_manifests(self):
        """Legacy mode should preserve local-only discovery during migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "models": [
                            {
                                "id": "local_model",
                                "name": "Local Model",
                                "description": "Local entry",
                                "type": "mjcf",
                                "path": "models/local.xml",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            provider_root = root / "providers" / "Drake_Models"
            provider_root.mkdir(parents=True)
            (provider_root / "model_pack.yaml").write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "drake-models",
                        "pack_name": "Drake Models",
                        "provider": "drake_models",
                        "models": [
                            {
                                "id": "provider_model",
                                "name": "Provider Model",
                                "description": "Provider entry",
                                "type": "urdf",
                                "path": "models/provider.urdf",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "UPSTREAM_DRIFT_PROVIDER_ROOTS": str(provider_root),
                    "UPSTREAM_DRIFT_DISCOVERY_MODE": "local-only",
                },
                clear=False,
            ):
                registry = ModelRegistry(config_path)

            assert registry.get_model("local_model") is not None
            assert registry.get_model("provider_model") is None

    def test_hybrid_mode_loads_local_and_provider_manifests(self):
        """Hybrid mode should merge legacy local models with provider packs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "models": [
                            {
                                "id": "local_model",
                                "name": "Local Model",
                                "description": "Local entry",
                                "type": "mjcf",
                                "path": "models/local.xml",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            provider_root = root / "providers" / "Drake_Models"
            provider_root.mkdir(parents=True)
            (provider_root / "model_pack.yaml").write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "drake-models",
                        "pack_name": "Drake Models",
                        "provider": "drake_models",
                        "models": [
                            {
                                "id": "provider_model",
                                "name": "Provider Model",
                                "description": "Provider entry",
                                "type": "urdf",
                                "path": "models/provider.urdf",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "UPSTREAM_DRIFT_PROVIDER_ROOTS": str(provider_root),
                    "UPSTREAM_DRIFT_DISCOVERY_MODE": "hybrid",
                },
                clear=False,
            ):
                registry = ModelRegistry(config_path)

            assert registry.get_model("local_model") is not None
            assert registry.get_model("provider_model") is not None

    def test_provider_manifest_preserves_symbolic_source_root(self):
        """Registered source aliases must not be treated as provider-relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("models: []\n", encoding="utf-8")

            provider_root = root / "providers" / "Movement-Optimizer"
            provider_root.mkdir(parents=True)
            (provider_root / "model_pack.yaml").write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "movement-optimizer-pack",
                        "pack_name": "Movement Optimizer",
                        "provider": "movement_optimizer",
                        "models": [
                            {
                                "id": "movement_optimizer_cli",
                                "name": "Movement Optimizer",
                                "description": "Optimization utility",
                                "type": "special_app",
                                "path": "src/optimizer.py",
                                "source_root": "movement_optimizer",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {"UPSTREAM_DRIFT_PROVIDER_ROOTS": str(provider_root)},
                clear=False,
            ):
                registry = ModelRegistry(config_path)

            model = registry.get_model("movement_optimizer_cli")

            assert model is not None
            assert model.source_root == "movement_optimizer"

    def test_provider_first_mode_prefers_provider_definition_on_duplicates(self):
        """Provider-first mode should let provider manifests override legacy duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "models": [
                            {
                                "id": "shared_model",
                                "name": "Legacy Shared",
                                "description": "Legacy entry",
                                "type": "mjcf",
                                "path": "models/local.xml",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            provider_root = root / "providers" / "Drake_Models"
            provider_root.mkdir(parents=True)
            (provider_root / "model_pack.yaml").write_text(
                yaml.safe_dump(
                    {
                        "manifest_version": "1.0.0",
                        "pack_id": "drake-models",
                        "pack_name": "Drake Models",
                        "provider": "drake_models",
                        "models": [
                            {
                                "id": "shared_model",
                                "name": "Provider Shared",
                                "description": "Provider entry",
                                "type": "urdf",
                                "path": "models/provider.urdf",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "UPSTREAM_DRIFT_PROVIDER_ROOTS": str(provider_root),
                    "UPSTREAM_DRIFT_DISCOVERY_MODE": "provider-first",
                },
                clear=False,
            ):
                registry = ModelRegistry(config_path)

            model = registry.get_model("shared_model")
            assert model is not None
            assert model.name == "Provider Shared"
            assert model.provider == "drake_models"

    def test_resolve_model_source_uses_shared_provider_policy(self):
        """ModelRegistry should expose the shared provider-backed source resolver."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "models": [
                            {
                                "id": "provider_model",
                                "name": "Provider Model",
                                "description": "Provider entry",
                                "type": "urdf",
                                "path": "models/provider.urdf",
                                "engine_type": "drake",
                                "capabilities": ["swing"],
                                "source_root": "providers/Drake_Models",
                                "working_dir": "python",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "providers" / "Drake_Models" / "python").mkdir(parents=True)

            registry = ModelRegistry(config_path)

            resolved = registry.resolve_model_source("provider_model", root)

            assert resolved.provider_id == "sibling-repo"
            assert (
                resolved.source_root == (root / "providers" / "Drake_Models").resolve()
            )
            assert (
                resolved.working_directory
                == (root / "providers" / "Drake_Models" / "python").resolve()
            )

    def test_get_engine_provider_paths_groups_models_by_engine_type(self):
        """Registry engine-path grouping should feed shared engine discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "src" / "config" / "models.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "models": [
                            {
                                "id": "provider_model",
                                "name": "Provider Model",
                                "description": "Provider entry",
                                "type": "urdf",
                                "path": "models/provider.urdf",
                                "engine_type": "drake",
                                "capabilities": ["swing"],
                                "source_root": "providers/Drake_Models",
                                "working_dir": "python",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "providers" / "Drake_Models" / "python").mkdir(parents=True)

            registry = ModelRegistry(config_path)

            grouped = registry.get_engine_provider_paths(root)

            assert grouped == {
                "drake": ((root / "providers" / "Drake_Models" / "python").resolve(),)
            }
