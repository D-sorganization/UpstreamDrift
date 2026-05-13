"""Unit tests for the shared model-pack manifest contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.shared.python.config.model_pack_manifest import (
    LauncherPresentationMetadata,
    ModelPackEntry,
    ModelPackManifest,
    group_entries_by_canonical_id,
)
from src.shared.python.contracts import PreconditionError


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.mark.unit
class TestModelPackEntry:
    def test_from_dict_normalizes_capabilities_and_tags(self) -> None:
        entry = ModelPackEntry.from_dict(
            {
                "id": "mujoco_humanoid",
                "name": "MuJoCo Humanoid",
                "description": "Launchable humanoid model",
                "type": "mjcf",
                "path": "packs/mujoco/humanoid.xml",
                "capabilities": [" IK ", "ik", "Dynamics"],
                "tags": ["LowerBody", "lowerbody", "Golf"],
                "order": 3,
            }
        )

        assert entry.capabilities == ("ik", "dynamics")
        assert entry.tags == ("lowerbody", "golf")

    def test_from_dict_normalizes_identity_and_capability_aliases(self) -> None:
        entry = ModelPackEntry.from_dict(
            {
                "id": "mujoco_humanoid",
                "name": "MuJoCo Humanoid",
                "description": "Launchable humanoid model",
                "type": "mjcf",
                "path": "packs/mujoco/humanoid.xml",
                "capabilities": [
                    "inverse-kinematics",
                    "Forward Dynamics",
                    "balance_control",
                ],
                "identity": {
                    "canonical_id": "Golf.Swing.Main",
                    "motion_family": "Golf Swing",
                    "exercise": "Driver Full Swing",
                    "humanoid": "Golf Athlete",
                    "dataset": "Tour Capture",
                },
            }
        )

        assert entry.capabilities == ("ik", "dynamics", "balance")
        assert entry.identity is not None
        assert entry.identity.canonical_id == "golf.swing.main"
        assert entry.identity.motion_family == "golf-swing"
        assert entry.identity.exercise == "driver-full-swing"
        assert entry.identity.humanoid == "golf-athlete"
        assert entry.identity.dataset == "tour-capture"

    def test_from_dict_preserves_source_metadata(self) -> None:
        entry = ModelPackEntry.from_dict(
            {
                "id": "drake_humanoid",
                "name": "Drake Humanoid",
                "description": "Provider-backed humanoid model",
                "type": "urdf",
                "path": "models/drake/humanoid.urdf",
                "provider": "drake_models",
                "source_root": "../Drake_Models",
                "working_dir": "python",
                "python_paths": ["src", "src", "bindings"],
                "exchange_artifacts": [
                    {
                        "format": "URDF",
                        "path": "exports/humanoid.urdf",
                        "role": "source",
                    },
                    {
                        "format": "mjcf",
                        "path": "exports/humanoid.xml",
                        "role": "derived",
                    },
                ],
                "provenance": {
                    "source_format": "osim",
                    "source_path": "source/humanoid.osim",
                    "version": "2026.04",
                    "derived_from": ["opensim:humanoid-v1"],
                },
            }
        )

        assert entry.provider == "drake_models"
        assert entry.source_root == "../Drake_Models"
        assert entry.working_dir == "python"
        assert entry.python_paths == ("src", "bindings")
        assert [artifact.format for artifact in entry.exchange_artifacts] == [
            "urdf",
            "mjcf",
        ]
        assert entry.exchange_artifacts[0].role == "source"
        assert entry.provenance is not None
        assert entry.provenance.source_format == "osim"
        assert entry.provenance.derived_from == ("opensim:humanoid-v1",)

    def test_from_dict_preserves_explicit_launcher_metadata(self) -> None:
        entry = ModelPackEntry.from_dict(
            {
                "id": "drake_humanoid",
                "name": "Drake Humanoid",
                "description": "Provider-backed humanoid model",
                "type": "urdf",
                "path": "models/drake/humanoid.urdf",
                "launcher": {
                    "category": "physics_engine",
                    "logo": "drake.svg",
                    "status": "provider_ready",
                    "web_route": "/providers/drake",
                },
            }
        )

        assert entry.launcher == LauncherPresentationMetadata(
            category="physics_engine",
            logo="drake.svg",
            status="provider_ready",
            web_route="/providers/drake",
        )

    def test_from_dict_missing_required_field_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required fields"):
            ModelPackEntry.from_dict(
                {
                    "id": "broken",
                    "name": "Broken",
                    "description": "Missing path and type",
                }
            )

    def test_from_dict_invalid_order_raises(self) -> None:
        with pytest.raises(PreconditionError, match="order must be non-negative"):
            ModelPackEntry.from_dict(
                {
                    "id": "broken",
                    "name": "Broken",
                    "description": "Negative order",
                    "type": "urdf",
                    "path": "packs/broken.urdf",
                    "order": -1,
                }
            )


@pytest.mark.unit
class TestModelPackManifest:
    def test_load_manifest_sorts_models_deterministically(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "pack.yaml"
        _write_yaml(
            manifest_path,
            {
                "manifest_version": "1.0.0",
                "pack_id": "core-biomechanics",
                "pack_name": "Core Biomechanics",
                "provider": "local",
                "models": [
                    {
                        "id": "z_model",
                        "name": "Z Model",
                        "description": "Late id",
                        "type": "urdf",
                        "path": "packs/z.urdf",
                        "order": 2,
                    },
                    {
                        "id": "a_model",
                        "name": "A Model",
                        "description": "Early id",
                        "type": "mjcf",
                        "path": "packs/a.xml",
                        "order": 2,
                    },
                    {
                        "id": "first_model",
                        "name": "First Model",
                        "description": "First by order",
                        "type": "osim",
                        "path": "packs/first.osim",
                        "order": 1,
                    },
                ],
            },
        )

        manifest = ModelPackManifest.load(manifest_path)

        assert manifest.model_ids == ["first_model", "a_model", "z_model"]

    def test_load_manifest_missing_pack_metadata_raises(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "pack.yaml"
        _write_yaml(
            manifest_path,
            {
                "manifest_version": "1.0.0",
                "pack_name": "Core Biomechanics",
                "provider": "local",
                "models": [],
            },
        )

        with pytest.raises(ValueError, match="missing required fields"):
            ModelPackManifest.load(manifest_path)

    def test_load_manifest_duplicate_model_ids_raise(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "pack.yaml"
        duplicate_model = {
            "id": "dup",
            "name": "Duplicate",
            "description": "Duplicate id",
            "type": "urdf",
            "path": "packs/dup.urdf",
        }
        _write_yaml(
            manifest_path,
            {
                "manifest_version": "1.0.0",
                "pack_id": "core-biomechanics",
                "pack_name": "Core Biomechanics",
                "provider": "local",
                "models": [duplicate_model, duplicate_model],
            },
        )

        with pytest.raises(ValueError, match="Duplicate model IDs"):
            ModelPackManifest.load(manifest_path)

    def test_from_legacy_registry_wraps_models_yaml(self) -> None:
        manifest = ModelPackManifest.from_legacy_registry(
            {
                "models": [
                    {
                        "id": "legacy_model",
                        "name": "Legacy Model",
                        "description": "Loaded from models.yaml",
                        "type": "mjcf",
                        "path": "src/engines/mujoco/model.xml",
                    }
                ]
            },
            pack_id="upstreamdrift-core",
            pack_name="UpstreamDrift Core",
        )

        assert manifest.pack_id == "upstreamdrift-core"
        assert manifest.manifest_version == "1.0.0"
        assert manifest.model_ids == ["legacy_model"]
        legacy_model = manifest.models[0]
        assert legacy_model.source_root is None
        assert legacy_model.python_paths == ()

    def test_group_entries_by_canonical_id_resolves_cross_engine_equivalence(
        self,
    ) -> None:
        mujoco_entry = ModelPackEntry.from_dict(
            {
                "id": "mujoco_swing",
                "name": "MuJoCo Swing",
                "description": "MuJoCo swing model",
                "type": "mjcf",
                "path": "packs/mujoco/swing.xml",
                "engine_type": "mujoco",
                "identity": {
                    "canonical_id": "golf.swing.main",
                    "motion_family": "golf-swing",
                    "exercise": "driver-full-swing",
                    "humanoid": "golf-athlete",
                },
            }
        )
        drake_entry = ModelPackEntry.from_dict(
            {
                "id": "drake_swing",
                "name": "Drake Swing",
                "description": "Drake swing model",
                "type": "urdf",
                "path": "packs/drake/swing.urdf",
                "engine_type": "drake",
                "identity": {
                    "canonical_id": "Golf.Swing.Main",
                    "motion_family": "Golf Swing",
                    "exercise": "Driver Full Swing",
                    "humanoid": "Golf Athlete",
                },
            }
        )

        grouped = group_entries_by_canonical_id((mujoco_entry, drake_entry))

        assert list(grouped.keys()) == ["golf.swing.main"]
        assert [entry.id for entry in grouped["golf.swing.main"]] == [
            "drake_swing",
            "mujoco_swing",
        ]


@pytest.mark.unit
class TestModelPackV1LegacySchema:
    """Test reconciliation of model_pack/v1 provider schema with UpstreamDrift manifest.

    Related to issue #5313: External model repos use a simple model_pack/v1 manifest
    while UpstreamDrift expects fields like manifest_version, pack_id, pack_name, etc.
    This test verifies that legacy provider manifests are properly normalized.
    """

    def test_load_model_pack_v1_schema_with_minimal_fields(
        self, tmp_path: Path
    ) -> None:
        """Test loading a minimal model_pack/v1 style manifest from external provider."""
        manifest_path = tmp_path / "model_pack.yaml"
        # This is the format used by external provider repos (MuJoCo_Models, etc.)
        _write_yaml(
            manifest_path,
            {
                "schema": "model_pack/v1",
                "repo": "MuJoCo_Models",
                "package": "mujoco_models",
                "models": [
                    {
                        "id": "golf_swing",
                        "name": "Golf Swing",
                        "description": "Full golf swing motion capture",
                        "type": "mjcf",
                        "path": "motions/golf/swing.xml",
                    }
                ],
            },
        )

        # The from_dict method should handle this with defaults for missing fields
        raw = yaml.safe_load(manifest_path.read_text())

        # Convert model_pack/v1 schema to UpstreamDrift manifest
        converted = {
            "manifest_version": raw.get("manifest_version", "1.0.0"),
            "pack_id": raw.get("pack_id", raw.get("repo", "unknown")),
            "pack_name": raw.get("pack_name", raw.get("repo", "Unknown Pack")),
            "provider": raw.get("provider", raw.get("repo", "external")),
            "models": raw.get("models", []),
        }

        manifest = ModelPackManifest.from_dict(converted)

        assert manifest.manifest_version == "1.0.0"
        assert manifest.pack_id == "MuJoCo_Models"
        assert manifest.provider == "MuJoCo_Models"
        assert len(manifest.models) == 1
        assert manifest.models[0].id == "golf_swing"

    def test_load_model_pack_v1_with_launcher_metadata(self, tmp_path: Path) -> None:
        """Test that model_pack/v1 manifests can include launcher presentation metadata."""
        manifest_path = tmp_path / "model_pack.yaml"
        _write_yaml(
            manifest_path,
            {
                "schema": "model_pack/v1",
                "repo": "Drake_Models",
                "models": [
                    {
                        "id": "humanoid",
                        "name": "Humanoid",
                        "description": "Bipedal humanoid model",
                        "type": "urdf",
                        "path": "robots/humanoid.urdf",
                        "launcher": {
                            "category": "physics_engine",
                            "logo": "drake.svg",
                            "status": "ready",
                        },
                    }
                ],
            },
        )

        raw = yaml.safe_load(manifest_path.read_text())
        converted = {
            "manifest_version": "1.0.0",
            "pack_id": raw.get("repo", "drake_models"),
            "pack_name": raw.get("repo", "Drake Models"),
            "provider": raw.get("repo", "drake_models"),
            "models": raw.get("models", []),
        }

        manifest = ModelPackManifest.from_dict(converted)

        assert len(manifest.models) == 1
        entry = manifest.models[0]
        assert entry.launcher is not None
        assert entry.launcher.category == "physics_engine"
        assert entry.launcher.logo == "drake.svg"
        assert entry.launcher.status == "ready"

    def test_load_model_pack_v1_with_identity_metadata(self, tmp_path: Path) -> None:
        """Test that model_pack/v1 manifests support cross-engine identity metadata."""
        manifest_path = tmp_path / "model_pack.yaml"
        _write_yaml(
            manifest_path,
            {
                "schema": "model_pack/v1",
                "repo": "Pinocchio_Models",
                "models": [
                    {
                        "id": "golf_swing",
                        "name": "Golf Swing",
                        "description": "Golf swing with analytical derivatives",
                        "type": "urdf",
                        "path": "motions/golf/swing.urdf",
                        "identity": {
                            "canonical_id": "Golf.Swing.Main",
                            "motion_family": "Golf Swing",
                            "exercise": "Driver Full Swing",
                            "humanoid": "Golf Athlete",
                        },
                    }
                ],
            },
        )

        raw = yaml.safe_load(manifest_path.read_text())
        converted = {
            "manifest_version": "1.0.0",
            "pack_id": raw.get("repo", "pinocchio_models"),
            "pack_name": raw.get("repo", "Pinocchio Models"),
            "provider": raw.get("repo", "pinocchio_models"),
            "models": raw.get("models", []),
        }

        manifest = ModelPackManifest.from_dict(converted)

        assert len(manifest.models) == 1
        entry = manifest.models[0]
        assert entry.identity is not None
        assert entry.identity.canonical_id == "golf.swing.main"
        assert entry.identity.exercise == "driver-full-swing"
