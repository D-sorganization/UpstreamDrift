"""Unit tests for the shared model-pack manifest contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.shared.python.config.model_pack_manifest import (
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
