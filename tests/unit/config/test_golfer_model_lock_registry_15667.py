"""Tests for GolferModelLock in ModelPackManifest and ModelRegistry (#15667)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from src.shared.python.config.model_pack_manifest import (
    ModelPackEntry,
    ModelPackManifest,
    validate_golfer_pack_lock,
)
from src.shared.python.config.model_registry import ModelConfig, ModelRegistry
from src.shared.python.model_appearance.canonical_golfer_gate import (
    GolferModelLock,
    NonCanonicalGolferError,
    controller_hash,
    lock_for_spec,
    solver_hash,
)
from src.shared.python.motion_matching.execution.spec_revisions import (
    with_current_shaft,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (REPO / "tests/fixtures/club_face/spec_driver.json").read_text(encoding="utf-8")
)
KP, KD = np.full(3, 200.0), np.full(3, 20.0)
KNOTS = np.array([0.0, 0.0, 0.0, 0.125, 0.25, 0.25, 0.25])
CTRL_HASH = controller_hash(KP, KD, KNOTS, {"hold": "zoh"})
SOLVER_HASH = solver_hash({"name": "rk4", "dt": 0.05})


def _canonical_spec() -> dict[str, Any]:
    """Return a canonical current-registry golfer spec for test fixtures."""
    spec = copy.deepcopy(SPEC)
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    spec["spec_revision"] = {"version": 1, "configuration": "skinned_hand_wrists"}
    return with_current_shaft(spec)


def _make_lock(spec: dict[str, Any] | None = None) -> GolferModelLock:
    """Build a valid GolferModelLock for the given canonical spec."""
    return lock_for_spec(
        spec or _canonical_spec(), controller=CTRL_HASH, solver=SOLVER_HASH
    )


def _entry_dict(
    model_id: str = "canonical_golfer", lock: dict[str, str] | None = None
) -> dict[str, Any]:
    """Construct a minimal ModelPackEntry payload."""
    payload: dict[str, Any] = {
        "id": model_id,
        "name": "Canonical Golfer",
        "description": "Golfer model pack entry",
        "type": "mjcf",
        "path": "packs/mujoco/golfer.xml",
    }
    if lock is not None:
        payload["golfer_model_lock"] = lock
    return payload


def _locked_entry() -> tuple[dict[str, Any], GolferModelLock, ModelPackEntry]:
    """Build a canonical spec, lock, and parsed ModelPackEntry."""
    spec = _canonical_spec()
    lock = _make_lock(spec)
    entry = ModelPackEntry.from_dict(_entry_dict("canonical_golfer", lock.to_dict()))
    return spec, lock, entry


def _local_registry(tmp_path: Path, *entries: ModelPackEntry) -> ModelRegistry:
    """Write entries to a temporary models.yaml and load a local-only ModelRegistry."""
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        yaml.safe_dump({"models": [e.to_dict() for e in entries]}), encoding="utf-8"
    )
    return ModelRegistry(cfg, discovery_mode="local-only")


class TestModelPackEntryGolferModelLock:
    """ModelPackEntry serialization and parsing of golfer_model_lock."""

    def test_from_dict_parses_and_serializes_golfer_model_lock(self) -> None:
        """ModelPackEntry parses golfer_model_lock via GolferModelLock.from_dict."""
        lock = _make_lock()
        entry = ModelPackEntry.from_dict(_entry_dict(lock=lock.to_dict()))
        assert entry.golfer_model_lock == lock
        assert isinstance(entry.golfer_model_lock, GolferModelLock)
        assert entry.to_dict()["golfer_model_lock"] == lock.to_dict()

    def test_non_golfer_pack_without_lock_is_unaffected(self) -> None:
        """Non-golfer entries without golfer_model_lock remain None and omit key."""
        entry = ModelPackEntry.from_dict(_entry_dict("pendulum_swing"))
        assert entry.golfer_model_lock is None
        assert "golfer_model_lock" not in entry.to_dict()

    @pytest.mark.parametrize(
        "bad_lock",
        [
            "not-a-mapping",
            {"plant_hash": "a" * 64},
            {**_make_lock().to_dict(), "registry_version": ""},
        ],
    )
    def test_from_dict_rejects_malformed_golfer_model_lock(self, bad_lock: Any) -> None:
        """Malformed golfer_model_lock payloads fail closed with ValueError."""
        with pytest.raises(ValueError):
            ModelPackEntry.from_dict(_entry_dict(lock=bad_lock))

    def test_v1_provider_manifest_preserves_golfer_model_lock(self) -> None:
        """Provider model_pack/v1 exercises forward golfer_model_lock."""
        lock = _make_lock()
        ex = {
            "id": "golfer_driver",
            "path": "golfer/driver.xml",
            "golfer_model_lock": lock.to_dict(),
        }
        manifest = ModelPackManifest.from_dict(
            {
                "schema": "model_pack/v1",
                "repo": "MuJoCo_Models",
                "package": "mujoco_models",
                "engine": "mujoco",
                "format": "mjcf",
                "models_root": "models",
                "exercises": [ex],
            }
        )
        assert len(manifest.models) == 1
        assert manifest.models[0].golfer_model_lock == lock


class TestModelRegistryGolferModelLock:
    """ModelConfig and ModelRegistry surfacing of golfer_model_lock."""

    def test_model_config_defaults_golfer_model_lock_to_none(self) -> None:
        """ModelConfig preserves None default for non-golfer models."""
        cfg = ModelConfig(
            id="pendulum",
            name="Pendulum",
            description="Simple pendulum",
            type="mjcf",
            path="pendulum.xml",
        )
        assert cfg.golfer_model_lock is None

    def test_registry_surfaces_lock_on_local_and_provider_models(
        self, tmp_path: Path
    ) -> None:
        """ModelRegistry surfaces golfer_model_lock for local and provider entries."""
        lock = _make_lock()
        cfg = tmp_path / "src" / "config" / "models.yaml"
        cfg.parent.mkdir(parents=True)
        local_models = [
            _entry_dict("local_golfer", lock.to_dict()),
            _entry_dict("non_golfer"),
        ]
        cfg.write_text(yaml.safe_dump({"models": local_models}), encoding="utf-8")

        provider_root = tmp_path / "providers" / "MuJoCo_Models"
        provider_root.mkdir(parents=True)
        pack_doc = {
            "manifest_version": "1.0.0",
            "pack_id": "mujoco-models",
            "pack_name": "MuJoCo Models",
            "provider": "mujoco_models",
            "models": [_entry_dict("provider_golfer", lock.to_dict())],
        }
        (provider_root / "model_pack.yaml").write_text(
            yaml.safe_dump(pack_doc), encoding="utf-8"
        )

        env = {
            "UPSTREAM_DRIFT_PROVIDER_ROOTS": str(provider_root),
            "UPSTREAM_DRIFT_DISCOVERY_MODE": "hybrid",
        }
        with patch.dict("os.environ", env, clear=False):
            registry = ModelRegistry(cfg)

        local_golfer = registry.get_model("local_golfer")
        non_golfer = registry.get_model("non_golfer")
        provider_golfer = registry.get_model("provider_golfer")
        assert local_golfer is not None and local_golfer.golfer_model_lock == lock
        assert non_golfer is not None and non_golfer.golfer_model_lock is None
        assert provider_golfer is not None and provider_golfer.golfer_model_lock == lock


class TestValidateGolferPackLock:
    """Validation of golfer packs against current-registry specs."""

    def test_validate_golfer_pack_lock_passes_when_locks_match(
        self, tmp_path: Path
    ) -> None:
        """Matching lock passes for both ModelPackEntry and ModelRegistry."""
        spec, lock, entry = _locked_entry()
        assert validate_golfer_pack_lock(entry, spec=spec) == lock
        assert (
            validate_golfer_pack_lock(
                entry, spec=spec, controller=CTRL_HASH, solver=SOLVER_HASH
            )
            == lock
        )
        registry = _local_registry(tmp_path, entry)
        assert (
            registry.validate_golfer_pack_lock(
                "canonical_golfer", spec=spec, controller=CTRL_HASH, solver=SOLVER_HASH
            )
            == lock
        )

    def test_validate_golfer_pack_lock_fails_when_current_spec_lacks_lock(
        self, tmp_path: Path
    ) -> None:
        """Current-registry golfer pack without golfer_model_lock fails closed."""
        spec = _canonical_spec()
        entry = ModelPackEntry.from_dict(_entry_dict("unlocked_golfer"))
        with pytest.raises(NonCanonicalGolferError) as exc_info:
            validate_golfer_pack_lock(entry, spec=spec)
        assert isinstance(exc_info.value, ValueError)

        registry = _local_registry(tmp_path, entry)
        with pytest.raises(NonCanonicalGolferError):
            registry.validate_golfer_pack_lock("unlocked_golfer", spec=spec)

    def test_validate_golfer_pack_lock_fails_on_plant_mismatch(self) -> None:
        """Changed dynamics fail closed with NonCanonicalGolferError naming plant_hash."""
        _, _, entry = _locked_entry()
        mutated_spec = _canonical_spec()
        mutated_spec["subject"]["mass_kg"] += 1.0
        with pytest.raises(NonCanonicalGolferError) as exc_info:
            validate_golfer_pack_lock(entry, spec=mutated_spec)
        assert exc_info.value.part == "plant_hash"

    def test_validate_golfer_pack_lock_fails_on_controller_or_identity_mismatch(
        self,
    ) -> None:
        """Mismatched controller hash or registry version fails closed."""
        spec, lock, entry = _locked_entry()
        with pytest.raises(NonCanonicalGolferError) as ctrl_exc:
            validate_golfer_pack_lock(
                entry, spec=spec, controller="0" * 64, solver=SOLVER_HASH
            )
        assert ctrl_exc.value.part == "controller_hash"

        stale_entry = replace(
            entry, golfer_model_lock=replace(lock, registry_version="legacy_v0")
        )
        with pytest.raises(NonCanonicalGolferError) as ver_exc:
            validate_golfer_pack_lock(stale_entry, spec=spec)
        assert ver_exc.value.part == "registry_version"

    def test_validate_golfer_pack_lock_fails_on_non_canonical_spec(self) -> None:
        """A deprecated/non-canonical spec is refused by validate_golfer_pack_lock."""
        _, _, entry = _locked_entry()
        legacy_spec = _canonical_spec()
        legacy_spec["subject"]["body_visual_backend"] = "loft"
        with pytest.raises(NonCanonicalGolferError) as exc_info:
            validate_golfer_pack_lock(entry, spec=legacy_spec)
        assert exc_info.value.part == "loft"

    def test_validate_golfer_pack_lock_leaves_non_golfer_pack_unaffected(self) -> None:
        """Non-golfer pack without spec or lock passes cleanly as None."""
        entry = ModelPackEntry.from_dict(_entry_dict("double_pendulum"))
        assert validate_golfer_pack_lock(entry) is None
