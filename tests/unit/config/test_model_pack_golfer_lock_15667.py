"""Model packs carry the golfer model lock to the registry (#15667, after #15659)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.shared.python.config.model_pack_manifest import ModelPackEntry
from src.shared.python.config.model_registry import ModelRegistry
from src.shared.python.model_appearance.canonical_golfer_gate import (
    GolferModelLock,
    NonCanonicalGolferError,
    verify_lock,
)
from tests.unit.model_appearance.test_canonical_golfer_gate_15659 import _lock

pytestmark = pytest.mark.unit


def _entry(**extra: object) -> dict[str, object]:
    return {
        "id": "golfer",
        "name": "Golfer",
        "description": "full-body golfer",
        "type": "mjcf",
        "path": "golfer.xml",
        **extra,
    }


def test_entry_without_a_lock_is_unchanged() -> None:
    entry = ModelPackEntry.from_dict(_entry())
    assert entry.golfer_model_lock is None
    assert "golfer_model_lock" not in entry.to_dict()


def test_entry_lock_roundtrips() -> None:
    lock = _lock().to_dict()
    entry = ModelPackEntry.from_dict(_entry(golfer_model_lock=lock))
    assert entry.golfer_model_lock == lock
    assert ModelPackEntry.from_dict(entry.to_dict()) == entry


@pytest.mark.parametrize(
    "bad",
    [
        {"plant_hash": "p" * 64},
        {**_lock().to_dict(), "solver_hash": ""},
        {**_lock().to_dict(), "extra": "x"},
        "not-a-mapping",
    ],
)
def test_malformed_entry_lock_is_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="model_lock"):
        ModelPackEntry.from_dict(_entry(golfer_model_lock=bad))


def test_registry_exposes_the_lock_for_the_gate(tmp_path: Path) -> None:
    expected = _lock()
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump({"models": [_entry(golfer_model_lock=expected.to_dict())]}),
        encoding="utf-8",
    )
    config = ModelRegistry(path).get_model("golfer")
    assert config is not None and config.golfer_model_lock is not None
    recorded = GolferModelLock.from_dict(config.golfer_model_lock)
    verify_lock(recorded, expected, scope="optimization")
    stale = GolferModelLock.from_dict({**expected.to_dict(), "plant_hash": "0" * 64})
    with pytest.raises(NonCanonicalGolferError) as info:
        verify_lock(stale, expected, scope="allocation")
    assert info.value.part == "plant_hash"
