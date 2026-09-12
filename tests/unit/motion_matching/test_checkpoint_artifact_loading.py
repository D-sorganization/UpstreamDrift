"""Regression tests for safe motion-matching checkpoint loading."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from src.shared.python.motion_matching._checkpoint_artifacts import (
    load_checkpoint_dict,
    load_surrogate_checkpoint,
    require_schema_version,
)


class _UnsafePayload:
    def __reduce__(self):
        return (eval, ("'unsafe'",))


@pytest.mark.unit
def test_load_checkpoint_dict_uses_weights_only_true(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"placeholder")
    seen: dict[str, object] = {}

    def fake_load(path, *, map_location=None):
        seen["path"] = path
        seen["map_location"] = map_location
        return {"state_dict": {}, "config": {}, "schema_version": "1.0"}

    monkeypatch.setattr(
        "src.shared.python.motion_matching._checkpoint_artifacts._load_weights_only_checkpoint",
        fake_load,
    )

    payload = load_checkpoint_dict(
        ckpt,
        map_location="cpu",
        required_keys=("state_dict", "config"),
        artifact_name="test checkpoint",
    )

    assert payload["schema_version"] == "1.0"
    assert seen == {"path": ckpt, "map_location": "cpu"}


@pytest.mark.unit
def test_load_weights_only_checkpoint_forwards_safe_torch_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.shared.python.motion_matching import _checkpoint_artifacts

    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"placeholder")
    seen: dict[str, object] = {}

    class _TorchModule:
        @staticmethod
        def load(path, *, map_location=None, weights_only=None):
            seen["path"] = path
            seen["map_location"] = map_location
            seen["weights_only"] = weights_only
            return {"state_dict": {}}

    monkeypatch.setitem(sys.modules, "torch", _TorchModule())

    payload = _checkpoint_artifacts._load_weights_only_checkpoint(
        ckpt,
        map_location="cpu",
    )

    assert payload == {"state_dict": {}}
    assert seen == {"path": ckpt, "map_location": "cpu", "weights_only": True}


@pytest.mark.unit
def test_load_checkpoint_dict_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint_dict(tmp_path / "missing.pt")


@pytest.mark.unit
def test_load_checkpoint_dict_rejects_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "src.shared.python.motion_matching._checkpoint_artifacts._load_weights_only_checkpoint",
        lambda path, *, map_location=None: ["state_dict"],
    )

    with pytest.raises(ValueError, match="not a dict payload"):
        load_checkpoint_dict(ckpt, artifact_name="test checkpoint")


@pytest.mark.unit
def test_load_checkpoint_dict_rejects_missing_required_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "src.shared.python.motion_matching._checkpoint_artifacts._load_weights_only_checkpoint",
        lambda path, *, map_location=None: {"state_dict": {}},
    )

    with pytest.raises(ValueError, match="missing required key\\(s\\): config"):
        load_checkpoint_dict(ckpt, required_keys=("state_dict", "config"))


@pytest.mark.unit
def test_load_surrogate_checkpoint_enforces_required_artifact_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "src.shared.python.motion_matching._checkpoint_artifacts._load_weights_only_checkpoint",
        lambda path, *, map_location=None: {"model_state_dict": {}},
    )

    with pytest.raises(ValueError, match="missing required key\\(s\\)"):
        load_surrogate_checkpoint(ckpt, artifact_name="test surrogate checkpoint")


@pytest.mark.unit
@pytest.mark.requires_torch
def test_load_checkpoint_dict_rejects_pickle_globals(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    ckpt = tmp_path / "unsafe.pt"
    torch.save({"payload": _UnsafePayload()}, ckpt)

    with pytest.raises(ValueError, match="cannot be loaded safely"):
        load_checkpoint_dict(ckpt, required_keys=("payload",))


@pytest.mark.unit
def test_require_schema_version_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        require_schema_version(
            {"schema_version": "0.9"},
            "1.0",
            artifact_name="test checkpoint",
        )


@pytest.mark.unit
def test_require_schema_version_accepts_expected_version() -> None:
    require_schema_version(
        {"schema_version": "1.0"},
        "1.0",
        artifact_name="test checkpoint",
    )
