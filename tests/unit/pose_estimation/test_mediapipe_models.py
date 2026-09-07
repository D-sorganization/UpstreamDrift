"""Pose-model resolution and verified download, without the network."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from src.shared.python.config.typed_settings import Settings
from src.shared.python.core.error_utils import ModelError
from src.shared.python.pose_estimation import mediapipe_models as models

pytestmark = pytest.mark.unit


def test_model_spec_lookup_and_urls() -> None:
    spec = models.model_spec("full")
    assert spec.filename == "pose_landmarker_full.task"
    assert spec.url.endswith(
        "/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )
    assert len(spec.sha256) == 64
    with pytest.raises(ValueError, match="unknown pose model variant"):
        models.model_spec("giant")


def test_sha256_and_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"not a real model"
    path = tmp_path / "pose_landmarker_lite.task"
    path.write_bytes(data)
    assert models.sha256_of(path) == hashlib.sha256(data).hexdigest()
    assert models.verify_pose_model(path, "lite") is False
    monkeypatch.setitem(
        models.POSE_MODELS,
        "lite",
        models.PoseModelSpec("lite", hashlib.sha256(data).hexdigest(), len(data)),
    )
    assert models.verify_pose_model(path, "lite") is True
    assert models.verify_pose_model(tmp_path / "absent.task", "lite") is False


def test_resolve_precedence_explicit_env_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.task"
    explicit.write_bytes(b"x")
    env_model = tmp_path / "env.task"
    env_model.write_bytes(b"y")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pose_landmarker_full.task").write_bytes(b"z")
    monkeypatch.setattr(models, "default_cache_dir", lambda: cache)

    settings = Settings(MEDIAPIPE_POSE_MODEL_PATH=str(env_model))
    assert models.resolve_pose_model(explicit, settings=settings) == explicit
    assert models.resolve_pose_model(None, settings=settings) == env_model
    plain = Settings()
    assert (
        models.resolve_pose_model(None, "full", settings=plain)
        == cache / "pose_landmarker_full.task"
    )


def test_resolve_missing_model_names_the_download_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models, "default_cache_dir", lambda: tmp_path / "empty")
    with pytest.raises(ModelError, match="--variant heavy"):
        models.resolve_pose_model(None, "heavy", settings=Settings())


def test_download_verifies_and_installs_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = b"verified model bytes"
    monkeypatch.setitem(
        models.POSE_MODELS,
        "lite",
        models.PoseModelSpec("lite", hashlib.sha256(good).hexdigest(), len(good)),
    )
    calls: list[str] = []

    def opener(url: str) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(good)

    path = models.download_pose_model("lite", tmp_path, opener=opener)
    assert path == tmp_path / "pose_landmarker_lite.task"
    assert path.read_bytes() == good
    assert calls == [models.model_spec("lite").url]
    # Already present and verified: no second fetch.
    models.download_pose_model("lite", tmp_path, opener=opener)
    assert len(calls) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["pose_landmarker_lite.task"]


def test_download_rejects_tampered_bytes_and_leaves_nothing(tmp_path: Path) -> None:
    def opener(url: str) -> io.BytesIO:
        return io.BytesIO(b"tampered")

    with pytest.raises(ModelError, match="failed verification"):
        models.download_pose_model("lite", tmp_path, opener=opener)
    assert list(tmp_path.iterdir()) == []
