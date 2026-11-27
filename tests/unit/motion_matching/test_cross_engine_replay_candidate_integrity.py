"""Replay config must preserve canonical candidate integrity (#14756 P5)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.shared.python.motion_matching import candidate_convert, candidate_io
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.cross_engine_replay import (
    CrossEngineReplayConfig,
)

pytestmark = pytest.mark.unit


def _config(path: Path) -> CrossEngineReplayConfig:
    """Build a one-engine replay config for the given archive."""
    return CrossEngineReplayConfig(path, duration_s=0.5, engines=("mujoco",))


@pytest.fixture
def canonical_path(tmp_path: Path) -> Path:
    """Write a canonical archive that also carries returned81 fields."""
    candidate = MatchedSwingCandidate(
        metadata=CandidateMetadata(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            profile=CandidateProfile.KINEMATIC,
            engine="mujoco",
            model_sha256="a" * 64,
            coordinate_names=("hip", "knee"),
        ),
        time_s=np.linspace(0.0, 0.5, 6),
        q=np.zeros((6, 2)),
    )
    path = tmp_path / "canonical.npz"
    candidate_io.save_candidate(candidate, path)
    # A mixed archive is still canonical. These otherwise valid returned81
    # fields must never let the converter launder failed canonical validation.
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    np.savez(path, **arrays, native_state=np.ones((6, 4)))
    return path


def test_valid_canonical_preserves_identity(
    canonical_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid mixed archive keeps canonical identity and skips conversion."""
    legacy = Mock(wraps=candidate_convert.convert_returned81_replay)
    monkeypatch.setattr(candidate_convert, "convert_returned81_replay", legacy)

    candidate = _config(canonical_path).load_candidate()

    assert candidate.metadata.engine == "mujoco"
    assert candidate.metadata.model_sha256 == "a" * 64
    assert candidate.metadata.coordinate_names == ("hip", "knee")
    np.testing.assert_array_equal(candidate.q, np.zeros((6, 2)))
    legacy.assert_not_called()


@pytest.mark.parametrize(
    "damage, message",
    [
        ("checksum", "checksum"),
        ("schema", "schema"),
        ("manifest", "Expecting"),
        ("object_manifest", "Object arrays"),
        ("missing_q", "q"),
    ],
)
def test_invalid_canonical_never_falls_back(
    canonical_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    message: str,
) -> None:
    """Corrupt canonical archives raise and never use the legacy converter."""
    with np.load(canonical_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if damage == "checksum":
        arrays["q"] = np.ones((6, 2))
    elif damage == "schema":
        manifest = json.loads(str(arrays[candidate_io.MANIFEST_KEY]))
        manifest["schema_version"] = "unsupported"
        arrays[candidate_io.MANIFEST_KEY] = np.array(json.dumps(manifest))
    elif damage == "manifest":
        arrays[candidate_io.MANIFEST_KEY] = np.array("{not json")
    elif damage == "object_manifest":
        arrays[candidate_io.MANIFEST_KEY] = np.array([{}], dtype=object)
    else:
        del arrays["q"]
    np.savez(canonical_path, **arrays)
    legacy = Mock(wraps=candidate_convert.convert_returned81_replay)
    monkeypatch.setattr(candidate_convert, "convert_returned81_replay", legacy)

    with pytest.raises((ValueError, KeyError), match=message):
        _config(canonical_path).load_candidate()
    legacy.assert_not_called()


def test_unexpected_canonical_error_propagates(
    canonical_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected canonical loader error propagates unchanged."""
    monkeypatch.setattr(
        candidate_io, "load_candidate", Mock(side_effect=RuntimeError("loader failed"))
    )
    with pytest.raises(RuntimeError, match="loader failed"):
        _config(canonical_path).load_candidate()


def test_numeric_returned81_still_converts(tmp_path: Path) -> None:
    """A manifest-free numeric returned81 archive still converts."""
    path = tmp_path / "returned81.npz"
    native_state = np.arange(24, dtype=float).reshape(6, 4)
    np.savez(path, time_s=np.linspace(0.0, 0.5, 6), native_state=native_state)

    candidate = _config(path).load_candidate()

    assert candidate.metadata.profile == CandidateProfile.KINEMATIC
    assert candidate.metadata.extra["legacy_format"] == "returned81_npz"
    np.testing.assert_array_equal(candidate.q, native_state[:, :2])
    np.testing.assert_array_equal(candidate.v, native_state[:, 2:])


def test_object_returned81_is_refused(tmp_path: Path) -> None:
    """An object-array returned81 archive is refused."""
    path = tmp_path / "object.npz"
    np.savez(
        path,
        time_s=np.linspace(0.0, 0.5, 6),
        native_state=np.full((6, 4), {}, dtype=object),
    )
    with pytest.raises(ValueError, match="Object arrays"):
        _config(path).load_candidate()


def test_unknown_archive_is_not_sent_to_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrecognised archive is refused without calling the converter."""
    path = tmp_path / "unknown.npz"
    np.savez(path, unrelated=np.zeros(2))
    legacy = Mock(wraps=candidate_convert.convert_returned81_replay)
    monkeypatch.setattr(candidate_convert, "convert_returned81_replay", legacy)
    with pytest.raises(ValueError, match="recognised candidate"):
        _config(path).load_candidate()
    legacy.assert_not_called()
