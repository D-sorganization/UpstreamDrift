"""Candidate resolution integrity in the parity report (#14756 P5, F-UD-030).

A canonical candidate whose checksum, schema or manifest fails validation must
fail closed: it may never fall through to the legacy loader, and no archive is
ever opened with ``allow_pickle=True``. Legacy numeric archives (``time_s``,
``q`` and the documented optional keys) are detected explicitly; object arrays,
unknown keys and malformed archives are refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import parity_report
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import (
    MANIFEST_KEY,
    save_candidate,
)

pytestmark = pytest.mark.unit

_N = 6


@pytest.fixture
def spies(monkeypatch):
    """Count legacy-loader calls and record every ``allow_pickle`` request."""
    calls = {"legacy": 0, "allow_pickle": []}
    legacy = parity_report._legacy_candidate_from_npz
    real_load = np.load

    def counting_legacy(*args, **kwargs):
        calls["legacy"] += 1
        return legacy(*args, **kwargs)

    def recording_load(*args, **kwargs):
        calls["allow_pickle"].append(kwargs.get("allow_pickle", False))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(parity_report, "_legacy_candidate_from_npz", counting_legacy)
    monkeypatch.setattr(np, "load", recording_load)
    return calls


def _canonical(tmp_path: Path) -> Path:
    cand = MatchedSwingCandidate(
        metadata=CandidateMetadata(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            profile=CandidateProfile.KINEMATIC,
            engine="fake",
            coordinate_names=("q0", "q1"),
        ),
        time_s=np.linspace(0.0, 0.5, _N),
        q=np.zeros((_N, 2)),
    )
    path = tmp_path / "canonical.npz"
    save_candidate(cand, path)
    return path


def _rewrite(path: Path, **changes) -> Path:
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays.update(changes)
    out = path.with_name("tampered.npz")
    np.savez(out, **arrays)
    return out


def _manifest(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data[MANIFEST_KEY]))


def test_valid_canonical_candidate_loads_without_legacy(tmp_path, spies) -> None:
    cand = parity_report._resolve_candidate(_canonical(tmp_path))
    assert cand.metadata.engine == "fake"
    assert spies["legacy"] == 0
    assert True not in spies["allow_pickle"]


def test_checksum_failure_fails_closed(tmp_path, spies) -> None:
    tampered = _rewrite(_canonical(tmp_path), q=np.ones((_N, 2)))
    with pytest.raises(ValueError, match="checksum"):
        parity_report._resolve_candidate(tampered)
    assert spies["legacy"] == 0
    assert True not in spies["allow_pickle"]


def test_schema_failure_fails_closed(tmp_path, spies) -> None:
    path = _canonical(tmp_path)
    manifest = {**_manifest(path), "schema_version": "candidate/v0"}
    tampered = _rewrite(path, **{MANIFEST_KEY: np.array(json.dumps(manifest))})
    with pytest.raises(ValueError, match="schema"):
        parity_report._resolve_candidate(tampered)
    assert spies["legacy"] == 0


def test_corrupt_manifest_fails_closed(tmp_path, spies) -> None:
    tampered = _rewrite(_canonical(tmp_path), **{MANIFEST_KEY: np.array("{not json")})
    with pytest.raises(ValueError):
        parity_report._resolve_candidate(tampered)
    assert spies["legacy"] == 0


def test_numeric_legacy_archive_is_detected_explicitly(tmp_path, spies) -> None:
    path = tmp_path / "legacy.npz"
    t = np.linspace(0.0, 0.5, _N)
    np.savez(path, time_s=t, q=np.zeros((_N, 3)), qd=np.ones((_N, 3)))
    cand = parity_report._resolve_candidate(path)
    assert cand.metadata.engine == "unknown"
    assert cand.q.shape == (_N, 3)
    assert spies["legacy"] == 1
    assert True not in spies["allow_pickle"]


def test_object_array_is_refused_without_unpickling(tmp_path, spies) -> None:
    path = tmp_path / "objects.npz"
    payload = np.empty(1, dtype=object)
    payload[0] = {"x": 1}
    np.savez(path, time_s=np.linspace(0.0, 0.5, _N), q=payload)
    with pytest.raises(ValueError, match="numeric"):
        parity_report._resolve_candidate(path)
    assert True not in spies["allow_pickle"]


@pytest.mark.parametrize(
    "arrays",
    [
        {"foo": np.zeros(3)},
        {"time_s": np.zeros(3), "q": np.zeros((3, 1)), "x": np.zeros(1)},
    ],
)
def test_unknown_archive_is_refused(tmp_path, spies, arrays) -> None:
    path = tmp_path / "unknown.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="not a recognised"):
        parity_report._resolve_candidate(path)
    assert True not in spies["allow_pickle"]


def test_malformed_archive_is_refused(tmp_path, spies) -> None:
    path = tmp_path / "garbage.npz"
    path.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="readable"):
        parity_report._resolve_candidate(path)
    assert spies["legacy"] == 0
