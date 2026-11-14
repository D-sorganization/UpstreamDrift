"""ML-WS-4 (#13781): native MuJoCo episode corpus generator manifest and gate.

Failing-first acceptance: same seed -> byte-identical manifest digest; manifest
records simulator, spec hash and hand-fix commit; generator refuses specs that
predate the hand fix; reuses neural-episode-store/1.0.0 without a new schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    N_JOINTS,
)
from src.shared.python.neural_motion.episodes import EPISODE_STORE_SCHEMA, EpisodeStore
from src.shared.python.neural_motion.episodes.mujoco_corpus import (
    CANONICAL_DRIVER_SPEC,
    HAND_FIX_COMMIT_SHA,
    MUJOCO_CORPUS_MANIFEST_SCHEMA,
    MuJoCoEpisodeCorpusGenerator,
    assert_spec_post_hand_fix,
    build_corpus_manifest,
    contract_rollout_arrays,
    load_canonical_driver_spec,
    raw_spec_sha256,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_FIXTURE_ROOT = (
    REPO_ROOT / "tests" / "fixtures" / "neural_motion" / "ml_ws4_mujoco_corpus_ci"
)


def _stale_spec() -> dict[str, Any]:
    """Minimal spec without the post-hand-fix grip-era marker."""
    return {
        "subject": {"mass_kg": 78.0, "stature_m": 1.71},
        "provenance": "pre-HX anthropometric candidate without grip contract era",
    }


@pytest.fixture
def contract_rollout():
    return contract_rollout_arrays


def test_manifest_records_simulator_spec_hash_and_hand_fix_commit(
    tmp_path: Path, contract_rollout
) -> None:
    spec = load_canonical_driver_spec()
    generator = MuJoCoEpisodeCorpusGenerator(rollout=contract_rollout)
    manifest = generator.generate(
        store_root=tmp_path / "corpus",
        spec=spec,
        seed=13781,
        episode_count=2,
    )
    assert manifest.schema == MUJOCO_CORPUS_MANIFEST_SCHEMA
    assert manifest.simulator == "motion_matching.FullBodySimulator"
    assert manifest.engine == "mujoco"
    assert manifest.hand_fix_commit == HAND_FIX_COMMIT_SHA
    assert manifest.spec_sha256 == raw_spec_sha256(CANONICAL_DRIVER_SPEC)
    assert manifest.spec_path == CANONICAL_DRIVER_SPEC.as_posix()
    assert manifest.store_manifest["schema"] == EPISODE_STORE_SCHEMA
    assert len(manifest.store_manifest["episode_ids"]) == 2


def test_same_seed_yields_byte_identical_manifest_digest(
    tmp_path: Path, contract_rollout
) -> None:
    spec = load_canonical_driver_spec()
    generator = MuJoCoEpisodeCorpusGenerator(rollout=contract_rollout)
    first = generator.generate(
        store_root=tmp_path / "a",
        spec=spec,
        seed=4242,
        episode_count=3,
    )
    second = generator.generate(
        store_root=tmp_path / "b",
        spec=spec,
        seed=4242,
        episode_count=3,
    )
    assert first.content_digest == second.content_digest
    assert first.as_json_bytes() == second.as_json_bytes()


def test_generator_fails_closed_when_spec_predates_hand_fix(
    tmp_path: Path, contract_rollout
) -> None:
    with pytest.raises(ValueError, match="hand-fix"):
        assert_spec_post_hand_fix(_stale_spec())
    generator = MuJoCoEpisodeCorpusGenerator(rollout=contract_rollout)
    with pytest.raises(ValueError, match="hand-fix"):
        generator.generate(
            store_root=tmp_path / "reject",
            spec=_stale_spec(),
            seed=1,
            episode_count=1,
        )


def test_build_corpus_manifest_is_deterministic() -> None:
    spec = load_canonical_driver_spec()
    store_manifest = {
        "schema": EPISODE_STORE_SCHEMA,
        "episode_ids": ["abc", "def"],
        "trial_index": {"t0": "abc", "t1": "def"},
        "content_digest": "deadbeef" * 8,
    }
    a = build_corpus_manifest(
        spec_path=CANONICAL_DRIVER_SPEC,
        spec_sha256=raw_spec_sha256(CANONICAL_DRIVER_SPEC),
        store_manifest=store_manifest,
        seed=99,
        episode_count=2,
    )
    b = build_corpus_manifest(
        spec_path=CANONICAL_DRIVER_SPEC,
        spec_sha256=raw_spec_sha256(CANONICAL_DRIVER_SPEC),
        store_manifest=store_manifest,
        seed=99,
        episode_count=2,
    )
    assert a.content_digest == b.content_digest
    assert a.as_json_bytes() == b.as_json_bytes()


def test_ci_fixture_is_under_one_megabyte_and_loads() -> None:
    assert CI_FIXTURE_ROOT.is_dir(), f"missing CI fixture at {CI_FIXTURE_ROOT}"
    total = sum(f.stat().st_size for f in CI_FIXTURE_ROOT.rglob("*") if f.is_file())
    assert total < 1_000_000, f"CI fixture size {total} exceeds 1 MB budget"
    manifest_path = CI_FIXTURE_ROOT / "corpus_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == MUJOCO_CORPUS_MANIFEST_SCHEMA
    assert manifest["hand_fix_commit"] == HAND_FIX_COMMIT_SHA
    store = EpisodeStore(CI_FIXTURE_ROOT)
    assert len(tuple(store.iter_episode_ids())) >= 1
    episode = store.read_episode(next(iter(store.iter_episode_ids())), lazy=True)
    assert episode.model_id == "full_body_mujoco"
    assert episode.control_basis == "polynomial_coefficient"
    assert tuple(episode.joint_names) == CANONICAL_JOINTS
    assert tuple(episode.coefficient_letters) == COEFFICIENT_LETTERS
    assert episode.coefficients is not None and episode.coefficients.shape == (
        N_COEFFS,
    )
    assert episode.q.shape[1] == N_JOINTS


def test_contract_rollout_arrays_are_finite() -> None:
    times = np.linspace(0.0, 0.04, 3)
    coeffs = np.arange(N_COEFFS, dtype=np.float64)
    out = contract_rollout_arrays(
        object(),
        seed=7,
        coefficients=coeffs,
        sample_times_s=times,
    )
    for key in ("q", "v", "u", "a_native", "q_next"):
        arr = out[key]
        assert np.all(np.isfinite(arr))
