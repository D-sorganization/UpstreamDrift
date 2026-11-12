"""ML-WS-5 (#13782): corpus generation run receipt and trainer-facing loader.

Failing-first acceptance: manifest hash mismatch rejects load; split_seed and
held-out buckets are recorded in the receipt; loader output feeds
DynamicsBaselineTrainer without a parallel store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.neural_motion.baselines import (
    BASELINE_SCHEMA,
    DynamicsBaselineTrainer,
    DynamicsBaselineTrainerConfig,
    DynamicsTaskKind,
)
from src.shared.python.neural_motion.episodes.mujoco_corpus import (
    MUJOCO_CORPUS_MANIFEST_SCHEMA,
    contract_rollout_arrays,
    load_canonical_driver_spec,
)
from src.shared.python.neural_motion.episodes.mujoco_corpus_run import (
    MUJOCO_CORPUS_RUN_RECEIPT_SCHEMA,
    MuJoCoCorpusGenerationRequest,
    load_mujoco_corpus_for_training,
    load_corpus_run_receipt,
    run_mujoco_corpus_generation,
    verify_corpus_manifest_digest,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_FIXTURE_ROOT = (
    REPO_ROOT / "tests" / "fixtures" / "neural_motion" / "ml_ws4_mujoco_corpus_ci"
)
CI_RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "neural_motion_matching"
    / "evidence"
    / "ml_ws5_mujoco_corpus_ci_receipt.json"
)


def test_verify_corpus_manifest_digest_rejects_tampered_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_bytes((CI_FIXTURE_ROOT / "corpus_manifest.json").read_bytes())
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verify_corpus_manifest_digest(manifest_path) == data["content_digest"]
    data["episode_count"] = int(data["episode_count"]) + 1
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="content_digest"):
        verify_corpus_manifest_digest(manifest_path)


def test_loader_rejects_manifest_digest_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest"):
        load_mujoco_corpus_for_training(
            CI_FIXTURE_ROOT,
            expected_corpus_manifest_digest="0" * 64,
        )


def test_run_records_split_seed_and_builds_receipt(tmp_path: Path) -> None:
    spec = load_canonical_driver_spec()
    manifest, split, receipt = run_mujoco_corpus_generation(
        MuJoCoCorpusGenerationRequest(
            store_root=tmp_path / "corpus",
            seed=13782,
            episode_count=4,
            split_seed=991,
            rollout=contract_rollout_arrays,
            spec=spec,
        )
    )
    assert manifest.episode_count == 4
    assert receipt.schema == MUJOCO_CORPUS_RUN_RECEIPT_SCHEMA
    assert receipt.generator_seed == 13782
    assert receipt.split_seed == 991
    assert receipt.corpus_manifest_digest == manifest.content_digest
    assert receipt.split_plan_digest == split.content_digest
    assert receipt.wall_clock_s >= 0.0
    assert sum(receipt.split_counts.values()) == 4
    assert "train" in receipt.split_counts


def test_committed_ci_receipt_is_self_consistent() -> None:
    assert CI_RECEIPT_PATH.is_file(), f"missing receipt at {CI_RECEIPT_PATH}"
    receipt = load_corpus_run_receipt(CI_RECEIPT_PATH)
    assert receipt.schema == MUJOCO_CORPUS_RUN_RECEIPT_SCHEMA
    bundle = load_mujoco_corpus_for_training(
        REPO_ROOT / receipt.store_root,
        receipt_path=CI_RECEIPT_PATH,
    )
    assert bundle.manifest.schema == MUJOCO_CORPUS_MANIFEST_SCHEMA
    assert receipt.split_seed == bundle.receipt.split_seed
    assert receipt.split_plan_digest == bundle.split.content_digest


def test_loader_feeds_dynamics_baseline_trainer(tmp_path: Path) -> None:
    spec = load_canonical_driver_spec()
    store_root = tmp_path / "corpus"
    _, split, _ = run_mujoco_corpus_generation(
        MuJoCoCorpusGenerationRequest(
            store_root=store_root,
            seed=42,
            episode_count=6,
            split_seed=7,
            rollout=contract_rollout_arrays,
            spec=spec,
            ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        )
    )
    bundle = load_mujoco_corpus_for_training(store_root)
    assert bundle.split.content_digest == split.content_digest
    trainer = DynamicsBaselineTrainer(
        DynamicsBaselineTrainerConfig(
            store=bundle.store,
            split=bundle.split,
            task=DynamicsTaskKind.FORWARD_ACCELERATION,
            seeds=(0,),
            output_dir=tmp_path / "pilot",
            active_dofs=(0,),
            overfit_smoke=True,
        )
    )
    card = trainer.run_pilot()
    assert card.schema == BASELINE_SCHEMA
    assert card.test_untouched is True
