#!/usr/bin/env python3
"""Record an ML-WS-5 corpus run receipt (CI fixture or full-size generation).

CI fixture (fast, committed receipt only):

    python3 scripts/neural_motion/ml_ws5_record_corpus_receipt.py --ci-fixture

Full-size software-contract run (data stays outside git; receipt written to path):

    python3 scripts/neural_motion/ml_ws5_record_corpus_receipt.py \\
        --output-dir /tmp/ml_ws5_full_corpus \\
        --seed 137770 \\
        --episode-count 512 \\
        --split-seed 137770 \\
        --receipt-path /tmp/ml_ws5_full_corpus/corpus_run_receipt.json
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.shared.python.neural_motion.episodes.mujoco_corpus import (
    contract_rollout_arrays,
    load_canonical_driver_spec,
)
from src.shared.python.neural_motion.episodes.mujoco_corpus_run import (
    MuJoCoCorpusGenerationRequest,
    build_corpus_run_receipt,
    read_corpus_manifest,
    resolve_repo_head_sha,
    run_mujoco_corpus_generation,
    verify_corpus_manifest_digest,
)
from src.shared.python.neural_motion.episodes.splits import build_family_splits
from src.shared.python.neural_motion.episodes.store import EpisodeStore
from src.shared.python.neural_motion.json_io import write_sorted_json

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_FIXTURE = _REPO_ROOT / "tests/fixtures/neural_motion/ml_ws4_mujoco_corpus_ci"
_DEFAULT_RECEIPT = (
    _REPO_ROOT
    / "docs/plans/neural_motion_matching/evidence/ml_ws5_mujoco_corpus_ci_receipt.json"
)


def _record_from_existing_ci_fixture(
    *, split_seed: int, seed: int, episode_count: int
) -> None:
    import tempfile

    spec = load_canonical_driver_spec()
    fixture_manifest = verify_corpus_manifest_digest(
        _CI_FIXTURE / "corpus_manifest.json"
    )
    with tempfile.TemporaryDirectory(prefix="ml_ws5_ci_") as tmp:
        tmp_root = Path(tmp) / "corpus"
        manifest, _, receipt_run = run_mujoco_corpus_generation(
            MuJoCoCorpusGenerationRequest(
                store_root=tmp_root,
                seed=seed,
                episode_count=episode_count,
                split_seed=split_seed,
                spec=spec,
                rollout=contract_rollout_arrays,
                repo_root=_REPO_ROOT,
            )
        )
    if manifest.content_digest != fixture_manifest:
        raise SystemExit(
            "CI fixture corpus_manifest digest does not match a fresh WS-4 run "
            f"({manifest.content_digest} != {fixture_manifest}); refresh the fixture"
        )
    rel_root = _CI_FIXTURE.relative_to(_REPO_ROOT)
    manifest = read_corpus_manifest(_CI_FIXTURE)
    store = EpisodeStore(_CI_FIXTURE)
    episodes = [store.read_episode(eid, lazy=True) for eid in store.iter_episode_ids()]
    split = build_family_splits(
        episodes,
        ratios=receipt_run.split_ratios,
        seed=split_seed,
        held_out_strata=receipt_run.held_out_strata,
    )
    receipt = build_corpus_run_receipt(
        store_root=rel_root,
        manifest=manifest,
        split=split,
        split_seed=split_seed,
        wall_clock_s=receipt_run.wall_clock_s,
        repo_commit_sha=resolve_repo_head_sha(_REPO_ROOT),
        split_ratios=dict(receipt_run.split_ratios),
        held_out_strata=dict(receipt_run.held_out_strata),
    )
    write_sorted_json(_DEFAULT_RECEIPT, receipt.as_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci-fixture",
        action="store_true",
        help="Write the committed CI receipt for the ml_ws4 CI fixture store",
    )
    parser.add_argument("--output-dir", type=Path, help="Corpus output directory")
    parser.add_argument("--seed", type=int, default=137810)
    parser.add_argument("--episode-count", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=137810)
    parser.add_argument("--receipt-path", type=Path)
    args = parser.parse_args()

    if args.ci_fixture:
        _record_from_existing_ci_fixture(
            split_seed=args.split_seed,
            seed=args.seed,
            episode_count=args.episode_count,
        )
        return

    if args.output_dir is None:
        parser.error("--output-dir is required unless --ci-fixture is set")

    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    spec = load_canonical_driver_spec()
    manifest, split, receipt = run_mujoco_corpus_generation(
        MuJoCoCorpusGenerationRequest(
            store_root=out_dir,
            seed=args.seed,
            episode_count=args.episode_count,
            split_seed=args.split_seed,
            spec=spec,
            rollout=contract_rollout_arrays,
            receipt_path=args.receipt_path,
            repo_root=_REPO_ROOT,
        )
    )
    print(manifest.content_digest)
    print(split.content_digest)
    print(receipt.content_digest)


if __name__ == "__main__":
    main()
