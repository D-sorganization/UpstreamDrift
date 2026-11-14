# ML-WS-5 MuJoCo Corpus Run Receipt and Loader

Governing issue: [#13782](https://github.com/D-sorganization/UpstreamDrift/issues/13782) (epic [#13777](https://github.com/D-sorganization/UpstreamDrift/issues/13777)).

Schema: `neural-mujoco-corpus-run/1.0.0`

## What Landed

- `src/shared/python/neural_motion/episodes/mujoco_corpus_run.py` — measured run
  receipt, manifest digest verification (fail closed), and
  `load_mujoco_corpus_for_training` for `DynamicsBaselineTrainer` / NM-05 loaders
- `scripts/neural_motion/ml_ws5_record_corpus_receipt.py` — record CI or full-size receipts
- CI evidence: `evidence/ml_ws5_mujoco_corpus_ci_receipt.json` (store stays in
  `tests/fixtures/neural_motion/ml_ws4_mujoco_corpus_ci/`)

## CI Fixture Receipt

```bash
python3 scripts/neural_motion/ml_ws5_record_corpus_receipt.py --ci-fixture
```

## Full-Size Software-Contract Corpus (Data Outside Git)

Uses the WS-4 `MuJoCoEpisodeCorpusGenerator` with the deterministic
`contract_rollout_arrays` path (no native engine required). Point `--output-dir`
at a directory outside the repository:

```bash
python3 scripts/neural_motion/ml_ws5_record_corpus_receipt.py \
  --output-dir /tmp/ml_ws5_full_corpus \
  --seed 137770 \
  --episode-count 512 \
  --split-seed 137770 \
  --receipt-path /tmp/ml_ws5_full_corpus/corpus_run_receipt.json
```

For native MuJoCo rollouts, omit the contract rollout by calling
`run_mujoco_corpus_generation` from application code with
`rollout=None` once `MujocoMatchingPlant` is available on the host.

## Loader

```python
from pathlib import Path
from src.shared.python.neural_motion.episodes import load_mujoco_corpus_for_training

bundle = load_mujoco_corpus_for_training(
    Path("tests/fixtures/neural_motion/ml_ws4_mujoco_corpus_ci"),
    receipt_path=Path("docs/plans/neural_motion_matching/evidence/ml_ws5_mujoco_corpus_ci_receipt.json"),
)
```

Refs #13782  
Refs #13777
