---
issue: 12970
summary: "pipeline: wire ground-support CLI neural flags through MotionMatchingController and record neural disposition on receipts"
branch: "orch/12970-cursor"
---

Ground-support pipeline CLI (`src/shared/python/motion_matching/pipeline/cli.py`) now routes
`--neural-mode`, `--neural-model`, `--no-neural-fallback`, and `--neural-checkpoint` through
`src/tools/motion_matching/neural_cli.py` and the existing `MotionMatchingController.prepare_run`
path (R07 #11147). Rejected neural runs with fallback disabled raise `NeuralRunRejectedError`
before IK/dynamics execute. Receipts include a `neural` block with requested versus executed mode,
model id, fallback decision, and disposition metadata.
