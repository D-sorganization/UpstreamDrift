---
issue: 9545
summary: "BunkerShot3D workbench solves run as bounded, cancellable API jobs (/v1/jobs/evaluate, /v1/jobs/compare, poll with per-shot progress, cancel) on the shared TaskManager; a cancelled or failed job never carries a result, and a completed one names its job id and input digest"
dl_state: "in_review"
next_step: "Merge the PR; then wire the web page to the job routes (progress bar, Cancel, stale-result guard by input digest), sand/swing inputs and /v1/verify save-reload."
branch: "feat/9545-bunker-jobs-s4"
---
