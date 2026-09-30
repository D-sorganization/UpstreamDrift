# Native Engine Nightly Lane Receipts (MS-43 #10342)

OpenSim and MyoSuite pytest markers are skipped in standard CI. These receipts
record the last native-lane execution on a qualified host (typically
ControlTower) with contract hashes and per-test outcomes.

## Refresh on ControlTower

```bash
# OpenSim (opensim-10003 venv)
bash scripts/ci/run_native_engine_lane.sh \
  --engine opensim \
  --venv /home/dieterolson/opensim-10003 \
  --out docs/development/matched_swing_program/evidence/nightly

# MyoSuite (environment with myosuite extra installed)
bash scripts/ci/run_native_engine_lane.sh \
  --engine myosuite \
  --out docs/development/matched_swing_program/evidence/nightly
```

Commit the updated `opensim_receipt.json` and `myosuite_receipt.json`. Freshness
tests warn after seven days and fail after thirty days on `main`.

## Workflow Integration

Do not edit `.github/workflows` in this repository for lane placement. The
existing `nightly-cross-engine.yml` job installs optional engine extras; schedule
ControlTower refresh via the fleet `run_job.py` detached-job pattern or manual
dispatch on a labeled runner.

## Local Verification

```bash
python -m pytest tests/docs/test_native_lane_freshness.py tests/scripts/test_run_native_engine_lane.py -v
python scripts/ci/check_architecture_budget.py
ruff check --fix scripts/ci/run_native_engine_lane.py tests/docs/test_native_lane_freshness.py tests/scripts/test_run_native_engine_lane.py
```
