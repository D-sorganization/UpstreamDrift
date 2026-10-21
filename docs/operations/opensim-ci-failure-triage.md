# OpenSim CI Failure Triage (UD-CI-002T)

Parent epic: [Repository_Management#2215](https://github.com/D-sorganization/Repository_Management/issues/2215).

This note satisfies [#13582](https://github.com/D-sorganization/UpstreamDrift/issues/13582) and unblocks tiering work in [#13577](https://github.com/D-sorganization/UpstreamDrift/issues/13577). It does **not** change `ci-requires-opensim.yml` (path/label gating remains [#13571](https://github.com/D-sorganization/UpstreamDrift/issues/13571) / PR [#13585](https://github.com/D-sorganization/UpstreamDrift/pull/13585)).

## Evidence

| Artifact | Purpose |
| -------- | ------- |
| [`docs/operations/evidence/opensim-ci-triage-20261021.json`](evidence/opensim-ci-triage-20261021.json) | Frozen 14-day snapshot aligned with RM-CI-000 / [#2216](https://github.com/D-sorganization/Repository_Management/issues/2216) |
| [`scripts/ci/opensim_workflow_triage.py`](../../scripts/ci/opensim_workflow_triage.py) | Regenerate the snapshot from GitHub Actions (`python3 -m scripts.ci.opensim_workflow_triage --window-days 14 --output …`) |
| [`tests/ci/test_opensim_workflow_triage.py`](../../tests/ci/test_opensim_workflow_triage.py) | Classification contract tests |

## Window and Headline Metrics

Snapshot generated **2026-10-21** (UTC), workflow `ci-requires-opensim.yml`, 14-day lookback:

| Metric | Value | Notes |
| ------ | ----- | ----- |
| Failure rate (non-cancelled) | **37.3%** | All events; cancelled superseded PR runs excluded |
| PR `pull_request` failure rate | **49.7%** | Matches the ~51% draft baseline called out in the epic |
| `main` push failure rate | **43.5%** | Post-merge signal; dominated by the Oct 20 regression |
| Success-run wall-clock **p90** | **22 953 s (~6.4 h)** | `run_started_at` → `updated_at`; includes docker-fleet queue time, not pytest-only |
| Flake (same `head_sha` success+failure) | **0** SHAs | No replay ambiguity in the window |

Cancelled runs (199) are concurrency noise from `cancel-in-progress` on PRs and are excluded from failure-rate math.

## Failure Mode Classification (14D)

| Category | Count | Share of failures |
| -------- | ----- | ----------------- |
| **test_defect** | 100 | 91% |
| **unknown** (expired/missing logs) | 7 | 6% |
| **infra_runner** (orphan cleanup / empty log) | 3 | 3% |
| infra_disk | 0 | — |
| infra_timeout (GitHub job limit) | 0 | — |
| env_misconfig | 0 | — |
| flake | 0 | — |

Infra buckets are **not** candidates for `@pytest.mark.quarantine`; they need runner or workflow fixes.

## Top Three Failure Signatures

| Rank | Signature | Category | Owner | Decision |
| ---- | --------- | -------- | ----- | -------- |
| 1 | `test_squat_joint_reactions_computed` **+** `test_native_constant_interval_matches_fresh_raw_replay` (dual failure) | test_defect | @core (lifting/native OpenSim) | **Fix** in [#13564](https://github.com/D-sorganization/UpstreamDrift/issues/13564) — deterministic since main `2e64f84` (#13496); no tolerance widening |
| 2 | `test_squat_joint_reactions_computed` only (partial log capture) | test_defect | @core (lifting/native OpenSim) | **Fix** (same root cause as rank 1) |
| 3 | `unclassified` / `log_unavailable` | unknown / infra_runner | @core | **Monitor** — logs expired on cancelled or superseded runs; re-pull before treating as a new defect |

Example failing run for signature 1: `https://github.com/D-sorganization/UpstreamDrift/actions/runs/39563097770`

### Regression Detail (Test Defect)

From job logs and [#13564](https://github.com/D-sorganization/UpstreamDrift/issues/13564):

1. **`test_squat_joint_reactions_computed`** — `acceleration_error_max` ≈ `1.15e-5` vs `5e-6` limit (introduced #13415; IK path touched #13456).
2. **`test_native_constant_interval_matches_fresh_raw_replay`** — optimizer infeasible at `nfev=80` (suspect #13478 native muscle-effort adapter reland).

Since **2026-10-20** the lane is effectively always red on `main` until these are fixed (open fix PR: `cursor/fix-opensim-ci-13564-90e0`).

## Follow-Ups

- Land [#13564](https://github.com/D-sorganization/UpstreamDrift/issues/13564) fix before re-baselining tiering SLOs.
- After [#13587](https://github.com/D-sorganization/UpstreamDrift/pull/13587) clears `code-quality` on `main`, re-run full `quality-gate` on dependent CI PRs.
- Optional: refresh this JSON weekly until failure rate stabilizes below the epic threshold.
