# Shipped Even Issues — Closure Evidence (2026-10-24)

Night-swarm record: fixes already on `main`; acceptance is exercised by the normal CI
unit suite (no duplicate subprocess harness).

## #13579 — UD-CI-014 Engine Docker Cache (PR #13598)

| Field | Value |
| --- | --- |
| Merge SHA | `d5112d5546aaceca2cc69b36f0d9223ce0219a85` |
| Landed | 2026-10-21 (PR #13598) |

**Acceptance pytest nodes:**

- `tests/unit/scripts/test_engine_cache_contract.py::test_docker_build_workflows_use_shared_buildkit_contract`
- `tests/unit/scripts/test_engine_cache_contract.py::test_no_workflow_uses_per_runner_type_local_or_gha_cache`

## #13072 — LIFT-4 Barbell Re-Track (PR #13092)

| Field | Value |
| --- | --- |
| Merge SHA | `600ee7650f7a4842959bb59d7b09140cc6acfa3d` |
| Landed | 2026-10-17 (PR #13092) |

**Acceptance pytest nodes:**

- `tests/unit/lifting/test_barbell_evidence.py::test_retracked_bundle_records_it_follows_its_source`
- `tests/unit/lifting/test_opensim_lift_ik_id.py::test_squat_receipt_range_violations_empty`

## #13104 — OSL-16 Loaded Foot Tracking (PR #13120)

| Field | Value |
| --- | --- |
| Merge SHA | `cb8b736d2e3c0c8f4bd45deafa53896a6f13a5f7` |
| Landed | 2026-10-17 (PR #13120) |

**Acceptance pytest nodes:**

- `tests/unit/lifting/test_loaded_foot_tracking.py::test_loaded_feet_stay_within_the_lift5_ankle_bounds`
- `tests/unit/lifting/test_loaded_foot_tracking.py::test_tracking_converges_in_the_rk4_substep`

## Anti-Phantom Note

GitHub issue bodies for these items cite implementation paths (for example
`src/shared/python/lifting/barbell_runs.py`), not this evidence file. An
evidence-only pull request therefore cannot use `Closes #n` without touching
production code solely to satisfy Rule 3. Prefer merging this document and
closing the issues manually with a link here, or a follow-up that changes the
issue templates / guard policy for shipped closures.
