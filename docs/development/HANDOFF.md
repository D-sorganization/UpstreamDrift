# Implementation Handoff — Fail-Closed Legacy Evidence Qualification & Dynamic Inertia Digest

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10799`
- Branch: `fix/10799-10800-qualification-integrity`
- Baseline commit: `a9651a161c`
- Implementation commit: `SELF`
- Pull request: not created
- Governing issue/epic: #10799, #10800 (parent review on PR #10798; TB-09 #10594, TB-10 #10595)
- Session: `antigravity-20260924-remediation-tour-baselines`

## Objective and Status

- Objective: Address bot review feedback on PR #10798:
  1. Keep qualification strictly fail-closed for unsigned legacy evidence (#10799) by eliminating silent auto-migration from `IndependentBaselineQualifier.qualify()`, and ensure `migrate_legacy_package` explicitly leaves packages unverified (`UNVERIFIED`, `has_native_replay=False`) until native evidence is regenerated.
  2. Hash actual pendulum dynamics inertia parameters (#10800) in `fixed_inertia_hash` (segment masses, center-of-mass ratios, rotational inertias) via `compute_pendulum_inertia_hash` using rollout `DoublePendulumDynamics` parameters rather than just link lengths or model labels.
- Status: Complete / ready for PR
- Completed:
  1. Removed `auto_migrate` parameter and auto-migration from `IndependentBaselineQualifier.qualify()`.
  2. Updated `migrate_legacy_package` to mark `statuses` with `scientific_qualification=ScientificQualificationStatus.UNVERIFIED` and `has_native_replay=False`.
  3. Implemented `compute_pendulum_inertia_hash()` in `src/engines/physics_engines/pendulum/python/motion_matching/qualification.py` digesting segment masses, COM ratios, and rotational inertias from actual `DoublePendulumDynamics` parameters.
  4. Updated `_assemble_baseline_package` and `generate_baseline_package_for_target` to pass `dynamics` and compute `fixed_inertia_hash` from actual dynamics parameters.
  5. Added regression and unit tests in `tests/unit/tour_baselines/test_qualification.py`.
  6. Updated `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md` (DL-#10799), and this `HANDOFF.md`.
- Remaining: Push branch, create pull request with auto-merge, and release agent lease.

## Files and Decisions

- Files changed:
  - `src/shared/python/tour_baselines/qualification.py`: Removed auto-migration from `qualify()`; updated `migrate_legacy_package` to set `UNVERIFIED` and `has_native_replay=False`, and compute pendulum inertia digest via `compute_pendulum_inertia_hash`.
  - `src/engines/physics_engines/pendulum/python/motion_matching/qualification.py`: Added `compute_pendulum_inertia_hash()`; wired `dynamics` through `_assemble_baseline_package`.
  - `tests/unit/tour_baselines/test_qualification.py`: Added assertions verifying fail-closed legacy rejection and unverified status after migration; added `test_pendulum_inertia_hash_reflects_dynamics_parameters`.
  - `SPEC.md`: Documented fail-closed legacy qualification and dynamics inertia parameter hashing.
  - `docs/development/DEVELOPMENT_LOG.md`: Added DL-#10799 entry.
  - `docs/development/HANDOFF.md`: Updated canonical handoff.
- Key decisions:
  - Fail-closed qualification ensures an untrusted or stripped candidate package cannot bypass cryptographic verification without native evidence.
  - Inertia hash digests actual mass/inertia arrays rather than link lengths, guaranteeing that modifications to model physics invalidate stored inertia digests.

## Validation

- `pytest tests/unit/tour_baselines/test_qualification.py` — 23 passed (100% green).
- `pytest tests/unit/tour_baselines/` — 102 passed (100% green).
- `pytest tests/unit/motion_matching/test_acceptance.py` — 21 passed (100% green).

## Blockers and Risks

- Blockers: None
- Risks/assumptions: None (improves integrity verification without changing valid package rollouts)

## Next Steps

1. Run ruff and mypy checks.
2. Push `fix/10799-10800-qualification-integrity` to origin.
3. Create PR referencing #10799, #10800.
4. Enable auto-merge (`--auto --squash`).
5. Release leases on #10799 and #10800 via `scripts.release_agent_lease`.

## Change Log

- `SELF` — Keep qualification fail-closed for unsigned legacy evidence and hash actual dynamics inertia parameters (#10799, #10800).
