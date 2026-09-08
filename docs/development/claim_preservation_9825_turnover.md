# Manufactured Claim Preservation: #9825

## Current Ownership

- Parent impact/acoustics #9700; follow-up to #9787 / merged PR #9804.
- Worktree: C:/Users/diete/Repositories/UpstreamDrift-impact-claim-preservation.
- Branch: fix/9825-preserve-reviewed-claims, protected-main base
  9f54c5e0b79d9ca0e81aebf52f90f07ffac832c0.
- Codex lease session impact-acoustics-01a07d8a-claims9825 expires
  2026-09-09T00:57:12.631295Z. Claim was unheld before acquisition.
- The former shared branch and local validated authority 6235789dc are
  preserved. Do not overwrite that branch or certify differing merged bytes
  using its old test results.

## Confirmed Protected-Main Findings

PR #9804 merged as 736ec2189a479fc1a9a7b45078b4d7d371b2590a, from b8da0c024.
The current registration source defines `_refresh_claims`, but `_reconcile`
still removes selected claims and appends new ones without calling it. Existing
helper-only tests pass without exercising that actual path. The current
registry retains reviewed `adjudication_outcome` fields and, for conservation,
`numeric_evidence`; those are at risk on a registration rerun.

The current manufactured record has SHA256
4fbb40ac2302e8111a19fc97903bc1877ab2cb07ced7258b8f78994ca4d627d8, schema 1.0.0,
no execution_profile, and seven source hashes. The producer now declares an
exact native authority profile and an expanded source list. A stale handoff
still described the formerly validated 0c0f3395 record; that is not the merged
artifact. Verify/regenerate with the actual pinned native environment and
governed writers; do not add profile labels to old numerical payloads.

## Implemented Preservation Contract

The exact vendor/ud-tools gitlink eab74a901a7c8467e1997049a73e2cfd2df74428 is initialized without changing the pin. Six real-path failures (6 failed, 5 passed, 6.36 s) demonstrate reordering, dropped reviewed outcomes and acceptance of changed science. The minimal production correction calls the existing `_refresh_claims` before registry mutation and assigns its result after candidate validation. The obsolete removal-set constant is removed; shared temporary-file test setup avoids duplication. All 11 focused tests pass in 1.23 s. No changed scientific assertion is automatically reviewed.

## Native Evidence and Regression

The initial broader audit records 124 passes, two failures and two expected failures. Both inherited expected-failure exemptions are removed; their two required tests fail on stale hashes/missing profile before regeneration. The isolated Linux CPython 3.11.15 authority environment retains NumPy 2.3.5, SciPy 1.15.3, MuJoCo 3.8.0 and Pinocchio 3.8.0. pytest-timeout 2.4.0 is added only as a test dependency; pip check passes. One BLAS/OMP/MKL thread and the unchanged 60-second test deadline are used.

Two independent validated processes produce identical SHA256 15d00b5e81b07b4f65aebe707f06f6686a6daf98a4536fda28d6ba682aa157fd. The canonical producer then emits identical bytes. Model, engines, free-body and constrained-motion numerical results exactly match the old record; existing design fields and limitations also match. The generated record restores its current execution profile, lock digest, 14 source hashes and comparison-policy fields. No profile is pasted into old numerical output. All 128 combined native/provenance/bootstrap/preservation contracts pass without skips or expected failures in 75.74 s.

## Claims and Publication

The actual registration is byte-idempotent and preserves all 328 claim identities/order, scientific fields, reviewed outcomes and numeric evidence. Counts remain 308 supported, five inconclusive and 15 untested. The summary writer changes only the document-evidence tier from 113 to 118; its five tests pass. Regenerated candidate inventory has 1,181 candidates. The single new census-table candidate PD-CAND-5fc9c4d85d6c025c is explicitly reviewed as an editorial projection of verified existing counts, preserving the prior classification and adding no scientific estimand. No unrelated review generator is rerun. Claim validation has zero unadjudicated candidates. All 498 registered numeric literals across 144 claims match their declared evidence; this includes 149 registered values not independently recomputed and 57 reported external values, not 498 new physical measurements.

Quarto 1.8.26 / LuaHBTeX MiKTeX 25.12 rebuild and the canonical optimizer retain 253 pages, 196 URI links and 255 outline entries. PDF bytes: 2,012,367; SHA256 bf855f791e142e9ff84a30af61966bee6a0fcf406fdb7f41616e0de8a6e567e5. Text across all pages matches after whitespace normalization and the sole declared 113-to-118 replacement. All 14 affected extraction/layout pages (18, 52, 55, 114, 133, 149, 194, 237, 248-253) are visually reviewed without clipping or overlap. The generated TeX additionally changes unordered callout options without changing their values. Existing MiKTeX update and deprecated fitz API notices are retained; no environment-wide update is applied.

Canonical release-review and release-bundle writers refresh manifests/checksums and pass computational publication qualification, including rendering all pages. The initial source digest refusal correctly required inventory regeneration after the summary QMD change. Repository Ruff 0.15.17 and formatting pass for 6,764 files. Design-manual status remains structurally valid but blocked-inventory-required; neither that gate nor computational publication is empirical or archival approval.

## Final Registry Integration

The post-refresh 68-test claim/release suite initially finds five failures: the summary still carries the prior paper digest and four legacy-migration contracts reject the new census snapshot. Regenerate the summary after the candidate inventory, then explicitly register only reviewed paper digest 2711476282ffac75ab1ea5d0fd8c7150c245f40df1088cb42b6c6d5dccf02aaa and its single replacement census candidate in the existing migration authority. Retain the prior digest, every explicit claim outcome and refusal of unknown digests/claims. No outcome partition or scientific statement is changed. All 68 tests then pass in 24.20 s; eight inherited Windows alias/deprecation warnings remain. The QMD and reviewed PDF stay unchanged. Refresh release records after this metadata correction; computational qualification passes again.

## Delivery and Next Actions

Source f0bbc4d50d589bbe871150f7de0523c15c02508b is published; normal commit/push hooks pass and the remote SHA is verified. Computational publication validation passes at that exact source. The first push passed security but stopped during unit collection because Python 3.12 lacked h5py. A local ignored .venv inherits the existing general test environment and supplies repository-locked h5py 3.16.0, NumPy 2.2.6 and SciPy 1.15.3; the normal unit hook then passes. The exact Linux scientific authority environment remains separate and unchanged. Research scripts are excluded by the existing mypy hook, which reports no files; do not call that a new strict type-check result.

Ready PR #9826: https://github.com/D-sorganization/UpstreamDrift/pull/9826. Its first CI observation is running, with initial superseded runs cancelled; no failure or success is inferred from pending checks. Apply the repository ci-watch-and-fix skill, retaining the fleet polling interval and no automatic retries of running work. Temporary heartbeat log: impact-9825-ci-watch.log. SPEC is keyed to #9826 and the development log is in_review. Do not close #9825 or the full impact program before protected CI/review and acceptance are verified. Preserve the former shared branch and its distinct historical authority.

## Related Program State

Tools finite-support response is published at 12bcf3d83 (665 Linux tests). Separate-G/C spectra 97d46055c pass 692 Linux golf/API tests and 49 focused controls; normal hooks pass and remote SHA is verified. Turnover bbc27dbe32181dd3d147841784a108fdc1aadeea is also published. AffineDrift #4298 merged normally as d7e51655d47d37a092b1bcd29d25972ce373b244 after required CI passed, with no unresolved review threads. Its auxiliary benchmark workflow performed no measurements despite green workflow status; no performance result is claimed.

Tools #5103/#5106 remain blocked by a private consumer lookup despite green public checks. #8920/#8556 retain bilateral-wrench and parameter-identification gates. Explicit stability assessment, transient/nonlinear impact, useful bandwidth, flexible-head radiation and physical/blinded acoustic validation remain open. No synthetic test supplies physical or listener validation.
