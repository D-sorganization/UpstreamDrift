# Impact Shaft Provider Integration (#9912)

## Current Continuation State

- Working directory: C:/Users/diete/Repositories/UpstreamDrift-impact-provider-pin.
- Branch: feat/9912-impact-provider-pin; implementation commit b6107f8e2; PR not created.
- Base: 6e3610a9b; Tools candidate: 608e85b249e6f61238ac96abbe7dc37428629b9e.
- Governing issue #9912, development entry DL-#9912, parent #9703/#9701/#9700.
- Pair: Tools #5133. No shared source is copied or modified in this consumer.

The new tests/shared_contracts/test_impact_shaft_provider.py exercises the strict
golf_club.distributed_shaft/1 public input format and canonical theme API through
the existing provider-resolution harness. The synthetic fixture verifies coupled
stiffness, integrated mass, canonical roundtrip/digest, exact source-byte checks
and preserved unqualified status. Adverse cases reject version changes, missing
calibration fields, numeric strings and attempted qualification promotion.
Theme coverage checks resolved defaults, custom tokens and independent output.

Before changing the old eab74a901 pin, all six tests failed: five missing shaft
module cases and one missing resolved-theme method. After selecting the exact
candidate, all six pass; the whole provider suite passes all 24 cases with five
existing import-alias deprecation warnings. Run with the established Python 3.12
environment, REQUIRE_REAL_TOOLS_REPO=1, TOOLS_REPO_PATH pointing to this worktree's
vendor/ud-tools, and pytest tests/shared_contracts --tools-mode=vendored -n 0
--no-cov. Numerical libraries use one native thread; Qt is offscreen.
Pinned Ruff 0.15.17 check/format passes all 6,840 files, along with manual,
document-catalog, size and title checks. Ten existing packaging/provenance tests
pass. The local development-log validator file is absent despite synced policy;
the central validator at ad9bcb885 reports 40 inherited findings versus 41 on
base, with no new findings after normalizing shifted diagnostic line numbers.
DL-#9830 and DL-#9825 now concisely record their verified merged results; detailed
evidence remains in their existing turnover documents. Other owners' entries
are preserved. The Python-only wheel built from b6107f8e2 installs with its full
declared core dependencies into a clean environment outside the checkout. Both
shaft and theme imports resolve under site-packages. Canonical wire/digest,
coupled inputs, tamper refusal and theme ownership checks pass; pip check passes.
Wheel SHA256: 6f6f259ff9679f921a5a05e515abcb4bb466589221ce6c7aefdc43b9ca656653.
Runtime: Python 3.12.10, NumPy 2.5.3, SciPy 1.18.1, Pydantic 2.13.5.
SKIP_UI_BUILD=1 is the existing Python-provider build path; this evidence does
not qualify a UI/release artifact. The final reviewed provider pin remains pending.

## Remaining Work

1. Incorporate the reviewed Tools repair revision and rerun the provider checks.
2. Repeat installed-consumer validation when the final provider pin changes.
3. Publish the actual paired PR, add its UD-PAIR reference to Tools #5133, and
   finish protected checks before merging in provider-then-consumer order.
4. Continue the full #9703 engine adapters and registered #9704 studies.

This pin does not close physical calibration, flexible impact, acoustic radiation
or blinded sweetness qualification. Tools #5133 still has protected CI failures;
private Gasification checkout access is separately unresolved. Current dirty
files belong to this branch; other worktrees and user-owned source are untouched.
The complete presence read found common handoff/SPEC/development-log overlap
with capture-product sessions #9898/#9913, but no implementation-path overlap.
Each session uses its own worktree; preserve their incoming metadata during
merges. An unrelated rejected identity-change warning is retained in inbox evidence.

## Preserved Incoming Handoff

The following text predates this provider branch and retains its original context.

# Scoped Ubuntu Dependency Installation Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Branch: fix/9894-ubuntu-ci-sources
- Baseline: 213c5a6ca
- Implementation commit: SELF
- Governing issue: #9894
- Pull request: #9896 (draft pending #9890 integration)
- Session: capture-product-01a08427-ubuntu-ci

## Changes and Evidence

Four standard CI dependency steps now use one Bash installer. It copies the
runner-provided signed Ubuntu deb822 source into a temporary directory and
uses isolated APT source, package-index and binary-cache paths for both update and install.
The original thirteen Qt/Xvfb libraries and each job condition are retained.
Signature and hash verification remain enabled; any failed index rejects update.
Lock timeouts and bounded retries remain. Temporary state is removed on exit;
no machine-wide package source files are edited or deleted.

Six real-Bash tests first failed before implementation, then passed with only
privileged commands stubbed. They cover source-copy integrity, scoped indexes,
missing/unsigned source refusal, transient retry, persistent update/install
failure, install suppression after failed update, and cleanup on failure.
Ruff and focused mypy pass. Actionlint reports the same three pre-existing
workflow diagnostics on baseline and changed stdin input. The first pre-push
unit run exposed an uninitialized pinned Tools submodule in this new worktree;
the exact eab74a901 pin is now initialized and all normal pre-push hooks pass,
including Bandit and the required unit subset. Actual Ubuntu CI run 34387111648 completed all four dependency transactions
successfully in 10-14 seconds on dbfd599b6.

## Coordination and Remaining Work

The comparison branch #9890 is being changed by a separate process. Its latest
Chrome source removal is preserved there; this isolated proposal offers a
job-scoped replacement. #9893 owns contributor-guide compatibility. Do not
modify those branches or managed agent policy here. The authoritative design
manual and standing UP-D0/UP-D1 records remain unchanged.

PR #9896 records its unique SPEC row and development-log link. Complete
protected validation and reconcile #9890 before integration. A readable signed
Ubuntu deb822 source is required; unsupported runner images fail explicitly.

## Dependency Integration

Merged the guide follow-up through 0787dde27, which includes instructor head
b4ed1e7d2. Preserved the Sidekick index, all product code/screenshots/benchmarks,
and both development-log entries. Resolved four workflow conflicts in favor
of this job-scoped installer, preserving the surrounding job conditions. Both
SPEC rows are retained. The canonical handoff stays specific to #9894; guide
qualification remains independently recorded in #9893.

The combined tree must pass protected checks before #9896 is ready. #9890 and
#9893 remain dependencies; no runner permissions or fleet closure conflict is
resolved by this merge.

Fifty combined installer/guide/Sidekick tests pass. Document catalog, title, size,
SPEC uniqueness and full-PR architecture checks pass. The development-log entry
for #9894 is concise to remain within the existing 50 KiB documentation limit.

## Protected Completion Candidate

Current main acbb3a0c (#9895) is integrated without dropping managed guidance.
PR #9896 now carries the final comparison controls (#9883), compatible guide
checker (#9892), and scoped installer (#9894). It preserves all code, images,
and benchmark evidence from #9890 and the checker qualification from #9893.
Earlier PRs remain open until their exact changes are verified on main; do not
close them merely as redundant. The public PR description covers this final
scope. Full protected validation remains required before epic #9863 closure.
