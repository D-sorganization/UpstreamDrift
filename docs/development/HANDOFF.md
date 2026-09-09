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
