# Scoped Ubuntu Dependency Installation Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Branch: fix/9894-ubuntu-ci-sources
- Baseline: 213c5a6ca
- Implementation commit: SELF
- Governing issue: #9894
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
initialize that exact pin and rerun normal hooks. Local tests do not perform actual Linux installs;
protected CI must qualify the real runner transaction.

## Coordination and Remaining Work

The comparison branch #9890 is being changed by a separate process. Its latest
Chrome source removal is preserved there; this isolated proposal offers a
job-scoped replacement. #9893 owns contributor-guide compatibility. Do not
modify those branches or managed agent policy here. The authoritative design
manual and standing UP-D0/UP-D1 records remain unchanged.

Publish a focused PR, record its unique SPEC row and development-log link,
and complete protected validation before integration. A readable signed
Ubuntu deb822 source is required; unsupported runner images fail explicitly.
