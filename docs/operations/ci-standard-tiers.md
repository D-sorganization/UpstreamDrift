# CI Standard Fast and Full Tiers

## Scope and Required Check

Issue #13577 separates event tiers without changing the required check name,
`quality-gate`, runner policy, test tolerances, coverage floors or test commands.
The reviewed required-check export is Repository_Management
`docs/operations/ci-required-checks-live.json`, collected 2026-10-21 at 06:06 UTC:
UpstreamDrift's exported required context is `quality-gate`. No repository
settings are changed by this implementation.

| Event | Tier | Full-Only Jobs |
| --- | --- | --- |
| `pull_request` | Fast | Expected to be skipped |
| `push` | Fast | Expected to be skipped |
| `merge_group` | Full | Required for code changes |
| `schedule`, `workflow_dispatch`, unknown events | Full | Required for code changes |

The full-only jobs are `security-scans`, `tests` (the integration matrix),
`rust-wheel-parity`, `core-only-install` and `alembic-postgres-round-trip`.
Their existing steps, dependency installation and tests are unchanged. Lint,
structure and pinning guards, the sharded unit gate, the always-on minimal lane,
frontend tests, Rust quality checks and consumer/contract checks remain on PRs.
Existing fork restrictions and docs/publication routing still apply.

The event provider is vendored byte-for-byte from Repository_Management at the
revision and SHA-256 recorded in
`scripts/config/ci_event_classifier_provenance.json`. It is separate from the
existing OpenSim lane classifier. Only its `tier` output is used: the existing
changed-path detector retains docs/publication authority. An empty PR diff
retains every path-specific gate; a failed diff or classifier fails its job.

## Fail-Closed Aggregation

One `quality-gate` job runs with `always()` on PR and merge-group events. Its
static dependency graph includes both tiers, so failed prerequisites and
unexpected child conclusions reach the aggregate. Expected fast-tier skips are
allowed only for the five named full-only jobs. A failed or cancelled child is
never excused as an expected skip; a skipped required full-tier child fails.
Missing or invalid tier output also fails. The aggregate checks that the tier
matches the actual event and rejects docs-only code waivers on non-PR events. The always-on lane and agent-context
check remain mandatory, including docs-only PRs.

## Known Failures Before Quarantine Adoption

This change adds no quarantine, rerun, `continue-on-error`, marker exclusion or
new baseline entry. Existing unit-gate quarantine ratchets remain unchanged;
their entries may only be removed under the existing policy. The OpenSim
failure triage is tracked by #13582 and its merged implementation #13589.
Reproducible defects and infrastructure failures are not classified as flakes.
Until #13580 implements the fleet quarantine standard, a failure of an executed
required child blocks its tier. Moving an integration job to the full tier
does not declare that job green or qualified.

## Validation and Live Acceptance

The workflow tests execute the production aggregate shell with fixture child
conclusions for fast/full, docs-only and fork cases. They reject failed,
cancelled and unexpectedly skipped full-tier children, invalid tier output,
and unexpected heavy-job failures in the fast tier. Classifier tests check its
pinned bytes, event mapping and conservative empty-diff behavior.

```bash
python3 -m pytest -o addopts= --confcutdir=tests/ci \
  tests/ci/test_ci_standard_workflow.py \
  tests/ci/test_required_check_contract.py \
  tests/ci/test_classify_event_tier.py -q
```

These are executable software checks, not live GitHub scheduler evidence.
The first actual merge-group canary must demonstrate that a failing full-tier
child prevents approval. No intentional failing canary, live acceptance or
post-deployment speedup is claimed here. #13577 remains open for that evidence.

## Observation Window and Rollback

UD maintainers own the seven complete UTC days after the workflow is deployed.
Record the deployment commit and exact window bounds. Collect PR workflow-run
creation timestamps and `quality-gate` job completion timestamps with complete
pagination; retain run IDs, attempt numbers, event, head SHA and conclusions.
For successful PR runs, measure wall time from run creation to aggregate
completion (including queue and dependency wait), and report the nearest-rank
P90, sample count and collection coverage. Report failed, cancelled, timed-out
and skipped attempts separately; none belongs in the successful-duration
denominator. Keep retried attempts identifiable rather than counting a rerun
as an independent PR arrival. An incomplete or empty sample is inconclusive.

The provisional fast-tier P90 target is 15 minutes. Investigate or revert this
workflow change if a complete seven-day sample exceeds that target, if the
full-tier canary permits a failed child, if required contexts stop reporting,
or if the required-check failure rate worsens against the stated baseline.
A safety-contract violation warrants immediate rollback without waiting for
the seven-day window. Revert the workflow and classifier adoption together;
do not alter branch protection, disable required checks, widen tolerances or
cancel other sessions' CI as a workaround. Post-deployment evaluation is not
a merge gate and remains outstanding until measured evidence is attached.
