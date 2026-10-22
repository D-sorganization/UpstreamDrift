# Development Admission Report Publication

## Purpose and Authority

Issue #13618 installs the protected report publisher from Repository_Management
#2299. A report is immutable inventory and a canonical gate result. It is not
permission to dispatch, a feature-slot reservation, delivery acceptance or
scientific qualification. A valid negative report remains a denial.

## Source Pins

- Provider: `D-sorganization/Repository_Management` at `5c573e3804494b47bf71c639107ae80f2cd4d6da`.
- Independently assembled runtime bundle SHA-256: `5a2efd76cd0771e7437c58b6af23346501ee6d3d289f927220d51e257b86236b`.
- Canonical workflow byte SHA-256: `8bb9194ef6fa081ddbd77879ad1ace5cc4c652b9fdbcf82db07372d73d5d5706`.
- Source workflow: CI Standard, numeric ID `216580685`.

The machine-readable authority for this consumer copy is
`scripts/config/development_admission_publisher_provenance.json`; the protected
caller repeats its reviewed runtime pins and tests require agreement.

## Protected Events and Private Provider Access

The caller listens only for completed CI Standard runs and invokes the local
reusable workflow for successful pushes. The canonical callee independently
requires the same repository, successful completed push on the default branch,
expected CI workflow ID, full source SHA and valid provider pins. Other events
publish no artifact. Distinct caller and callee concurrency prefixes isolate
repository, run and attempt identities.

Public repositories cannot directly call an internal reusable workflow. The
local callee is byte-identical to the reviewed private provider; no provider
runtime parser is copied or reimplemented. Explicit reusable secrets map the
existing JULES App ID/private key to a contents-read token restricted to
Repository_Management, minted after validation. Only the provider checkout uses
it, with persisted credentials disabled. No credentials or repository settings
are configured by this change.

## Canonical Copy Drift Guard

The existing Fleet Policy Gate reads the protected private provider using its
existing App integration. Before development-governance mode selection, it
compares the local workflow bytes with that trusted provider checkout using
`cmp`. Missing or changed files fail, even when a local manifest claims a
matching hash. The manifest identifies expected bytes; it does not authenticate
itself. Canonical workflow drift requires a reviewed copy/pin update rather
than an automatic review renewal.

## Artifacts and Consumer Verification

Each publication contains exactly `admission-report.json` (at most 5 MiB) and
`admission-receipt.json` (at most 16 KiB), retained for seven days. Its immutable
name includes both source CI and publisher run IDs and attempts.

The source CI SHA is separate from the publishing workflow's default-branch
SHA. A future consumer must authenticate the protected local caller/callee
revisions, canonical-copy binding, approved provider pins and artifact's
publisher attempt. It then cross-checks the receipt against the independent
source run attempt, report digest and exact current subject revision. A
receipt is not a signature, and a successful publishing job cannot override a
negative report. Missing, stale or malformed data never means empty capacity.

## Validation and Remaining Acceptance

The workflow tests bind source pins, caller secrets and trigger scope, and
execute the actual drift-check shell for matching, modified and missing files.
The pre-implementation run failed seven cases; the focused boundary and
existing workflow contract tests pass. Full canonical pre-PR qualification,
authorized governance-mode selection and natural post-merge publication remain
separate required evidence. Issue #13618 stays open until its live publication
criteria are demonstrated; source merge alone is insufficient.

No Dashboard dispatch is activated. Trusted work classification, atomic
operation-backed reservations and inherited record reconciliation remain
separate rollout gates. The user's first Dashboard dry run and real job remain
reserved. No cap, coverage floor, tolerance, capacity, protection or historical
completion date changes are included.


## Metadata Freshness Limitation

A successful CI Standard run and later metadata-only main commits are different
subjects. Run 39694791031 succeeded for `ac77271c`; the subsequent collation
commit `3fd2a59e` changed SPEC, the development log and its fragment without
starting another code CI push run. Both immutable snapshots were rehearsed with
the same protected provider: the CI snapshot has 742 records and 201 active;
the later snapshot has 743 records and 202 active. Both correctly deny admission.

The present publisher must keep its exact CI subject binding. Do not relabel
that historical snapshot as current inventory. A future enforcement consumer
must reject stale subjects; refreshing metadata-only snapshots needs a separate
qualified publication/provenance contract before enforcement is enabled. This
known limitation does not authorize accepting stale data or treating a CI result
for one commit as evidence for another.
