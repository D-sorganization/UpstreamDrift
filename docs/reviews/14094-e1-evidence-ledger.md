# E1 — Review Evidence Ledger and Reconciliation

Epic: https://github.com/D-sorganization/UpstreamDrift/issues/14094
Owner approval: comment 6509376187, “All 8 are approved”.
Scope: E1 documentation correction only; this does not close the eight-scope epic.

## Provenance and Verification Boundary

- Source: eight-page UpstreamDrift_Comprehensive_Architecture_and_Roadmap_Review.pdf, dated 2026-10-29.
- Parent verified/materialized the PDF and supplied extracted text plus visual/chart observations.
- This worker did not view local PDF pixels. Windows Library installation failed at os.setxattr; no further transfer attempted.
- The original draft had no fresh repository inspection. Integration inspected the specific source paths below at `e1ee48deb8e39171ccfa934bbdbc09924b8342b2` on 2026-10-30; other current-source claims remain unknown. No primary scientific article, benchmark or CI job was independently inspected here.
- “Document inconsistency” means a discrepancy in supplied source evidence, not a verified current repository defect.
- Commit 5dd695f8 identifies limited source excerpts delivered to panels; it is not asserted to be current HEAD.
- Completed review panels: th_8b23fac6e169 and th_cacbe94898b1.
- Each panel: one round; four Codex gpt-6-astra experts and Codex moderator; 4/4 participation; quorum met; formal consensus false.
- Both panels: Architect/Scientist/Skeptic partly, Operator agree. Neither granted unconditional roadmap endorsement.
- Owner subsequently approved E1–E8 bounded scopes; no child issues or broad implementation follow automatically.
- Local panel receipts document discussion completion, not repository verification or scientific qualification. Admission returned `created_by: __loopback__`; these receipts do not establish agent-token authentication. No new panel calls were made for integration.

## Verification Vocabulary

- D: established from parent-extracted document text or parent visual observation.
- P: supplied parent/operator receipt; not independently refreshed by this worker.
- A: reproduced arithmetic with explicit assumptions.
- U: current state, underlying measurement, or interpretation unverified.
- S: source text inspected at the pinned integration snapshot; not runtime validation.
- A discrepancy can be D/A while its proposed replacement remains U.
- “Measured” below means described as measured by the source; it does not certify the measurement.
- Proposed, projected, skipped, executed, and passing are distinct statuses.

## Claim Ledger

| ID  | PDF location / precise claim                                                                | Provenance / status                                                                                                 | Reconciliation and required evidence                                                                                                                                                                                                                                                                                                                               | Owner / dependency                |
| --- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| L01 | pp.1,3: all 91 open issues; p.3 chart HCM=3 versus p.4 table HCM=4                          | D: parent text and chart observation; U: authoritative snapshot                                                     | Preserve 3 and 4. Need dated issue export, chart source rows, category rules and unique-ID reconciliation. Do not choose either value from arithmetic alone.                                                                                                                                                                                                       | E1; current-source access pending |
| L02 | pp.3–4: cluster totals 92 versus claimed 91                                                 | D/P: parent summed cluster observations; A: difference=1; U: independent recount                                    | Full 12-cluster series is not available in retained extraction. Supplied total92 is a reported input, not a new sum. Need complete chart/table values plus issue IDs and overlap policy.                                                                                                                                                                           | E1; export and figure source      |
| L03 | pp.1,5: reflection consumes32% total runtime and causes5.8x slowdown; p.8: >=5x cache gate  | D; A: conditional incompatibility; U: workload/denominators                                                         | If only that32% is eliminated, overall ceiling1.470588x. If95% of it is eliminated,1.436782x. Keep5.8x as unreconciled; may use a different denominator or include other changes. Request raw profile, workload, baseline and timing scope.                                                                                                                        | E4 benchmark protocol             |
| L04 | pp.5–6: Rust26x per aerodynamic force evaluation; p.8 wheel gate                            | D: kernel scope; U: measurements/current routing                                                                    | Retain kernel label. No claim of26x full simulation. Require kernel workload share, setup/transfer overhead, matched accuracy, hardware and raw timings. Signing/wheels do not establish WDAC acceptance.                                                                                                                                                          | E4; WDAC hold                     |
| L05 | p.6: Clarabel8.4x and deterministic polynomial-time convergence                             | D: source assertion; U: measurement and applicability                                                               | Label reported/unverified. Specify equivalent convex-conic problem, feasibility, solution quality, termination and overhead; no extrapolation to arbitrary nonconvex allocation.                                                                                                                                                                                   | E4                                |
| L06 | pp.5–6,8: MJX85–100min baseline to<7min with warmstarts;12.5x                               | D: mixed baseline/projection; A:90/7=12.857x,85/7=12.143x,100/7=14.286x                                             | Preserve projected endpoint. For “under7”, ratios are strictly greater than these bounds. Distinguish cold/warm compile, preprocessing, failures, matched objective/feasibility and hardware.                                                                                                                                                                      | E4                                |
| L07 | p.6 Figure2A speedups; Figure2B solver completeness percentages                             | D/P: visual observation; U: measured/projected provenance and denominator                                           | Label each bar measured-by-source/projected/unverified. Completeness is undefined until numerator, denominator, capability cases, engine/version and test evidence are provided. Do not invent percentages or error bars.                                                                                                                                          | E1 labels; E4/E5 definitions      |
| L08 | p.6: torque-only Pc M^-1 S^T tau called zero-velocity counterfactual                        | D: equation/wording; S: interface text; U: runtime behavior                                                         | Keep equation and flag naming/force-allocation question. Zero velocity alone retains gravity. At the integration snapshot, `_dynamics_interface.py:267-315` already defines v=0 AND u=0 and separates the control-preserved diagnostic; `sub_protocols.py:480-490` agrees. E2 owns further definitions and rank policy; runtime conformance was not tested.        | E2                                |
| L09 | pp.7–8: Hicks peakforce<5%BW, moment<1%BW\*h; qualification receipt                         | D; P: parent discrepancy with PMC4321112; U: primary-source audit                                                   | Proposed thresholds remain unverified. Parent flags external-force/COM-height normalization and peak+RMS. Do not replace attributed thresholds until E3 cites primary locations. Separate provenance, consistency and human validity.                                                                                                                              | E3                                |
| L10 | pp.6,8: activation-zero passive correction, reserve1.34 and spinal4.9BW versus1.2–1.8BW     | D: reported values and projected correction; U: measurements/normalization                                          | Preserve reported values; do not tune to target range as proof. Verify activation/excitation, equilibrium, passive elements, reserve denominator and uncertainty through E3/E8.                                                                                                                                                                                    | E3/E8                             |
| L11 | pp.4,8 and related review claims: friction.85 and5segmentshaft                              | D: assumptions; U: calibration/convergence; exact shaft claim page requires source confirmation                     | Treat as hypotheses. Need independent data, calibration/validation separation, sensitivity and discretization evidence.                                                                                                                                                                                                                                            | E8                                |
| L12 | pp.2–3,8: stepping->autodiff->fullbiomechanics hierarchy                                    | D: proposal; U: implemented constraints                                                                             | ADR excerpts establish intent only. E5 evaluates independent capabilities, frames, units, state and unsupported operations before interface changes.                                                                                                                                                                                                               | E5                                |
| L13 | pp.2–3,8: ~1960shadow files, shadow imports and expiry cohortsOct31/Jan31                   | D: historical claims; S: budget snapshot; P/API: #13989 removed one exception, not files; U: historical file cohort | Preserve approximate count and date context. Snapshot `scripts/config/architecture_budget.json` contains106 exceptions and zero2026-10-31 expiry rows. #13989 removed the `build_outputs` function-lines exception; it did not delete a file cohort. Source/generated/removed file counts remain separate E6 work. Do not start duplicate deadline/migration work. | E6                                |
| L14 | p.7: JWT secret/binding fix; LANcompute, containment/evalDoS, assertions and hardwarelimits | D: reported risks/proposals; U: present defects or enforcement                                                      | A secret alone does not prove request-path enforcement. E7 owns inspection/test matrix. Proposed or skipped checks cannot be passing evidence. No probes/settings changes in E1.                                                                                                                                                                                   | E7                                |
| L15 | pp.1,3–4,8: two activePRs14083/14082; merge14083                                            | D: document snapshot; P: later14083merge                                                                            | Add dated supersession note below. Merged state does not imply scientific/parity validation, and skipped smoke is not a pass. Do not recommend remerging.                                                                                                                                                                                                          | E1 operational notes              |
| L16 | p.8 broad4phase/16week roadmap and53of53 parity                                             | D: targets; U: feasibility/completion                                                                               | Label planned targets, not commitments or achieved outcomes. Inventory denominators and dependencies first; bounded investigations do not authorize broad migration.                                                                                                                                                                                               | E1/E5/E6                          |
| L17 | p.7 security92/100, package/CVE and staleCodeQL claims                                      | D: source assertions; U: scoring method, advisories and current scans                                               | Retain as dated reported claims. Require scoring rubric, authoritative advisory and scan receipt before treating as current security status.                                                                                                                                                                                                                       | E7; no fresh assertion            |

## Dated Operational Evidence Supplied After the PDF

| Receipt | Supplied fact                                                                          | Verification limit / consequence                                                                                                                 |
| ------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| P01     | #14083 merged2026-10-29 20:41UTC; parity-smoke skipped                                 | Parent verified; not refreshed here. Update active/merge recommendation, retain skipped status.                                                  |
| P02     | #14084 CAP12manual merged2026-10-29 21:27UTC                                           | Parent verified; documentation merge does not release CAP11 or prove solver parity.                                                              |
| P03     | CAP11licensing remains held per operator                                               | Hold remains; no permission to lift it.                                                                                                          |
| P04     | SPEC.md349968bytes queue risk already owned by#14082                                   | Parent-supplied; no duplicate archive task. Number is not a locally measured current file size.                                                  |
| P05     | #13989 merged2026-10-28 19:30:56UTC, commit `fed67279cf52574c84cd5c36ce0f130a8ce519b7` | App API diff verified during integration: one `build_outputs` exception removed from architecture budget; no file-cohort deletion.               |
| P06     | Review epic#14094 and approvalcomment6509376187                                        | Issue and approval comment independently read through existing App-only wrapper during integration; no duplicate E1 PR in returned open-PR list. |
| P07     | Two local panels completed; exact JSON receipts retained in task workspace             | Locally observed service result; model discussion is not fresh scientific/source verification.                                                   |

## Reproducible Counting Rule for Later Reconciliation

1. Freeze export timestamp, repository, filters, state and API/source identity. Preserve the raw export hash.
2. Count issues and pull requests separately. A PR is not an issue in the issue-population denominator.
3. Use unique issue numbers for the unique-open-issue total. Preserve duplicate source rows for diagnosis.
4. Declare whether epics and subissues both count. Declare mutually exclusive primary-cluster assignment or explicitly overlapping membership.
5. For exclusive clusters require exactly one assignment per issue and compare membership sum to unique issue total.
6. For overlapping clusters report memberships and unique issues separately; a sum of92 need not contradict91 unique issues.
7. Preserve unassigned, multiply assigned, closed, duplicate and unavailable items as explicit reconciliation categories.
8. Rebuild chart and table from the same snapshot/rules. Do not repair a count by silently dropping a row.
9. Full chart values and issue IDs are not present here, so no independent12-cluster recount or current91-issue certification is claimed.

## Arithmetic Reproduction

Reproduce the arithmetic from the repository root with standard-library Decimal:

```powershell
python3 -c "from decimal import Decimal as D; f=D('0.32'); print(1/(1-f)); print(1/(1-f*D('0.95'))); print(1-1/D(5)); print(1-1/D('5.8')); print(D(85)/7,D(90)/7,D(100)/7); print(D(90)/D('12.5'))"
python3 -c "import json; c=json.load(open('scripts/config/architecture_budget.json')); print(len(c['exceptions'])); print(sum(e.get('expires_on')=='2026-10-31' for e in c['exceptions']))"
```

The second command is a current-checkout query; the recorded106/0 result belongs to the pinned snapshot, not future main.
Input values are explicitly reported source claims, not new measurements.

- HCM supplied chart/table difference:4-3=1.
- Supplied reported cluster-total difference:92-91=1; this is not a reproduced membership sum.
- Removable fraction f=.32: ceiling=1/(1-f)=1.470588...
- Remove95% of that fraction:1/(1-.32\*.95)=1.436781...
- Required removable fraction for5x=.8; for5.8x=.827586...
- Projected under7min ratios exceed12.142857x from85min and14.285714x from100min.
- A 12.5x reduction of90min is7.2min; this does not substantiate under7min.
  Checks certify arithmetic consistency only, not experimental assumptions.
  Current issue counts, completeness denominators, all12cluster sums and benchmark validity remain not evaluated.

## Document Correction Boundaries

- Append this ledger/supersession note or incorporate its labels; retain original claims until supported reconciliation.
- Do not silently rewrite chart values, infer missing bars or claim a refreshed repository audit.
- E2/E3 own scientific definitions/primary-source audit; E7 owns enforcement inspection. E1 records their open questions without performing their work.
- Integration posted an E1-only claim and prepared this isolated documentation branch. No runtime code, child issues, authentication controls, production probes or benchmark workloads changed.
- Integration consists of this Markdown ledger and a change fragment, under500 changed lines; arithmetic is inline to keep the PR documentation-only.
- Any future PR must follow the repository's current claim and contribution workflow when available.
- E1 corrects evidence labels and dated supersession notes; unresolved source collection and E2-E8 investigations remain separate.

## E1 Acceptance Checklist

- [x] Page-level claim ledger with provenance and explicit current-source uncertainty.
- [x] Conflicting counts preserved without choosing an unsupported replacement.
- [x] Measured-by-source, projected and unsupported performance separated.
- [x] Undefined completeness denominators flagged.
- [x] Supplied mergedPR/skippedcheck receipts and deduplication owners recorded.
- [x] Reproducible arithmetic supplied; missing recount inputs explicitly reported.
- [x] Under500lines; no shared checkout mutation.
- [ ] Documentation PR reviewed and merged; no completion claim for the full epic.
- [ ] Authoritative inventory/chart data resolves counts and denominator questions.
