# Delivery Backlog

## Authority and Planning State

[Epic #10188](https://github.com/D-sorganization/UpstreamDrift/issues/10188)
and its native GitHub sub-issues are the task authority. Each issue contains
scope, ownership, dependencies, checkbox acceptance, tests written first and
turnover restrictions. This document is a navigation/dependency view, not a
second completion ledger. All children were open at planning delivery.

## Work Packages

| Stage | Issue                                                                   | Scope                                        | Dependencies                                         | Suggested Owner         |
| ----- | ----------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------- | ----------------------- |
| GS-00 | [#10189](https://github.com/D-sorganization/UpstreamDrift/issues/10189) | Installed Protocol and Compatibility Profile | None                                                 | Lead                    |
| GS-01 | [#10190](https://github.com/D-sorganization/UpstreamDrift/issues/10190) | Immutable Domain and Capability Ports        | Canonical part independent; GS-00 for vendor details | Worker With Lead Review |
| GS-02 | [#10191](https://github.com/D-sorganization/UpstreamDrift/issues/10191) | Pure Codec and Independent Fixtures          | GS-00, GS-01                                         | Worker                  |
| GS-03 | [#10192](https://github.com/D-sorganization/UpstreamDrift/issues/10192) | Durable Transport and Uncertainty            | GS-00, GS-01, GS-02                                  | Lead                    |
| GS-04 | [#10193](https://github.com/D-sorganization/UpstreamDrift/issues/10193) | Full Impact State and Contact Qualification  | GS-01                                                | Lead/Physics Specialist |
| GS-05 | [#10194](https://github.com/D-sorganization/UpstreamDrift/issues/10194) | Session Service and Local Adapter            | GS-01–03; GS-04 for qualified mode                   | Lead Then Worker        |
| GS-06 | [#10195](https://github.com/D-sorganization/UpstreamDrift/issues/10195) | Replay and One-Impact Submission             | GS-04, GS-05                                         | Lead Then Worker        |
| GS-07 | [#10196](https://github.com/D-sorganization/UpstreamDrift/issues/10196) | API, Desktop and Web Controls                | GS-05, GS-06                                         | Worker                  |
| GS-08 | [#10197](https://github.com/D-sorganization/UpstreamDrift/issues/10197) | Windows and Remote Deployment                | GS-03, GS-05, GS-07                                  | Deployment Engineer     |
| GS-09 | [#10198](https://github.com/D-sorganization/UpstreamDrift/issues/10198) | Licensed Acceptance and Support Matrix       | GS-00–08                                             | Lead Reviewer + Worker  |
| GS-10 | [#10199](https://github.com/D-sorganization/UpstreamDrift/issues/10199) | Other Simulators and Optional Relay          | GS-01, GS-05                                         | Lead Discovery          |
| GS-11 | [#10200](https://github.com/D-sorganization/UpstreamDrift/issues/10200) | Native Avatar and Course Feedback            | GS-00; GS-06 fallback                                | Lead/Vendor Specialist  |

## Implementation Sequence

1. Lead executes GS-00; worker can independently implement the non-vendor
   GS-01 domain contract. Do not hand unresolved signs/framing to the worker.
2. Lead reviews the GS-01 public surface and freezes the GS-00 profile.
   Worker implements GS-02. Lead investigates GS-04 regressions in parallel.
3. Lead owns GS-03 lifecycle and journal implementation. GS-05 integrates the
   shared service and local reference adapter; clearly labeled manual/demo
   operation can precede completed scientific qualification.
4. GS-06 connects the qualified model/run to existing presentation. GS-07
   supplies thin UI/API controls; GS-08 qualifies Windows and remote topology.
5. GS-09 is the real release gate. GS-10 and GS-11 remain optional discovery;
   close research only with an accurate outcome, not a fictitious implementation.

Suggested narrow PRs inside GS-03: bounded framer; durable journal; client
lifecycle; integrated fault tests. Inside GS-04: failing preservation test;
public solver fix; contact extraction; engine evidence. Update child scope or
create further children if one change cannot be reviewed independently.

## Definition of Ready

- Applicable instructions and existing public source/test seams reviewed.
- Child lease checked and posted; no conflicting edits or duplicate PR.
- Dependencies satisfied for the selected slice; contract/profile version named.
- Files owned by the worker, red tests, exclusions and expected outputs recorded.
- All uncertain scientific/vendor claims marked unverified; no guessed goldens.

## Definition of Done

- Tests demonstrate failure before implementation and pass after the fix.
- Always-on boundary validation, public-facade access and shared logic reviewed.
- Appropriate regression, lint, formatting, typing and repository gates pass.
- Required source/semantic contracts and feature-parity entries are updated.
- Canonical handoff and DL-#10188 record exact commit, commands and outcomes.
- Focused PR references the child; feature closure requires implemented
  acceptance. A plan-only PR must not close the core epic or feature children.
- Live claims include actual licensed evidence; simulation/scientific approval
  is not inferred from a successful network transaction.

## Risk Register

| Risk                              | Owner            | Mitigation / Release Gate                                       |
| --------------------------------- | ---------------- | --------------------------------------------------------------- |
| Sparse Vendor Protocol            | GS-00 Lead       | Freeze observed profile; unresolved semantics remain disabled   |
| Silent Extra Shot                 | GS-03 Lead       | Intent journal, no automatic resend, uncertainty recovery tests |
| Incorrect Units or Spin Direction | GS-01/02 Review  | Independent vector goldens and live direction calibration       |
| Peak Speed Mistaken for Contact   | GS-04 Lead       | Proven event extraction and explicit demo provenance            |
| Dropped Impact Parameters         | GS-04 Lead       | Field-influence tests through public solver seam                |
| No Native Avatar or Course API    | GS-11 Lead       | Useful companion view; vendor-gated bonus                       |
| Competing Producers               | GS-08 Owner      | Single-session ownership and operator-visible conflict          |
| Other Simulator SDK Access        | GS-10 Lead       | Capability discovery before implementation; local fallback      |
| Version Drift                     | GS-09 Reviewer   | Versioned support matrix and post-update requalification        |
| UI/Engine Work Collisions         | Each Child Owner | Issue leases and isolated topic worktrees                       |

## Effort Planning

Treat this as several reviewable increments rather than a one-file connector.
Transport and scientific qualification are the highest-risk work. Estimate
implementation only after GS-00; SDK access and model acceptance are external
uncertainties, so calendar commitments before those gates would be misleading.
