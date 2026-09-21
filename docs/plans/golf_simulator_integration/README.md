# Model-Driven Golf Simulator Integration

## Decision and Delivery Status

**Epic:** [#10188](https://github.com/D-sorganization/UpstreamDrift/issues/10188).
**Planning PR:** [#10201](https://github.com/D-sorganization/UpstreamDrift/pull/10201).
**Status:** proposed architecture and implementation handoff; no outbound
GSPro client has been implemented or live-qualified by this change.
**Reviewed:** 2026-09-15, baseline
`395d3de876d8f87b7b00785ef2f430d0e9387ce7`.

Yes: UpstreamDrift has useful model, impact, flight and rendering components
from which to build a GSPro integration. GSPro exposes a public shot-input
interface. Deliver model-generated launch conditions to that interface and
let GSPro simulate and display the resulting flight. Start with a driving
range and operator-managed course play. Scientific model qualification and
successful shot delivery are separate gates.

The supported initial visual design is a synchronized UpstreamDrift golfer
view beside GSPro, on a second monitor, or in a separately composed display.
Arbitrary golfer meshes, skeletal animation, cameras and ball trajectories
inside GSPro are **not documented capabilities of its public shot API**.
Treat native integration as a vendor-dependent research extension. A screen
composition is not a golfer inserted into GSPro's 3D scene.

## Reading Order

1. [Feasibility and Evidence](FEASIBILITY.md): verified capabilities and gaps.
2. [Architecture and Contracts](ARCHITECTURE.md): lead-owned hard decisions.
3. [Delivery Backlog](BACKLOG.md): child issues, dependencies and acceptance.
4. [Acceptance Runbook](ACCEPTANCE_RUNBOOK.md): offline and licensed checks.
5. [Support Matrix](SUPPORT_MATRIX.md): supported topologies, boundaries, and runbooks.
6. [Commercial Simulator Evaluation](COMMERCIAL_SIMULATOR_EVALUATION.md): E6, Creative Golf, TrackMan, and Flight Relay evaluation.
7. [Native Avatar and Course Feedback Research](NATIVE_AVATAR_COURSE_FEEDBACK_RESEARCH.md): native mesh/rigging, course telemetry, aim control, and companion presentation.
8. [Next Agent Instructions](NEXT_AGENT.md): bounded worker tasks and turnover.

Current execution state lives in [the canonical handoff](../../../AGENT_HANDOFF.md)
and [development log](../../development/DEVELOPMENT_LOG.md), entry DL-#10188.
This directory is a proposal, not a replacement scientific design manual.

## User Outcomes and Release Boundaries

| Milestone | User Outcome                                   | Acceptance Boundary                                                                                |
| --------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| M0        | Understand installed GSPro compatibility       | Versioned protocol characterization and license-mode check                                         |
| M1        | Send an explicit test shot to a licensed range | Validated codec, robust transport, visible flight and receipt                                      |
| M2        | Replay a model swing and submit its impact     | One provenance-linked launch event; model eligibility evidence                                     |
| M3        | Choose GSPro or the local reference viewer     | Same application service, contract suite and UI capability model                                   |
| M4        | Operate on a local PC or simulator host        | Windows packaging, authenticated remote bridge, support runbook                                    |
| Bonus A   | Animate the actual model within GSPro          | Vendor-supported SDK/extension evidence and separate feasibility gate                              |
| Bonus B   | Let a controller play autonomous rounds        | Authorized course/lie/result/control interface; manual context enables operator-assisted play only |

Core completion requires M0–M4. Bonus outcomes may conclude as documented
unsupported capabilities; they must not delay a useful companion integration
or be marked implemented without evidence. Another commercial simulator is
optional: local reference + GSPro prove destination interchangeability first.

## Success Measures

- One model run, one immutable replay and one impact identify each submitted shot.
- Duplicate user clicks, seek/replay and reconnection do not silently add shots.
- Every shot ends with an auditable receipt, rejection, cancellation before send,
  or explicitly uncertain delivery; a socket write alone never means success.
- Invalid or scientifically unqualified model inputs cannot masquerade as
  qualified shots. Clearly labeled manual/demo inputs remain useful for setup.
- UI selection changes adapters, not physics, launch conversion or shot policy.
- Licensed visual acceptance includes straight, left/right, chip and putt
  scenarios; an unsupported scenario remains disabled in the support matrix.

## Scope Exclusions

Do not modify GSPro binaries, scrape private game memory, ship GSPro assets,
duplicate existing physics kernels, or create an alternative launch-monitor
analytics implementation. Do not add unattended online/tournament submissions.
Do not imply exact agreement between UpstreamDrift and GSPro flight models.
The user requested review, a detailed epic and turnover; this PR delivers that
planning foundation and delegates acceptance-document preparation, not the
entire production integration.
