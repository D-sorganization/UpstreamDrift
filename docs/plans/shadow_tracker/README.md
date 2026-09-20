# Shadow Tracker

**Status: Image-Contract Handoff Ready; Fitter Runtime Not Implemented.**

Epic: [#10122](https://github.com/D-sorganization/UpstreamDrift/issues/10122).
Source inspection baseline: `f8daa71aa263a60c54c785b1ac2d4bc060eb2a71`,
2026-09-14. Recheck interfaces at the implementation commit.

Shadow Tracker turns video of a golfer into timestamped body and club outlines,
fits a subject and initial state, then searches for controls whose **continuous
forward simulation** projects onto those outlines throughout the swing. The
output is an inspectable, reproducible, physically admissible motion hypothesis
with explicit uncertainty. Historical footage and newly captured footage share
the same evidence model, but support different strengths of conclusions.

Here, “shadow” means the golfer's image silhouette, **not a cast ground shadow**.
“Same silhouette throughout” means matching the changing observed silhouette
at each timestamp, not holding the address outline constant.

## Start Here

For current review findings and next tasks, read
[Current Turnover](TURNOVER_CURRENT.md) and
[Continuation Prompt](CONTINUATION_PROMPT.md). Several prototypes have merged;
real segmentation/rendering/ingestion acceptance and the fitter remain incomplete.

| Need                                           | Document                                        |
| ---------------------------------------------- | ----------------------------------------------- |
| Product Intent and Epic Acceptance             | [Epic](EPIC.md)                                 |
| Critical Feasibility and Literature            | [Feasibility](FEASIBILITY.md)                   |
| Pipeline, Optimization, and Decisions          | [Architecture](ARCHITECTURE.md)                 |
| Existing Model and Product Boundaries          | [Integration Map](INTEGRATION.md)               |
| Records, Units, Failure States, and Provenance | [Data Contracts](DATA_CONTRACTS.md)             |
| Archive Mining and New Capture                 | [Capture and Archives](CAPTURE_AND_ARCHIVES.md) |
| Test Strategy and Scientific Gates             | [Validation](VALIDATION.md)                     |
| Ordered Work and Acceptance Tests              | [Work Packages](WORK_PACKAGES.md)               |
| Agent Pickup, Review, and Turnover             | [Agent Handoff](AGENT_HANDOFF.md)               |
| Machine-Readable Task Index                    | [Roadmap](roadmap.json)                         |

## Installation

For immediate worker pickup, use [Ready Tasks](READY_TASKS.md) and the
[Image-Evidence Contract Freeze](CONTRACT_FREEZE.md). Read
[Qualification Findings](QUALIFICATION_FINDINGS.md) before touching model code.

The setup now includes a tested model-qualification experiment under
`scripts/shadow_tracker/`. There is no installable Shadow Tracker fitter,
launcher tile, segmentation pipeline, or model download.
Use the repository's normal development environment for future implementation.
Optional segmentation, rendering, and engine dependencies must remain lazy and
version-pinned when selected by their respective work packages.

## Usage

Pick the first unblocked work package, read its narrow source boundary, claim its
GitHub issue, and follow the handoff. The future user workflow is:

1. Import a local video or reviewed archive entry, or open a Capture Rig session.
2. Review the golfer identity, usable swing interval, timing, masks, and cameras.
3. Fit the body shape and initial state; inspect alternative depth hypotheses.
4. Run bounded forward-dynamics fitting and independently replay candidates.
5. Compare source/mask/render overlays and uncertainty before exporting.

Implementation status lives in GitHub. The roadmap is a dependency index, not
proof of completion. The epic stays open after this planning setup is merged.
