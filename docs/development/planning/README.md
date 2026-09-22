# Future Development and Validation

This folder preserves future ideas and external experiments for Board review.
Deferred does not mean completed. Do not fabricate physical measurements,
participant trials, plant approvals, calibration, or subjective preference data.

- `catalog.json` is the machine-readable index; Markdown records hold rationale,
  evidence, required resources, independent software scope and acceptance.
- `source-<issue>.json` preserves the original issue before deferral.
- `DV-<issue>.md` is its stable plan. Ideas without source issues can be Markdown
  drafts; no active issue is required merely to retain an idea.
- Software remains in active issues. Mixed issues stay open with links here.
- Board approval and verified prerequisites allow activation, not a success claim.
- Publish on the default branch before closing an external-only issue as
  `not_planned`, labelled `roadmap`, with a durable record link.
- Project Steward links these plans as `parked` charter features and lists
  resource decisions in `docs/project/STATUS.md` under `Decisions Needed`.
- Coding agents skip deferred experiments. Only a named human/equipment owner
  can supply unavailable physical inputs. Existing release gates still apply.

Validate from Repository_Management using
`python -m shared_scripts.deferred_planning <this-folder>`.
The central procedure is `docs/fleet-deferred-validation.md` in that repository.
Keep private evidence in its existing controlled store; do not copy it into a
public proposal. Record reactivation decisions and exact evidence references.
