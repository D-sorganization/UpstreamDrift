# ADR-0019: Mission-Drift Calculators

- Status: Accepted
- Date: 2026-04-25
- Decision Makers: Architecture Team
- Related Issues/PRs: #3059

> Renumbered to ADR-0019 on 2026-05-23 to remove a duplicate ADR-0005 slot.

## Context

The repository had accrued a large body of code related to industrial process engineering (`calc_backend/` and `upstream_drift_tools/process_calculators/`), including large vendor PDFs for PSA (pressure swing adsorption). This caused "mission drift" away from the core humanoid and biomechanical modeling suite, inflating repo size and slowing CI.

## Decision

We have decided to select Path 2 from issue #3059: split these modules out. The `calc_backend` and `upstream_drift_tools/process_calculators/` directories have been removed from this repository to be hosted in a sibling repo (`upstream-drift-tools`). All vendor PDFs have been deleted from the tree.

## Alternatives Considered

1. Path 1: Keep but isolate. This would involve moving everything to `src/upstream_drift_tools/` as an officially scoped sub-project and moving PDFs to Git LFS. This was rejected because it still burdens the core repository with unrelated domain code.

## Consequences

- Positive: Reduces repository footprint, clarifies the project's core mission, eliminates large binary vendor documentation from the git tree, and speeds up cloning/CI.
- Negative: Users who need both the biomechanical tools and the industrial process calculators must now manage multiple repository checkouts.
- Follow-ups: Set up the new sibling repository and migrate any outstanding CI pipelines for those calculators to the new repo.

## Validation

A CI check has been added to enforce that no `.pdf` files larger than 1MB can be committed to the tree going forward.
