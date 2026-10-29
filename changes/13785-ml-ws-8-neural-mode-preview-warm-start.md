---
issue: 13785
summary: "ML-WS-8: ProposalInitializer protocol and --neural-mode preview warm-start hook into MJX knot optimiser"
---

- Added `ProposalInitializer` protocol and deterministic `StubProposalInitializer` in `src/shared/python/motion_matching/proposal_initializer.py`.
- Implemented `validate_proposal` checking proposal shape, finiteness, and bounds with explicit contract error messages.
- Wired `--neural-mode preview` to resolve a proposal initializer and inject `init_delta` into `optimise_reference`.
- Implemented graceful classical fallback on invalid proposals with `fallback_reason` tracking, failing closed when `--no-neural-fallback` is set.
- Recorded `neural.proposal` receipt metadata `{source, sha256, accepted, fallback_reason}` in `mjx_optimisation_receipt.json` and pipeline receipt.
- Added comprehensive unit tests in `tests/unit/motion_matching/test_proposal_initializer.py`.
