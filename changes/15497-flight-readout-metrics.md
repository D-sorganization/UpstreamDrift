---
issue: 15497
summary: "Derive display-neutral apex, descent, rollout and lateral readout numbers from a BunkerFlight"
dl_state: "in_review"
next_step: "Independent review and CI; leave this slice unarmed per owner instruction"
branch: "claude/15497-flight-metrics"
---

Add `src/tools/bunker_shot_gui/flight_metrics.py`: a pure, GUI-free
`flight_metrics(BunkerFlight | None)` that returns apex height and downrange,
landing descent angle, ground rollout length, total distance and signed lateral
offset in SI. No flight (refused or unflown) gives no metrics (ADR-0032). The
desktop, API and web slices of #15497 can consume it instead of each
re-deriving the numbers. No physics, trace-contract, API, renderer or web
changes. Refs #15497.
