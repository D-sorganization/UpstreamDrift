# Experimental v2 Bundle Representation

This separate research codec is not the canonical v1 loader, a forward replay,
or an acceptance receipt. A caller must not interpret a successful load as
proof of feasibility, contact, model provenance, or independent simulation.

The rollout grid `time` and the reference grid `reference_time` each contain
at least two strictly increasing samples. The reference grid covers the rollout
interval; the three reference arrays use that grid rather than the rollout
grid. An impact is either an in-interval `t`, `q`, `qd` event, or exactly
`{"status": "none"}`. The latter records the supplied event status; the codec
does not determine whether contact actually occurred.

`migrate_v1` copies the original recorded event and gives the reference arrays
the original common time grid. It refuses a v1 event outside the rollout rather
than replacing it with a fabricated no-impact event. An arrays digest detects
accidental archive changes, but the digest in the manifest does not authenticate
the manifest. Consumers requiring provenance must obtain an expected manifest
digest and model identity through an independently trusted channel.

Motor-specific assets and controller-state completeness, physical gates,
independent runtime replay, schema integration, and actual best-match evidence
remain future work. This branch does not close an epic or qualify a PR.
