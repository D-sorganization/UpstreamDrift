# NM-03 Episode Storage, Splits and Dataset Views

Governing issue: [#10618](https://github.com/D-sorganization/UpstreamDrift/issues/10618)
(epic [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603)).

Schema: `neural-episode-store/1.0.0`

## What Landed

Versioned, content-hashed episode storage under
`src/shared/python/neural_motion/episodes/`:

- Immutable `EpisodeRecord` with DbC (finite arrays, SI units, allow-listed
  control bases, compact-1.0 27-joint / 189-coeff order)
- Identity channels (`q`, `v`, `u`, `a_native`) separated from predictive
  `q_next`; unavailable channels stay `None` (never silent zeros)
- HDF5 shard store with gzip compression, deterministic episode ids, lazy/eager
  read parity and corrupt-shard fail-closed checks
- Compact-1.0 adapter that preserves 27 / 189 without reinterpretation
- Family-level splits (near-duplicates / augmentations share a family); geometry,
  contact and club held-out strata; real-data eval bucket; source-copy aliases
- Train-only normaliser with immutable resume; thin views (instantaneous
  dynamics, sequence matching, observation masks, feasibility); window cache
  keyed by source + transform version

## Evidence

[`evidence/nm03_episode_store_receipt.json`](evidence/nm03_episode_store_receipt.json)

## Limitations

Software-contract fixtures only. No training, speed claims, or ingestion of
native corpora — that begins at NM-04.
