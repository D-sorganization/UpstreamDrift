# Open-Release Qualification

## Qualified Scope

The release bundle provides a deterministic model ladder from the analytical
double pendulum through forward planar two-hand, moving-base/flexible-club,
coupled forward modal-shaft, synthetic distributed-shaft, matched-task
arm--wrist allocation and phenomenological transmission, reduced spatial
common-state, coupled
two-engine spatial forward-contact, uncertainty/control, and synthetic
experimental-readiness tiers. The CLI lists
the canonical command for each
preset, and the manifest hashes source, data, figures, chapters, and the
rendered article.

Run:

```bash
python -m scripts.research.proximal_distal_energy.qualify_open_release list-presets
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
```

`validate` fails on a missing file, changed bytes, unsafe path, or record
mismatch. Regeneration is an explicit `write` action after all scientific and
visual gates pass; validation never silently updates expected hashes.
The final lossless compaction gate preserves page, URI-link, and outline
counts. It applies no arbitrary manuscript-size ceiling; repository CI instead
checks PDF identity and GitHub's 100,000,000-byte hard file boundary.

## Claim Status

- Planar interaction-dynamics and negative-couple feasibility are supported at
  their declared tiers.
- Geometry response is supported through reduced spatial common-state inverse
  dynamics.
- Reduced/distributed shaft response is supported for a synthetic structural
  case, not calibrated equipment.
- Same-state arm--wrist actuator allocations can produce the same declared
  club moment while changing internal loads. The preload-continuity advantage
  is conditional on the declared dead-zone transmission family; neither result
  identifies a scapular or muscle strategy.
- Passive post-killswitch contact persistence is supported only for the
  declared reduced spatial carriage model; articulated anatomy remains
  untested.
- A universal control strategy is unsupported.
- Human experimental predictions remain untested.

## License and Source Boundaries

Repository-authored code and documentation are released under the repository's
MIT license. Bibliographic records link to publisher or archive sources but do
not relicense external papers. The WSCG deck and extracted tables are registered
project-originated evidence with their source boundary preserved. No private
participant data are included or authorized for public release.

## Open Completion Gates

The manifest records, rather than conceals, five open gates: subject-scaled
articulated spatial contact with calibrated grip and distributed shaft; an
equipment-calibrated distributed beam and grip coupled into a subject-scaled
forward solve; measured tissue-level preload and slack identification;
governed held-out human evaluation; and external archival deposit with a
persistent identifier. The last item requires an external publication action
and is not represented as complete.
