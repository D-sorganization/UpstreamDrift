# ADR-0061: HRG-HX6 Hand Data and Asset Licensing

- Status: Proposed (human sign-off required; see Sign-Off Gate)
- Date: 2026-11-07
- Decision Makers: RD Barb (draft); repository owner (sign-off)
- Related Issues/PRs: #13763, #13757, #13760, #13762, #13330

## Context

The HRG-HX epic (#13757) needs hand shape and size data. Three source families
were proposed: ANSUR II / Greiner 1991 anthropometry, the MPI-IS MANO and
SMPL-X parametric body models, and the MyoSuite myoHand asset. UpstreamDrift
is distributed under the MIT License, so anything committed here is
redistributed under terms at least as broad as its own licence allows.

This ADR is an engineering licence inventory, not legal advice. It quotes the
licence text each source publishes as observed on 2026-11-07 and states what
remains unverified.

Two facts already in the repository shape the decision:

- HX3 (#13760) shipped ANSUR II **summary ratios** (means, SDs and percentiles
  of `handlength/stature` and `handbreadth/handlength`) in
  `src/shared/python/hand_size.py`, with sha256 hashes of the source CSVs. No
  per-subject rows are committed. This landed before HX6 sign-off.
- The MyoSuite `myo_sim` tree is vendored under
  `shared/models/myosuite/myo_sim/` (Apache-2.0 wrapper, `VERSION` 0.1.0), and
  the bilateral myoHand research fixture (#13330) lives under
  `shared/models/myosuite/golf/hand/`.

## Licence Inventory

| Source | Published licence (observed 2026-11-07) | Redistribution | Commercial use | Verified |
| --- | --- | --- | --- | --- |
| ANSUR II (2012 survey; Gordon et al. 2015, NATICK/TR-15/007) | PSU Open Lab page states the data "were published internally in 2012. They were made available publicly in 2017." No licence or distribution statement is shown on that page. | Not stated on the page | Not stated on the page | Partial: public release confirmed; the report's distribution statement and US Government work status are **not verified** |
| Greiner 1991 (US Army Natick technical report) | Not retrieved | Not verified | Not verified | **No** |
| MANO (MPI-IS) | "Software Copyright License for non-commercial scientific research purposes", © 2020 Max-Planck-Gesellschaft, no version number | "shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part except that you may make one copy for archive purposes only" | Prohibited; commercial contact `ps-license@tue.mpg.de` | Yes (licence page) |
| SMPL-X (MPI-IS) | "Software Copyright License for non-commercial scientific research purposes", © 2024, no version number | Same no-redistribution clause as MANO | Prohibited; commercial licensing via Meshcapade | Yes (licence page) |
| MyoSuite `myo_sim` wrapper | Apache License 2.0 (`shared/models/myosuite/myo_sim/LICENSE`) | Permitted with notice | Permitted | Yes (vendored file) |
| myoHand upstream lineage: MoBL-ARMS (SimTK `upexdyn`) | SimTK project page lists "Creative Commons ANC Use Agreement" | Full terms not retrieved | The "NC" designation indicates a non-commercial restriction; full text **not verified** | Partial |
| myoHand upstream lineage: 2nd Hand model (SimTK `hand_muscle`) | SimTK project page lists "MIT Use Agreement" | Full terms not retrieved | Full terms not retrieved | Partial |

`myo_sim/hand/README.md` states the myoHand model was converted from both
OpenSim models above, including bone meshes. The Apache-2.0 wrapper licence
therefore does not by itself establish the rights in the geometry it carries.

## Decision (Proposed)

1. **No MANO or SMPL-X meshes, parameters, shape spaces or derived blend data
   in the repository.** Both licences forbid redistribution and commercial use.
   Integration is allowed only as an optional local-only loader that reads
   files from a user-supplied path the user obtained under their own licence.
   Nothing derived from those files may be committed, cached in CI artefacts,
   or embedded in evidence bundles.
2. **ANSUR II summary statistics may ship** as computed ratios with source
   hashes and citation, as `hand_size.py` does today. Per-subject ANSUR rows
   must not be committed until the report's distribution statement is
   verified. This is conditional on sign-off; if the owner declines, the HX3
   ratios are replaced by de Leva (1996) length plus a cited published breadth
   ratio.
3. **Greiner 1991 constants must not ship** until its distribution status is
   verified and recorded in this ADR.
4. **No new myoHand mesh is vendored**, and no myoHand geometry becomes a
   shipping default (consistent with ADR-0057), until the MoBL-ARMS
   "Creative Commons ANC" terms are retrieved and confirm that geometry derived
   from it may be redistributed under this repository's licence. The existing
   vendored `myo_sim` tree and the #13330 research fixture stay as research
   assets pending that review; this ADR does not delete them.

## Sign-Off Gate

Per #13763, human sign-off is required before:

- HX3 data is treated as cleared (decision 2 retroactively covers the ratios
  already merged), and
- any HX5 follow-up proposes vendoring third-party hand assets.

Until the owner records sign-off on #13763, this ADR stays **Proposed** and
decisions 1, 3 and 4 apply as the conservative default.

## Alternatives Considered

| Option | Why not selected |
| --- | --- |
| Vendor MANO/SMPL-X hand shape spaces | Licence forbids redistribution and commercial use |
| Treat the Apache-2.0 MyoSuite wrapper as clearing all myoHand geometry | The README names upstream OpenSim models whose licences differ; at least one is marked non-commercial |
| Block all ANSUR-derived values until a lawyer reviews | The data is publicly released US Army survey data and only aggregate ratios ship; the owner can still reject this in sign-off |

## Consequences

- Positive: the repository keeps a redistributable licence posture; HX5 has a
  clear asset gate; the HX3 ratios have a recorded basis.
- Negative: no parametric hand shape model is available in-tree; articulated
  hands must use self-authored geometry (ADR-0057 palm plus phalanx capsules).
- Follow-up: retrieve the full MoBL-ARMS "Creative Commons ANC" agreement text
  and the NATICK/TR-15/007 and Greiner 1991 distribution statements, then
  update the inventory and request sign-off.

## Validation

Documentation only. Licence texts were read from the MANO and SMPL-X licence
pages, the PSU Open Lab ANSUR II page and the SimTK `upexdyn` and `hand_muscle`
project pages on 2026-11-07, and from the vendored `myo_sim/LICENSE` and
`myo_sim/hand/README.md`. Rows marked not verified were not confirmed against
a primary licence text.
