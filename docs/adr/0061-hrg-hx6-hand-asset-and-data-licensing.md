# ADR-0061: HRG-HX6 Hand Asset and Anthropometric Data Licensing

- Status: Proposed (human sign-off required; not legal advice)
- Date: 2026-11-07
- Decision Makers: RD Barb (draft); repository owner (sign-off pending)
- Related Issues/PRs: #13763, #13757, #13760, #13762, ADR-0057

## Context

The HRG-HX epic (#13757) needs hand geometry and hand-size data. Issue
#13763 requires a licensing decision before ANSUR-derived constants ship
(HX3) or hand assets are vendored (HX5, ADR-0057). This ADR records the
licence text found for each source, the decision proposed for each, and
what still lacks verification. It is an engineering record, not legal
advice, and makes no blanket redistribution claim.

Two facts found while drafting change the starting point:

1. **ANSUR-derived constants already ship.** HX3 (#13760) landed
   `src/shared/python/hand_size.py` and entries in `core/physics_constants.py`;
   ANSUR II ratios also appear in `model_appearance/foot.py`,
   `model_appearance/pelvis_width.py` and `estimation/dime_human_priors.py`.
   The sign-off this ADR requests is therefore retrospective for those files.
2. **MyoHand geometry is already in the tree.** `shared/models/myosuite/myo_sim/`
   (VERSION `0.1.0`) contains `hand/myohand.xml` and its meshes. ADR-0057
   treats MyoHand as excluded "until HX6 verifies" its lineage; that
   exclusion applies to the full-body hand option, not to this existing
   research copy.

## Licence Evidence (Retrieved 2026-11-07)

| Source | Licence as published | Exact text located | Verification status |
| --- | --- | --- | --- |
| MANO (MPI-IS) | "Software Copyright License for non-commercial scientific research purposes" (mano.is.tue.mpg.de/license.html) | Use "for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects"; "shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part". Commercial: ps-license@tue.mpg.de | Licence page read directly; no version number published |
| SMPL-X (MPI-IS) | Same title, © 2024 Max-Planck-Gesellschaft (smpl-x.is.tue.mpg.de/modellicense.html) | Same research-only scope; excludes "commercial, pornographic, military, or surveillance" purposes; no distribution "except that you may make one copy for archive purposes only". Commercial: smpl@max-planck-innovation.de or Meshcapade | Licence page read directly; no version number published |
| myo_sim wrapper | Apache License 2.0 (`shared/models/myosuite/myo_sim/LICENSE`; fetched MyoFullBody pin `93b0ca8f…` in `myofullbody/assets.py`) | In-tree LICENSE file | Verified in tree |
| MyoHand geometry lineage | `myo_sim/hand/README.md`: converted from OpenSim MoBL (simtk.org/projects/upexdyn) and 2nd-Hand (simtk.org/projects/hand_muscle) | SimTK project pages list MoBL-ARMS as "Creative Commons ANC Use Agreement" and 2nd-Hand as "MIT Use Agreement"; full agreement text not retrieved | **Not verified.** `myofullbody/assets.py` records MoBL-derived terms as "BSD-3 non-commercial", which conflicts with the SimTK label; the conflict must be resolved from the agreement text |
| ANSUR II (2012 survey, public 2017) | US Army survey; Gordon et al. 2015, NATICK/TR-15/007 | DTIC-indexed ANSUR II reports read "Approved for public release; distribution is unlimited" (search-index text; DTIC returned 403 to direct fetch). PSU OPEN Design Lab page says the data "were made available publicly in 2017" and gives no licence terms | Distribution statement **not read from the primary PDF**; CSV hashes recorded in `hand_size.py` |

## Decision (Proposed)

1. **MANO and SMPL-X: no meshes, parameters, blend shapes or derived
   fits in the repository, test fixtures, release artifacts or CI caches.**
   Both licences forbid redistribution and commercial use. At most, a
   later issue may add an optional loader that reads a user-supplied local
   path and never copies, caches or uploads the files. Code that already
   declines SMPL-X (`model_appearance/skinned_body.py`) stays as it is.
2. **ANSUR II: summary constants may ship** (ratios, means, standard
   deviations and percentiles with citation and file hashes), on the
   basis of the public-release statement. **Per-subject rows must not be
   committed.** The 3-D scans are not public and are out of scope. Before
   this ADR is accepted, someone must read the distribution statement from
   the primary NATICK/TR-15/007 PDF and add its exact wording here.
3. **MyoHand: no new vendoring, and no use as a shipping full-body hand
   option** until the MoBL-ARMS and 2nd-Hand agreement texts are read and
   the "Creative Commons ANC" against "BSD-3 non-commercial" conflict is
   settled. The existing `shared/models/myosuite/myo_sim/hand` copy stays
   research-only (HRG-GS-M8 fixture, #13330) under the owner's 2026-10-07
   non-commercial ruling recorded in `myofullbody/assets.py`. A commercial
   deployment must re-clear it.
4. **Provenance rule for any future hand asset:** record the upstream
   revision, file hashes, wrapper licence, geometry lineage with its
   licence text, and redistribution rights before it is committed,
   following ADR-0057 acceptance item 6.

## Alternatives Considered

| Option | Decision |
| --- | --- |
| Vendor MANO/SMPL-X hand meshes for realism | Rejected: the licence forbids redistribution and commercial use |
| Remove all ANSUR-derived constants until legal review | Rejected for now: the public-release statement supports summary statistics; revisit if the primary text says otherwise |
| Accept MyoHand for full-body use based on the Apache-2.0 wrapper | Rejected: the wrapper licence does not settle the OpenSim geometry lineage |

## Consequences

- Positive: HX3 constants get an explicit, auditable basis; HX5 has a clear
  gate; nothing new is redistributed.
- Negative: a realistic articulated hand stays blocked until the SimTK
  agreements are read; the MANO/SMPL-X route needs a commercial licence.
- Follow-up: (a) quote the primary ANSUR II distribution statement;
  (b) retrieve and quote the MoBL-ARMS and 2nd-Hand use agreements and fix
  whichever of `assets.py` or this ADR is wrong; (c) get owner sign-off and
  move Status to Accepted.

## Validation

This ADR is documentation only and changes no runtime behaviour. Licence
pages were retrieved on 2026-11-07, as cited above; rows marked "Not
verified" are not clearance. Issue #13763 stays open until the human
sign-off required by its acceptance criteria is given.
