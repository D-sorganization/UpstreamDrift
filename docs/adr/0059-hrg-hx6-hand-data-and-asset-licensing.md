# ADR-0059: HRG-HX6 Hand Data and Asset Licensing

- Status: Proposed (human sign-off required; not legal advice)
- Date: 2026-11-03
- Decision Makers: RD Barb (draft); repository owner (sign-off)
- Related Issues/PRs: #13763, #13757, #13760, #13762, #13330

## Context

The HRG-HX epic (#13757) needs hand size data (HX3, #13760) and may later
want articulated hand assets (HX5, #13762, see
[ADR-0057](0057-hrg-hx5-articulated-hand-option.md)). Issue #13763 asks for
a licensing decision that cites the licence text for each source and makes
no blanket redistribution claim.

Two facts in the repository already bear on this decision:

1. **HX3 already ships ANSUR II-derived constants.**
   `src/shared/python/hand_size.py` holds eight summary ratios (mean hand
   length/stature and hand breadth/length by sex, plus a pooled p5–p95
   band) computed from the ANSUR II public CSVs. No per-subject rows are
   stored in the repository. #13760 closed before this ADR existed, so the
   sign-off that #13763 asked for has not been recorded.
2. **MyoSuite `myo_sim` assets are already vendored.**
   `shared/models/myosuite/myo_sim/` holds the MyoHand/MyoArm MJCF and
   roughly 200 STL meshes under the upstream Apache-2.0 `LICENSE`. The
   bilateral MyoHand golf fixture from HRG-GS-M8 (#13330) uses them.

## Evidence

Retrieved on 2026-11-03. "Verified" means read from the publisher's own
page. "Not verified" means the primary text could not be retrieved or read
in full.

| Source | Licence evidence | Status |
| --- | --- | --- |
| MANO (MPI-IS, `mano.is.tue.mpg.de/license.html`) | "Software Copyright License for non-commercial scientific research purposes". Single-user, non-exclusive, non-transferable. "shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part". Commercial use: ps-license@tue.mpg.de. | Verified (no version number in the text) |
| SMPL-X (MPI-IS, `smpl-x.is.tue.mpg.de/modellicense.html`) | Same non-commercial research licence family; redistribution prohibited "except that you may make one copy for archive purposes only". Commercial licensing through Meshcapade. Page footer © 2024 Max-Planck-Gesellschaft. | Verified (no version number in the text) |
| MyoSuite `myo_sim` wrapper | Apache License 2.0 (`shared/models/myosuite/myo_sim/LICENSE`, upstream `MyoHub/myo_sim` README). | Verified |
| MyoHand geometry lineage | The vendored `myo_sim/hand/README.md` says MyoHand derives from the OpenSim MoBL-ARMS model (`simtk.org/projects/upexdyn`) and the 2nd-Hand model (`simtk.org/projects/hand_muscle`). The SimTK project pages label their downloads "Creative Commons ANC Use Agreement" (MoBL-ARMS) and "MIT Use Agreement" (2nd-Hand). | Not verified: the agreement texts were not read in full, and neither source says which meshes carry which terms. "ANC" may mean attribution and non-commercial use. |
| ANSUR II (Gordon et al. 2014, NATICK/TR-15/007; data at `openlab.psu.edu/ansur2/`) | The PSU page says the data "were made available publicly in 2017" and gives no licence or distribution statement. The DTIC report pages (ADA611869) returned HTTP 403. A secondary search summary calls the report a US Government work in the public domain. | Not verified |
| Greiner 1991 (NATICK/TR-92/011, DTIC ADA244533) | Mirrored in the Internet Archive "US Government Mirrored Documents" collection. No distribution statement was readable from the item page. | Not verified |

## Decision (Proposed)

1. **MANO and SMPL-X: no meshes, parameters, or derived vertex data in the
   repository.** The existing SMPL-X mesh backend
   (`humanoid_character_builder/generators/_mesh_smplx.py`) loads models
   from a user-supplied `model_dir` and ships no model files. Keep that
   pattern: an optional, local-only loader where the user obtains the model
   under their own licence. Do not commit generated SMPL-X/MANO meshes,
   betas fitted to private captures, or golden files derived from them.
2. **ANSUR II summary constants: keep them pending sign-off.** The shipped
   values are aggregate statistics (means, standard deviations,
   percentiles) with source hashes and citations. Per-subject ANSUR II rows
   must not be committed. The owner should confirm the public-release
   status of NATICK/TR-15/007 and the PSU CSVs, then either accept this
   decision or tell HX3 to replace the constants. Until then the constants
   stay, and the open question is tracked on #13763.
3. **Greiner 1991: cite, do not ship.** No Greiner-derived constants are in
   the repository. Adding any requires the report's distribution statement
   to be recorded here first.
4. **MyoHand meshes: no new vendoring; the existing copy stays
   research-only.** The Apache-2.0 wrapper licence does not settle the
   rights to geometry derived from MoBL-ARMS or 2nd-Hand. Until the owner
   reviews the SimTK agreements:
   - do not copy further `myo_sim` meshes into the repository;
   - do not make MyoHand a default or a packaged asset (consistent with
     ADR-0057);
   - treat the existing `shared/models/myosuite/myo_sim/` copy and the M8
     fixture as research fixtures. Do not describe them as cleared for
     commercial redistribution.
   If the owner finds the MoBL-ARMS terms to be non-commercial, open a
   follow-up issue to decide whether that copy should move behind a
   user-supplied path like SMPL-X.

## Alternatives Considered

| Option | Decision |
| --- | --- |
| Vendor MANO/SMPL-X hand models for convenience | Reject: the verified licence text forbids redistribution. |
| Remove the ANSUR II constants now | Defer: they are aggregate statistics with provenance; removing them without a licence finding would churn HX3 for no benefit. |
| Treat the Apache-2.0 `myo_sim` licence as covering all its meshes | Reject: the upstream README names OpenSim sources whose own use agreements are separate and unreviewed. |

## Sign-Off Checklist (Owner)

- [ ] Confirm the ANSUR II public-release status and accept or reject
      decision 2.
- [ ] Review the MoBL-ARMS and 2nd-Hand SimTK use agreements and accept or
      amend decision 4.
- [ ] Accept decisions 1 and 3.

Until every box is checked, HX5 must not propose vendoring hand assets and
this ADR stays Proposed.

## Consequences

- Positive: each source now has recorded licence evidence, and the
  unverified ones are named.
- Negative: MyoHand-based work stays research-only, and HX3's shipped
  constants depend on a sign-off that is still open.
- Follow-up: after sign-off, set Status to Accepted and update ADR-0057's
  asset provenance gate.

## Validation

This ADR is documentation only. It makes no code change, and none of it is
legal advice. The repository facts were checked against
`src/shared/python/hand_size.py`,
`shared/models/myosuite/myo_sim/{LICENSE,hand/README.md}` and
`_mesh_smplx.py` at the commit this ADR lands on.
