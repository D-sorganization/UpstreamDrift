# ADR-0061: HRG-HX6 Hand Shape and Size Asset Licensing

- Status: Proposed (human sign-off required before acceptance)
- Date: 2026-11-07
- Decision Makers: RD Barb (draft); Dieter (sign-off owner)
- Related Issues/PRs: #13763, #13757, #13760 (PR #13840), #13762
  ([ADR-0057](0057-hrg-hx5-articulated-hand-option.md)), #13330

## Context

The HRG-HX epic (#13757) needs hand shape and size data. Four candidate
sources were raised by the hands panel: the ANSUR II survey, Greiner 1991,
the MPI-IS MANO and SMPL-X models, and the MyoSuite myoHand asset. The panel
input in #13763 was explicitly not legally verified. This ADR records the
licence text actually retrieved on 2026-11-07, marks what could not be
retrieved, and proposes a per-source decision. It is an engineering record,
not legal advice; acceptance requires the owner's sign-off.

Repository state at drafting time (`main` at `b0e3a8e5ba`):

- HX3 (PR #13840, merge `eed5077d7b16`) already ships ANSUR II-derived
  summary ratios in `src/shared/python/hand_size.py` (means, SDs and p5/p95
  of `handlength/stature` and `handbreadth/handlength`). No raw ANSUR rows
  are committed.
- No Greiner 1991 values are used anywhere under `src/`.
- No MANO or SMPL-X files, parameters or loaders are tracked.
- The HRG-GS-M8 fixture (#13330) commits generated MuJoCo XML under
  `shared/models/myosuite/golf/hand/` (`myohand_left_chain.xml`,
  `myohand_right_chain.xml`, two bilateral scenes and a receipt pinning
  `myo_sim` at `33f3ded946f5`). The chains carry body poses, joints and
  inertias derived from `myo_sim`; they reference meshes by name and do not
  commit mesh files.

## Licence Evidence

| Source | Licence as retrieved | Retrieved from | Verified |
| --- | --- | --- | --- |
| MANO | "Software Copyright License for non-commercial scientific research purposes" (© 2020). Grant: "for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects". Restriction: "shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part except that you may make one copy for archive purposes only." Commercial: ps-license@tue.mpg.de | `https://mano.is.tue.mpg.de/license.html` | Yes |
| SMPL-X | Same licence family, © 2024; same non-commercial grant and the same no-redistribution clause, applied to "the Model & Software". Commercial: smpl@max-planck-innovation.de or Meshcapade | `https://smpl-x.is.tue.mpg.de/modellicense.html` | Yes |
| myo_sim (myoHand wrapper) | Apache License, Version 2.0 (`LICENSE` at `myo_sim` commit `e18294651445`) | `github.com/MyoHub/myo_sim` | Yes |
| myoHand geometry lineage | `myo_sim/models/arm/README_hand.md`: "derived from the MoBL upper-extremity OpenSim model and the 2nd-Hand model" | same repository | Yes |
| MoBL-ARMS upper-extremity model | SimTK project licence field: "Creative Commons ANC Use Agreement" (attribution, non-commercial) | `https://simtk.org/projects/upexdyn/` | Licence name only; full agreement text not retrieved |
| 2nd-Hand model | SimTK project licence field: "2nd_Hand_Model MIT Use Agreement" | `https://simtk.org/projects/hand_muscle` | Licence name only; full agreement text not retrieved |
| ANSUR II | Source page requests the citation Gordon et al. 2015, NATICK/TR-15/007, and states no other terms. Believed to be a US Government work with Distribution Statement A | `https://www.openlab.psu.edu/ansur2/` | Citation yes; distribution statement **not verified** (DTIC returned HTTP 403) |
| Greiner 1991 | Believed to be a US Army Natick hand-anthropometry technical report (DTIC ADA244533); title, number and distribution statement unconfirmed | DTIC citation page | **Not verified** (HTTP 403) |

No row above makes a blanket redistribution claim. Rows marked "not
verified" must stay that way until the text is read from the primary source.

## Decision (Proposed)

1. **MANO and SMPL-X: never in the repository.** No meshes, shape or pose
   parameters, or derived blend-shape tables may be committed, vendored,
   generated into fixtures or bundled into any release. The verified
   no-redistribution clause rules this out regardless of research use. A
   future optional loader may read a user-supplied, locally licensed file
   from a path the user provides, must not cache derived data inside the
   repository tree, and must be off by default.
2. **ANSUR II summary constants: may ship, with citation.** Statistical
   summaries (ratios, means, SDs, percentiles) computed from the public CSVs
   may stay in `hand_size.py` with the Gordon et al. 2015 citation and the
   source hashes it already records. Raw ANSUR rows must not be committed.
   This item depends on the owner confirming the public-release status,
   because the DTIC distribution statement could not be retrieved.
3. **Greiner 1991: not cleared for use.** No values may be added until the
   report's distribution statement is read from the primary source and
   recorded here.
4. **myoHand assets: research fixture only; no vendoring.** The Apache-2.0
   wrapper licence does not cover the upstream geometry. MoBL-ARMS carries a
   non-commercial Creative Commons agreement, so myoHand-derived body data is
   treated as non-commercial. No myoHand mesh may be vendored. The committed
   M8 generated chains stay labelled as a non-commercial research contact
   fixture and must be excluded from any commercial distribution. HX5's
   MyoHand option (ADR-0057) stays excluded as a shipping default.
5. **Gate.** ADR-0057's asset-provenance gate stays in force. Any future
   third-party hand asset needs a row in the table above, with the licence
   text read from its primary source, before it lands.

## Alternatives Considered

1. **Ship MANO/SMPL-X parameters for research builds only.** Rejected: the
   verified licence forbids any sharing or distribution, and a public
   repository is distribution.
2. **Remove the M8 generated chains now.** Deferred to the owner: removal
   changes tests and research evidence (#13330), and the non-commercial
   status may be acceptable while UpstreamDrift is not sold. Recorded as a
   follow-up decision, not made here.
3. **Replace ANSUR II with de Leva (1996) only.** Rejected: de Leva gives
   hand length but no breadth; HX3 already uses de Leva as a length-only
   cross-check.

## Consequences

- Positive: each hand data source has a recorded licence basis; the
  MANO/SMPL-X prohibition and the myoHand non-commercial lineage are
  explicit before any integration depends on them.
- Negative: the myoHand fixture's non-commercial lineage restricts any
  future commercial packaging that includes it; Greiner 1991 stays unusable
  until its status is read.
- Follow-ups: owner sign-off on decisions 1–5; owner decision on the M8
  chains for commercial packaging; retrieve the DTIC distribution statements
  for NATICK/TR-15/007 and ADA244533 and update this table; read the full
  MoBL-ARMS and 2nd-Hand agreement texts.

## Validation

Documentation only; no runtime behaviour changes. Repository state was
checked with `git ls-files` (no MANO/SMPL-X files; myoHand generated XML
without meshes) and a search of `src/` for Greiner values (none). Licence
rows were retrieved on 2026-11-07 from the URLs listed. Acceptance needs the
owner's sign-off on #13763, not a passing CI run.
