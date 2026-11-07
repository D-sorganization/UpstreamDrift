# ADR-0061: HRG-HX6 Hand Shape and Size Data Licensing

- Status: Proposed (human sign-off required; see "Sign-Off Gate")
- Date: 2026-11-07
- Decision Makers: RD Barb (draft); repository owner (sign-off)
- Related Issues/PRs: #13763, #13757, #13760 (PR #13840), #13762 (ADR-0057),
  #13330

## Context

The HRG-HX epic (#13757) needs hand length, hand breadth and, later, hand
shape. Four sources were raised: the ANSUR II anthropometric survey, Greiner
1991, the MPI-IS MANO and SMPL-X body models, and the MyoSuite MyoHand model.
UpstreamDrift is MIT-licensed, so anything committed here may be
redistributed, including commercially. This ADR records what each source's
licence text says, what the repository holds today, and the decision.

This is an engineering record, not legal advice. Every quotation below was
read from the primary source on 2026-11-07. Where no licence text was found,
the row says so and makes no claim.

## Source Licences (Read 2026-11-07)

| Source | Primary text | What it says | Verified |
| --- | --- | --- | --- |
| ANSUR II data (public CSVs) | [PSU OPEN Design Lab](https://www.openlab.psu.edu/ansur2/) | "published internally in 2012. They were made available publicly in 2017." No licence or terms-of-use text on the page. | Public availability: yes. Licence text: none found |
| ANSUR II report, Gordon et al., NATICK/TR-15/007 | [DTIC ADA611869](https://apps.dtic.mil/sti/pdfs/ADA611869.pdf) | Report cover: "Approved for public release; distribution is unlimited". | Yes, for the report. Not a data licence |
| Greiner 1991 (US Army technical report) | Not retrieved | Not checked. No code in `src/` cites it. | No |
| MANO | [mano.is.tue.mpg.de/license.html](https://mano.is.tue.mpg.de/license.html) | "Software Copyright License for non-commercial scientific research purposes". Use limited to "non-commercial scientific research, non-commercial education, or non-commercial artistic projects". "may not be reproduced, modified and/or made available in any form to any third party without Max-Planck's prior written permission". Commercial: ps-license@tue.mpg.de. | Yes |
| SMPL-X | [smpl-x.is.tue.mpg.de/modellicense.html](https://smpl-x.is.tue.mpg.de/modellicense.html) | Same licence title and use limits. "shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part except that you may make one copy for archive purposes only." Commercial: smpl@max-planck-innovation.de or Meshcapade. | Yes |
| MyoSuite `myo_sim` at `33f3ded946f55adbdcf963c99999587aadaf975f` | [`LICENSE`](https://github.com/MyoHub/myo_sim/blob/33f3ded946f55adbdcf963c99999587aadaf975f/LICENSE) | Apache License, Version 2.0. The repository page names no origin or third-party licence for the meshes in `meshes/`. | Wrapper licence: yes. Mesh lineage: no |

## Repository State (2026-11-07)

- **ANSUR II.** `src/shared/python/hand_size.py` (HX3, PR #13840) ships
  derived summary ratios (sex-specific means, SDs and p5–p95 bounds of
  `handlength/stature` and `handbreadth/handlength`) plus the source file
  hashes. No raw ANSUR rows are committed. `foot.py` and `pelvis_width.py`
  use the same pattern. HX3 merged before this sign-off, which #13763 said it
  should not.
- **Greiner 1991.** Not used. HX3 took breadth from ANSUR II instead.
- **MANO and SMPL-X.** No model files are tracked. The SMPL-X generator
  (`humanoid_character_builder/generators/_mesh_smplx.py`) loads from a
  caller-supplied `model_dir` through the optional `smplx` package. The HMR 2.0
  sidecar (`src/tools/hmr2_sidecar/`) runs a user-installed tool as a
  subprocess and loads no SMPL files.
- **MyoHand.** `myo_sim` is a git submodule (`shared/models/myosuite/myo_sim`)
  and is not vendored. The 12 tracked files under `shared/models/myosuite/golf/`
  are generated MJCF scenes that reference the submodule's meshes. They
  contain no mesh files (ADR-0057, #13330).

## Decision

1. **MANO and SMPL-X: never in the repository.** No meshes, parameters,
   blend shapes, regressors or derived vertex data. Integration stays
   local-only behind a user-supplied path or an external subprocess, as it is
   today. Tests use synthetic fixtures. Re-sharing derived outputs (for
   example fitted betas) also needs owner review, because the licence covers
   modified forms.
2. **ANSUR II derived summary constants may ship (subject to sign-off).**
   Allowed: aggregate statistics computed from the public 2017 release, with
   the citation (Gordon et al., NATICK/TR-15/007) and the sha256 of each
   source file. Not allowed: committing the raw per-subject CSVs. The release
   is a US Government survey that is publicly available, and the summary
   report is approved for public release. However, no licence text was found
   for the data files, so this ADR does not claim they are public domain.
3. **Greiner 1991: not a source** until its distribution statement is read
   and recorded here.
4. **MyoHand meshes: not vendored.** Keep `myo_sim` as a pinned submodule.
   Generated scenes may reference its meshes but must not copy them. Vendoring
   any mesh, or making MyoHand the default hand (ADR-0057), needs the
   per-asset lineage recorded first. That means the upstream source model and
   its licence, for each mesh file used. Apache-2.0 on the wrapper does not
   settle this.

## Alternatives Considered

| Option | Decision |
| --- | --- |
| Ship SMPL-X/MANO hand meshes for shape realism | Reject: the licence forbids redistribution and commercial use |
| Commit raw ANSUR II CSVs for reproducibility | Reject: no licence text found. Hashes plus a download location reproduce the constants without redistributing the data |
| Remove ANSUR-derived constants until a licence text is found | Not recommended: they are aggregates of a public US Government release with a public-release report. The owner may still choose this at sign-off |
| Vendor MyoHand meshes under Apache-2.0 | Defer: the mesh lineage is not established |

## Sign-Off Gate

#13763 requires human sign-off. The repository owner should record one of
these on #13763:

- **Approve:** Status becomes Accepted, which retroactively clears the HX3
  constants (PR #13840).
- **Amend:** state the changes. If ANSUR constants must go, open a removal
  issue against `hand_size.py`, `foot.py` and `pelvis_width.py`.

Until then, no new licence-dependent integration may land. That covers new
ANSUR-derived constants, any MyoHand mesh vendoring, and any MPI-IS asset use
beyond the existing local-only loaders.

## Consequences

- Positive: one record of what may ship. HX5 (ADR-0057) and any later hand
  shape work have a clear rule.
- Negative: MANO/SMPL-X hand shape realism stays out of reach for shipped
  models. MyoHand stays research-only until its mesh lineage is audited.
- Follow-up: audit the `myo_sim` mesh lineage, and read the Greiner 1991
  distribution statement if anyone proposes it as a source.
