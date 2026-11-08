# ADR-0061: HRG-HX6 Licensing of Hand Shape and Size Sources

- Status: Proposed (owner sign-off required before HX3 or HX5 relies on it)
- Date: 2026-11-07
- Decision Makers: RD Barb (draft); repository owner (sign-off)
- Related Issues/PRs: #13763, #13757, #13760 (HX3), #13762 (HX5), ADR-0057

## Context

The HRG-HX epic (#13757) needs hand breadth, length and shape data for the
golfer hand proxy. Four sources were proposed: MANO, SMPL-X, the MyoSuite
`myo_sim` hand, and the ANSUR II and Greiner (1991) anthropometric surveys.
HX3 cannot ship hand-size constants and HX5 (ADR-0057) cannot propose a
third-party mesh until their licensing is recorded.

This ADR records licence texts as retrieved on 2026-11-07. It is an
engineering record, not legal advice. Every statement marked **unverified**
needs the owner's confirmation before anything depends on it.

## Licence Record

| Source | Licence text (retrieved 2026-11-07) | Grant | Redistribution |
| --- | --- | --- | --- |
| MANO (MPI-IS) | "Software Copyright License for non-commercial scientific research purposes", © 2020 Max-Planck-Gesellschaft, <https://mano.is.tue.mpg.de/license.html> | "for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects" | "shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part except that you may make one copy for archive purposes only" |
| SMPL-X (MPI-IS) | "Software Copyright License for non-commercial scientific research purposes", © 2024 Max-Planck-Gesellschaft, no version number on the page, <https://smpl-x.is.tue.mpg.de/modellicense.html> | Same non-commercial grant, applied to the "Model & Software" | Same no-redistribution clause; commercial use through Max-Planck-Innovation |
| MyoSuite `myo_sim` | Apache License, Version 2.0, January 2004 (`LICENSE` at the `main` branch of <https://github.com/MyoHub/myo_sim>) | Apache-2.0 for the repository content | Apache-2.0 terms apply to the wrapper. The README does not cite where the hand meshes come from, so their lineage is **unverified** |
| ANSUR II | No licence or terms-of-use text on the distribution page (<https://www.openlab.psu.edu/ansur2/>). The page says the data "were published internally in 2012. They were made available publicly in 2017." Files: `ANSUR_II_MALE_Public.csv`, `ANSUR_II_FEMALE_Public.csv` | Public release; the exact distribution statement is **unverified** | Cite Gordon et al. (2015), *2012 Anthropometric Survey of U.S. Army Personnel: Methods and Summary Statistics*. That report's DTIC distribution statement could not be retrieved and is **unverified** |
| Greiner (1991) | US Army Natick technical report on hand anthropometry. The text and distribution statement have not been retrieved | **Unverified** | **Unverified** |

Both MPI-IS licences also exclude commercial, pornographic, military and
surveillance uses, including training algorithms for those uses.

## Current Repository State

No MANO or SMPL-X model files (`.pkl`, `.npz`) are in `src/`. The existing
SMPL-X generators
(`src/shared/python/humanoid_character_builder/generators/mesh_generator_smplx.py`
and `_mesh_smplx.py`) load models only from a user-supplied local directory
and report themselves unavailable when that directory is missing. The MyoHand
fixture from HRG-GS-M8 (#13330) is research evidence only (ADR-0057).

## Decision

1. **MANO and SMPL-X.** No meshes, shape parameters, joint regressors, or
   values fitted from them may be committed or redistributed. The only allowed
   integration is an optional local loader that reads a user-supplied path,
   skips cleanly when the path is missing, and states the MPI-IS licence in
   its docstring. The existing SMPL-X generators already meet this rule.
   Default and CI paths must not depend on these models.
2. **ANSUR II summary constants.** Pending owner sign-off, HX3 may ship
   *summary statistics only* (for example percentiles of hand length and
   breadth), each with the source file, variable name, sample, unit
   conversion and the Gordon et al. (2015) citation. Do not vendor the raw
   CSVs or individual-level records. Until sign-off, keep any such constants
   behind a test fixture, not in shipping defaults.
3. **Greiner (1991).** Do not use until its text and distribution statement
   are retrieved and recorded here.
4. **MyoHand meshes.** Do not vendor or ship. Apache-2.0 covers the wrapper;
   it does not establish rights to meshes whose source is not cited. ADR-0057
   remains in force: MyoHand stays excluded until the mesh lineage is traced
   to a source whose licence permits redistribution.
5. **Recording rule.** Any later asset or data source used for hand shape
   must add a row to the licence record above with the quoted licence text,
   the retrieval date, the upstream revision and file hashes. A blanket
   statement such as "public domain" or "open source" is not enough.

## Alternatives Considered

| Option | Decision |
| --- | --- |
| Commercial Meshcapade or Max-Planck licence for MANO or SMPL-X | Not selected. It needs an owner purchasing decision and would still forbid redistribution to users |
| Vendor MyoHand meshes under Apache-2.0 | Rejected while mesh lineage is unverified |
| Ship raw ANSUR II CSVs | Rejected. Summary constants meet HX3's need and limit exposure if the distribution statement turns out to be narrower |
| Parametric primitives scaled from anthropometric summaries (ADR-0057 option) | Preferred hand geometry path. It needs no third-party mesh |

## Consequences

- Positive: HX3 and HX5 have a recorded, testable boundary. Nothing in the
  repository today has to change.
- Negative: the hand cannot use SMPL-X or MANO shape fidelity in default or
  CI paths.
- Follow-up: the owner signs off on decisions 2 and 4. Someone retrieves the
  Gordon et al. (2015) and Greiner (1991) distribution statements and records
  them here before decision 2 becomes Accepted.

## Validation

Documentation only. The licence texts were retrieved from the URLs above on
2026-11-07. The repository state was checked by searching `src/` for model
files and reading the SMPL-X generator loaders. This ADR makes no legal
clearance claim and does not authorize shipping any asset.
