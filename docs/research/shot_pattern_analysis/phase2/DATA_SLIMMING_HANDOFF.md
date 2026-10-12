# Shot Data Slimming Handoff

## Scope at Turnover

The user believes the incoming agent intends to slim the shot data before
committing a smaller representation to Git. This is a proposed successor task,
not an implemented or agreed compression/schema design. This session stopped
without converting data, deleting rows, staging bulk data or rewriting history.
All10.44million attempted simulation rows are complete; no new simulation is needed.

## Preserve the Original Sources

Original study root:
`/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2`.
Approximate local sizes measured at stop (filesystem usage, not upload budgets):

| Path Under the Study Root | Size | Role |
| --- | ---: | --- |
| `final_v1` | 1.1 GiB | Original Primary Final Rows and Receipts |
| `control_final_v1` | 194 MiB | Control Final Rows and Receipts |
| `supplement_final_v1` | 98 MiB | Supplemental Final Rows and Receipts |
| `control_pilot_v1` | 13 MiB | Required Control Provenance |
| `pilot_v1` | 87 MiB | Historical Pilot Evidence |
| `data_archives` | 1.5 GiB | Manifested Original Data Archives |
| `supplement_report_v1` | 2.6 GiB | Original Report Assets |
| `supplement_report_v2_layout_repair` | 2.6 GiB | Accepted Selective Layout Child |
| `validation` | 1.4 GiB | Execution, Review and Failed-Attempt Provenance |

Also retain `/tmp/shot-pattern-delivery-resume` (about15GiB): release staging,
verified downloaded/restored copies, illustrated source and earlier delivery
receipts. Similar size or filenames do not establish safe duplication. Do not
delete this tree or ignored/untracked study files without a separate exact
inventory comparison and preserved recovery source.

The existing data archives are **already tracked on this integration branch**
(`git ls-files docs/research/shot_pattern_analysis/phase2/data_archives` lists
ZIPs and manifests). Some large content therefore already exists in branch
history; the successor must inspect history and remote reachability before
assuming slimming the working tree alone reduces the pending Git transfer.
No history rewrite is authorized by this turnover.

## Existing Reusable Packaging and Restoration

Read [Data Archive Guide](data_archives/README.md) and
[Reproduction](REPRODUCING.md). The existing
`scripts/research/shot_pattern_analysis/archive_data.py` provides deterministic,
size-bounded ZIP packaging and `verify_archives`; it does not slim the dataset.
Its default80,000,000-byte parts are distinct from the historical500,000,000-byte
figure-release parts. ZIP_STORED preserves already compressed member bytes;
repackaging alone should not be described as row/column reduction.

Archive bundle names map explicitly to `final_v1`, `control_final_v1`,
`supplement_final_v1` and the required `control_pilot_v1` prerequisite. Preserve
manifest identity before extraction and verify every restored path, byte count
and hash. The original research release is already published; do not overwrite
its assets or silently change an existing manifest to describe a compact child.

## Successor Design Decisions and Acceptance

Clarify which deliverable is intended: a lossless alternative encoding, a
reproducible row/column subset, or aggregate statistics with an externally
restorable original. No selection, format, precision reduction, Git LFS policy
or size target was agreed here. Measure both per-file limits and incremental Git
object/push size before committing; compression does not remove existing Git
history. Do not amend old scientific receipts or rewrite shared history.

Create any compact version in a new directory. Record source archive/manifests,
input hashes, schema, units, dtypes, selection rules and stable shot identifiers,
row counts including unsupported outcomes, software/runtime and exact conversion
command. Preserve primary/control/supplement distinctions, pairing and support
population definitions. Avoid decimal rounding or dropped outcomes that would
change exact equality, scoring or the accepted result populations.

Demonstrate exact round-trip row/value equivalence for a lossless conversion.
For an intentionally reduced scientific derivative, explicitly disclose what is
omitted and prove that its claimed supported analyses reproduce the frozen
accepted statistics and report populations; it must not impersonate the full
original. Keep a manifest of the compact output and a tested restoration route
to the original. Reuse existing readers/archives where applicable and add focused
negative controls for missing/changed provenance before changing consumers.

This handoff records proposed acceptance boundaries only. It does not authorize
this stopped session to implement slimming or claim a smaller dataset is ready.
