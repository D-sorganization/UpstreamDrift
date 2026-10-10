# Reproducible Study Data Archives

Each dataset directory contains deterministic ZIP parts, each smaller than 80 MB,
and `archive_manifest.json`. The manifest records SHA-256 and byte counts for every
archive and every original file. Archives preserve the complete source directory
name and all relative paths, including raw gzip CSV rows, plans, summaries,
scoring baselines, execution receipts, and figures. Packaging is lossless; it does
not substitute for scientific acceptance.

Completed pilot datasets are `pilot_v1`, `control_pilot_v1`, and
`supplement_pilot_v1`. Completed final datasets are `final_v1`,
`control_final_v1`, and `supplement_final_v1`; their saved-data audits and
fresh-restoration evidence are linked from the [Current Turnover](../TURNOVER.md).
Pilot results must not be presented as final 10,000-shot results.

## Verification and Restoration

From the repository root, verify the bundle before restoring it:

```python
from pathlib import Path
from zipfile import ZipFile
from scripts.research.shot_pattern_analysis.archive_data import verify_archives
import json

phase = Path("docs/research/shot_pattern_analysis/phase2")
bundles = {
    "final_primary_v1": "final_v1",
    "final_control_v1": "control_final_v1",
    "supplement_final_v1": "supplement_final_v1",
    "control_pilot_v1": "control_pilot_v1",
}
for bundle_name, dataset in bundles.items():
    bundle = phase / "data_archives" / bundle_name
    verify_archives(bundle)
    manifest = json.loads((bundle / "archive_manifest.json").read_text())
    assert manifest["source_name"] == dataset
    # Use a fresh destination. Do not overwrite an existing study run.
    assert not (phase / dataset).exists()
    for archive in manifest["archives"]:
        with ZipFile(bundle / archive) as source:
            source.extractall(phase)
```

Restoration returns the paths used by the immutable final-run acceptance receipt.
Run the independent matrix audit after restoration; archive integrity alone does
not verify scientific calculations. Existing local unpacked copies are retained
and ignored by Git to avoid duplicating the same data in two representations.

The control pilot is a provenance prerequisite even when rebuilding only final
results. `load_reviewed_common_targets` verifies its `matrix_plan.json` against
the pilot audit bound by `FINAL_RUN_ACCEPTANCE.md`. Keep the tracked
`common_targets.json`, `FINAL_RUN_ACCEPTANCE.md`, and
`validation/independent_control_pilot_audit.json` alongside the restored datasets.
Do not substitute a newly generated pilot plan or infer targets from final shots.

## Rebuilding Archives

```bash
python -m scripts.research.shot_pattern_analysis.archive_data \
  docs/research/shot_pattern_analysis/phase2/pilot_v1 \
  /tmp/pilot-v1-rebuilt-archives
```

The publisher rejects existing destinations, nested output directories, hidden
pending work, symbolic links, and files larger than one archive budget. It verifies
every stored payload before atomically publishing the bundle. Twelve focused tests
cover reproducibility, round-trip bytes, corruption, inventory completeness, and
input rejection, publication-time source changes, and unsafe ZIP member metadata. ZIP storage deliberately preserves existing compressed bytes.

## Verified Pilot Restoration

The pilot archives were restored into a fresh temporary directory, with every
file checked against the stored byte count and SHA-256. Loading each restored
matrix produced exactly the same source-bound report as its original local
copy. The original bundles were verified again after the check. See the
[Restoration Receipt](../validation/pilot_archive_restoration_review.json).
This is a packaging and portability check; it does not replace scientific
validation or establish acceptance of final datasets. Separate final audit/restoration receipts provide that evidence.

To repeat the check from the repository root, supply a new receipt path:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python \
  docs/research/shot_pattern_analysis/phase2/validation/restore_pilot_archives.py \
  /tmp/pilot-archive-restoration-review.json
```

The script compares the restored reports with the existing local unpacked
pilots, uses a temporary destination, and refuses to replace an existing output
receipt. Its source hash and the exact archive hashes are included in the receipt.

## Rebuilding the Publication

After restoring the final data, follow the [Publication Reproduction Guide](../REPRODUCING.md).
Keep the accepted simulation snapshots separate from the reviewed reporting revision.
