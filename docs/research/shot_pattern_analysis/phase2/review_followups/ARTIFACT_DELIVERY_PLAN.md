# Shot Pattern Phase-Two Artifact Delivery Plan

## Current Delivery and Stop Status

The user stopped goal pursuit on October12,2026 for another agent to take over.
Read [Current Successor Handoff](../RESUME_HANDOFF.md) and
[Data Slimming Handoff](../DATA_SLIMMING_HANDOFF.md). No data slimming or new
artifact publication was performed during turnover.

The live article/PDF is accepted at deployed9002c8a2f940:53/53 revision-bound
checks, all nine assets, four browser/visual cells and the exact39-page PDF hash
`fd5506da87f5e619a550fb8cc871a0e8195b6fb98965a99809e3238a0f01508b`.
The historical33-page illustrated PDF in the original release is a prior edition.
The [research prerelease](https://github.com/D-sorganization/UpstreamDrift/releases/tag/shot-pattern-phase2-research-2026-10-10)
contains25files:21original payloads, two earlier receipts and two additional
live-verification archives. Original payloads and all98links were remotely
restored and verified. Do not recreate or overwrite the release.

Integration PR #12278 remains unmerged with one protected full-unit alias failure;
local canonical all eight gates passed before that full-suite failure. AffineDrift
#5022 remains unmerged at stop with auto-merge armed and browser CI running.
These states are separate from completed live publication.

## Historical Packaging Recipes and Evidence

The remainder preserves the exact earlier packaging inventories and execution
recipes. Statements such as33-page current report, unopened PR, failed older
gates, former DELIVERY checkout or pending publication describe those historical
checkpoints, not current instructions. Do not execute these examples as a
restart plan: the old main-integration worktree was safely retired, the release
already exists, and the current integration checkout is the safe-followup tree.
Use the current handoff and existing accepted manifests for successor work.



This earlier packaging plan binds the phase-two delivery data to the accepted repaired child v2 graphics. Its text-only REPORT.pdf was not an acceptable primary visual deliverable at that checkpoint. The illustrated article has since passed the 33-page, 13-figure document review and is the primary report in the v3 staging inventory above. Preserve the accepted dataset and reviewed figures; this document records no simulation or graphics regeneration.

## Current Acceptance Evidence

The v6 strict public-loader receipt is `phase2/validation/supplement_v6_final_graphics_review_v2/strict_loader_receipt.json` (SHA256 `0153d81e1a7c7c91c66c6f0f378081ca8474efdf8a5d2b4a94f15fd16e898bf4`). Its execution receipt is in the same directory. It binds 384 ordered comparisons, 26 unique cells, and 6,456 assets. This confirms source/asset reconciliation, not visual acceptance.

The selective density layout repair replaces 49 PDF/SVG/PNG triples (147 assets). Its scope receipt is `phase2/validation/supplement_v6_final_graphics_review_v2/density_layout_repair_scope.json` (SHA256 `7599cc36a0008857c5579502512510f8521f8452f53b88b771e3d5db3462caba`); publication receipt `layout_repair_child_publication_receipt.json` (SHA256 `cb9501c25bb8dfbbc1741705a25181a024fb9ebc9f8c6dfdb08e31948d375c66`); reconciliation receipt `layout_child_reconciliation_receipt.json` (SHA256 `4362a99963fbf8c35b5fd3988c8215856b9b9c7976b1376c3b3386b8f80a8918`); and 384-SVG clearance receipt `layout_child_clearance_receipt.json` (SHA256 `decbca752833ccd2a2ac959ecbfdfa9f964301fc66812248d4382c0357177052`). The repaired child is `phase2/supplement_report_v2_layout_repair/`.

All 49 repaired PNGs passed scoped reviews by Root, Sol, and Astra. Luna rendered the 49 replacement PDFs and SVGs with the qualified helper into native raster comparisons and reviewed all seven contact sheets at contact scale. Durable Luna receipt: `phase2/validation/supplement_v6_final_graphics_review_v2/layout_child_native_vector_review.json` (SHA256 `60b6b1506abde49dad348ce6c7a502f2032c61a4d11b227e88c55edcf0621a1c`). This records no contact-scale mismatch or clipping; it is not a full-size visual review of every replacement or scientific validation. The existing 4,048-vector structural review and 172 predeclared representatives remain separately bound by the v2 validation receipts.

The prior 1,122 PDF text-box overflow flags were resolved against painted-glyph evidence. Receipt: `phase2/validation/supplement_v6_vector_review_v2/bbox_painted_ink_boundary_review.json` (SHA256 `d44b5982b3513bb6b93d52927328cdbefa555994ffb277884bcfac38611f7e60`). Across 10,842 word-edge probes, all flagged words had painted on-page ink with at least 6.4952 pt edge clearance, and no raster had ink in its outer 2-pt band. Six representative pages were visually inspected. The worker rendered all pages and flushed per-file results, then exited 1 only while serializing its aggregate summary due mixed float/string JSON keys; a separate finalizer completed the receipt with exit 0 from saved results, without rerendering. The detailed gzip record is `bbox_painted_ink_per_pdf.jsonl.gz` (SHA256 `0cc3e1581a32e6711260ec6586ee94c5fe5eeb548570caf4678f7859b22edc56`).

These receipts are present in the DELIVERY checkout. The underlying graphic assets have scoped acceptance receipts, and the illustrated report has separately passed document compilation and 33-page visual review; the accepted PDF and source archive are bound by `phase2/validation/illustrated_article_v1/review.json` and the v3 staging receipts above. The recorded `scripts.pre_pr` run failed; Sol’s Expat issue has since received a focused fix, but that does not clear the repository-wide test and Fleet Policy failures. A corrected full pre-PR attempt later terminated with exit 143 before a final summary; it is incomplete and does not supersede the failed complete run. Do not describe the repository-wide gate as passing. The published research prerelease is bound to the exact DELIVERY SHA above, while the PR remains explicitly blocked from merge until the gates are resolved. This does not claim production readiness or merge readiness, and it does not bypass hooks or protection.

## Archive API and Fresh Restore Semantics

The source-controlled API is `scripts/research/shot_pattern_analysis/archive_data.py`. `build_archives(source, output, max_bytes=...)` writes a deterministic ZIP set and `archive_manifest.json`, then verifies before atomic publication. The manifest uses an `archives` object keyed by ZIP basename and a `files` object keyed by complete archive member path. The public `verify_archives(output)` verifies archive hash/size, member safety, exact inventory and member hashes/sizes; it returns the file count. There is no public restore function.

For a fresh restore, first compare the downloaded manifest byte-for-byte with the accepted local manifest. Then map the prefixed remote filenames back to the exact basenames recorded in that manifest, call `verify_archives`, extract each ZIP, compare the complete extracted file inventory to manifest `files`, and verify every extracted byte count and SHA256. This proves remote byte restoration, not visual or scientific acceptance.

The accepted local primary/control archives were checked once against their receipts and copied byte-identically to `/tmp/shot-pattern-delivery-resume/release-assets-primary-control-prepared/`. The frozen final `REPORT.md` and PDF retain links into `phase2/supplement_report_v1/`. Keep the original baseline study checkout intact. For the publication/consumer tree, restore the accepted child archive to a fresh temporary `phase2` tree, verify its archive and full extracted inventory first, then route the extracted `supplement_report_v2_layout_repair/` directory to the canonical `supplement_report_v1/` path expected by those unchanged report links. This is a delivery-path mapping only; do not edit the report bytes or replace the baseline in the study checkout. Record both the pre-routing extracted child hashes and the final routed path in the restoration receipt.

| Bundle | ZIP asset name | Bytes | ZIP SHA256 | Manifest SHA256 |
| --- | --- | ---: | --- | --- |
| Primary | `primary_final_figures_v1_001.zip` | 223,739,795 | `6ba794355bf4575e2a6fb24a3dc7473dbd0279a628c3e99bbe191ce2a4785cc2` | `3375d756c1aa39fb985ebccfb46600de8e1e07714f3dee5561ec7e5ae6229d44` |
| Control | `control_final_figures_v1_001.zip` | 168,462,247 | `0101c2cbc5cba3ffb0b6bded0979cb63fb28adb52f446fbda1a4384c226e35c0` | `492962d0dc0a2f43ea95b143d352b4234331a8a2c42d10b33c1f7a55c9dbae9c` |

Their accepted restoration receipts are `publication_primary_bundles_v1/restoration.json` and `publication_control_bundles_v1/restoration.json`. Those historical local restorations do not establish a fresh remote restore.

The supplement v2 bundle was built successfully with the DELIVERY archive API into `/tmp/shot-pattern-delivery-resume/supplement-v2/`: 6,465 files in six ZIPs. All six parts are below the 500,000,000-byte limit. The manifest SHA256 is `5d776ea73d9564812431416a01c2661a3f9335de1554ef111d24673b23125f74`. Part hashes/sizes are captured in `/tmp/shot-pattern-delivery-resume/local_fresh_restoration_receipt_v2.json` (receipt SHA256 `a038f900e8208342651f1f06489c4903f9e6b6ab7daac51597a16c8f0e5ea985`); bundle creation and API verification succeeded. The receipt also records an offline fresh extraction of all three local bundles: primary 690/690 files, control 548/548, supplement 6,465/6,465, every member size/hash exact. This is a local restoration rehearsal, not a remote release download. The full restored trees are retained under `/tmp/shot-pattern-delivery-resume/local-fresh-restoration-v2/`. At the earlier 19-asset checkpoint, `report__REPORT.pdf` had accepted source-byte provenance but lacked embedded research graphics. It remains a labelled historical text-only asset; `ILLUSTRATED_REPORT.pdf` is the accepted primary report in the 21-asset v3 inventory. Unique actual upload basenames have also been staged byte-identically in `/tmp/shot-pattern-delivery-resume/release-assets-v2/`, including direct `report__REPORT.md` (SHA256 `ff281f76ee649c44c88afda58701329e7898e8964f9efb6d8aeec73dcd77d79b`), `report__REPORT.pdf` (SHA256 `70a56243ff17ac833a60037562a07777b2bdac4aa4267a9956856a3492763b16`), and six accepted social PNGs. Those PNGs cover Driver, Seven Iron, and Pitching Wedge with density and overhead images at lean0, curve1, rotation2, combined2 (shaft-twist SD 2°, not delivered-face SD 2°); each hash matches its accepted primary archive manifest member. The historical sorted 19-file inventory is `/tmp/shot-pattern-delivery-resume/release-assets-v2.sha256` (SHA256 `292819d22784aebcc2c5c4453ec8c2f2308a9d6073bb9fd9b98ddb954dac6f72`); the durable inventory is `validation/artifact_delivery_v2/release_staging_inventory.json` (SHA256 `65f66433077e0ae6795dc480abd451b5e7f6c7de70e6c201abca08290c4541bc`). It included all six dynamically generated supplement ZIPs and unique manifest names. The authoritative current 21-asset inventory is `validation/artifact_delivery_v3/release_staging_inventory_v3.json`; its remote status is verified by the v3 receipt above.

## Build and Verify Supplemental Bundle

Run from STUDY only after DELIVERY includes the accepted final child v2 source and all review evidence is integrated. Use a fresh output directory; every ZIP must remain at or below decimal 500,000,000 bytes. The script automatically creates as many ZIP parts as required; never assume a fixed part count.

```bash
set -euo pipefail
STUDY=/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift
SOURCE="$STUDY/docs/research/shot_pattern_analysis/phase2/supplement_report_v2_layout_repair"
OUT=/tmp/shot-pattern-delivery-resume/supplement-v2
cd "$STUDY"
python -m scripts.research.shot_pattern_analysis.archive_data "$SOURCE" "$OUT" --max-bytes 500000000
```

Verify the bundle from DELIVERY’s source-controlled API and save hashes before staging:

```bash
set -euo pipefail
DELIVERY=/home/dieterolson/.codex/worktrees/shot-pattern-main-integration/UpstreamDrift
OUT=/tmp/shot-pattern-delivery-resume/supplement-v2
cd "$DELIVERY"
PYTHONPATH=. python - "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
from scripts.research.shot_pattern_analysis.archive_data import verify_archives
p = Path(sys.argv[1])
count = verify_archives(p)
manifest = json.loads((p / "archive_manifest.json").read_text())
assert all((p / name).stat().st_size <= 500_000_000 for name in manifest["archives"])
print("members", count)
for path in sorted(p.iterdir()):
    if path.is_file(): print(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

## Unique Release Asset Names and Remote Restore

Stage copies under distinct actual basenames; `gh release upload` labels do not rename files. Enumerate every generated ZIP dynamically. Keep each bundle’s manifest separately prefixed because all three source manifests are named `archive_manifest.json`. The manifest contents retain original ZIP names, so download into isolated directories and remove exactly one `primary__`, `control__`, or `supplement__` prefix before verifying.

The report has passed human visual review and the phase-two research prerelease is complete at the exact pushed DELIVERY commit SHA stated above. The release is a research artifact publication only; it does not establish production readiness. The ready-for-review PR must disclose the failed repository-wide gate evidence, carry a clear `Blocked:` section, and must not claim merge readiness. The commands that follow preserve the historical procedure; do not rerun or duplicate the completed remote publication.

```bash
set -euo pipefail
REPO=D-sorganization/UpstreamDrift
TAG=shot-pattern-phase2-research-2026-10-10
FINAL_PUSHED_DELIVERY_SHA='<exact pushed DELIVERY commit SHA>'
PREP=/tmp/shot-pattern-delivery-resume/release-assets-primary-control-prepared
SUP=/tmp/shot-pattern-delivery-resume/supplement-v2
STAGE=/tmp/shot-pattern-delivery-resume/release-assets-v2
mkdir "$STAGE"
cp --reflink=auto "$PREP/primary_final_figures_v1_001.zip" "$STAGE/primary__primary_final_figures_v1_001.zip"
cp --reflink=auto "$PREP/primary_archive_manifest.json" "$STAGE/primary__archive_manifest.json"
cp --reflink=auto "$PREP/control_final_figures_v1_001.zip" "$STAGE/control__control_final_figures_v1_001.zip"
cp --reflink=auto "$PREP/control_archive_manifest.json" "$STAGE/control__archive_manifest.json"
cp --reflink=auto "$SUP/archive_manifest.json" "$STAGE/supplement__archive_manifest.json"
while IFS= read -r -d '' f; do cp --reflink=auto "$f" "$STAGE/supplement__$(basename "$f")"; done < <(find "$SUP" -maxdepth 1 -type f -name '*.zip' -print0 | sort -z)
cmp "$PREP/primary_archive_manifest.json" "$STAGE/primary__archive_manifest.json"
cmp "$PREP/control_archive_manifest.json" "$STAGE/control__archive_manifest.json"
cmp "$SUP/archive_manifest.json" "$STAGE/supplement__archive_manifest.json"
sha256sum "$STAGE"/* > /tmp/shot-pattern-delivery-resume/release-assets-v2.sha256
# Execute only after the exact pushed SHA and ready-for-review PR are recorded.
NOTES=/tmp/shot-pattern-delivery-resume/release-notes-research-prerelease.md
cat > "$NOTES" <<EOF
Research prerelease at exact DELIVERY commit $FINAL_PUSHED_DELIVERY_SHA.
Repository-wide pre-PR validation exited nonzero; see the ready-for-review PR for gate results and blocked merge disposition. This publication is not production or merge-readiness approval.
EOF
gh release create "$TAG" --repo "$REPO" --target "$FINAL_PUSHED_DELIVERY_SHA" --title 'Shot Pattern Phase Two Research Artifacts' --notes-file "$NOTES"
gh release upload "$TAG" --repo "$REPO" "$STAGE"/*
```

Fresh remote download and verify (only after successful upload):

```bash
set -euo pipefail
REPO=D-sorganization/UpstreamDrift
TAG=shot-pattern-phase2-research-2026-10-10
STAGE=/tmp/shot-pattern-delivery-resume/release-assets-v2
RESTORE=/tmp/shot-pattern-delivery-resume/remote-restoration-v2
mkdir -p "$RESTORE/primary" "$RESTORE/control" "$RESTORE/supplement"
gh release download "$TAG" --repo "$REPO" --dir "$RESTORE/primary" --pattern 'primary__*'
gh release download "$TAG" --repo "$REPO" --dir "$RESTORE/control" --pattern 'control__*'
gh release download "$TAG" --repo "$REPO" --dir "$RESTORE/supplement" --pattern 'supplement__*'
# Establish manifest identity before parsing any downloaded manifest.
cmp "$STAGE/primary__archive_manifest.json" "$RESTORE/primary/primary__archive_manifest.json"
cmp "$STAGE/control__archive_manifest.json" "$RESTORE/control/control__archive_manifest.json"
cmp "$STAGE/supplement__archive_manifest.json" "$RESTORE/supplement/supplement__archive_manifest.json"
cd /home/dieterolson/.codex/worktrees/shot-pattern-main-integration/UpstreamDrift
PYTHONPATH=. python - "$RESTORE" <<'PY'
import hashlib, json, sys
from pathlib import Path
from zipfile import ZipFile
from scripts.research.shot_pattern_analysis.archive_data import verify_archives

base = Path(sys.argv[1])
for label in ("primary", "control", "supplement"):
    bundle = base / label
    (bundle / f"{label}__archive_manifest.json").rename(bundle / "archive_manifest.json")
    manifest = json.loads((bundle / "archive_manifest.json").read_text())
    remote_zips = sorted(bundle.glob(f"{label}__*.zip"))
    expected_names = set(manifest["archives"])
    assert {p.name[len(label) + 2:] for p in remote_zips} == expected_names
    for path in remote_zips:
        path.rename(bundle / path.name[len(label) + 2:])
    assert verify_archives(bundle) == len(manifest["files"])
    restored = base / f"{label}-restored"
    restored.mkdir(exist_ok=False)
    for name in manifest["archives"]:
        with ZipFile(bundle / name) as archive:
            archive.extractall(restored)
    actual = {p.relative_to(restored).as_posix() for p in restored.rglob("*") if p.is_file()}
    assert actual == set(manifest["files"]), (label, len(actual), len(manifest["files"]))
    for member, record in manifest["files"].items():
        path = restored / member
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    print(label, "freshly restored", len(actual), "files")
PY
```

After the verified extraction, route the supplement into a fresh publication tree while keeping report bytes unchanged. The consumer layout is `phase2/REPORT.md` alongside `phase2/supplement_report_v1/`; copy the frozen accepted `REPORT.md` byte-for-byte from DELIVERY, then move the verified child directory into the linked canonical basename. Example:

```bash
set -euo pipefail
DELIVERY=/home/dieterolson/.codex/worktrees/shot-pattern-main-integration/UpstreamDrift
PHASE2_TREE=/tmp/shot-pattern-delivery-resume/publication/docs/research/shot_pattern_analysis/phase2
mkdir -p "$PHASE2_TREE"
cp --reflink=auto "$DELIVERY/docs/research/shot_pattern_analysis/phase2/REPORT.md" "$PHASE2_TREE/REPORT.md"
cmp "$DELIVERY/docs/research/shot_pattern_analysis/phase2/REPORT.md" "$PHASE2_TREE/REPORT.md"
test ! -e "$PHASE2_TREE/supplement_report_v1"
mv /tmp/shot-pattern-delivery-resume/remote-restoration-v2/supplement-restored/supplement_report_v2_layout_repair \
  "$PHASE2_TREE/supplement_report_v1"
rg -o 'supplement_report_v1/[^)]+' "$PHASE2_TREE/REPORT.md" | while IFS= read -r rel; do test -f "$PHASE2_TREE/$rel"; done
```

The plan’s consumer target is the markdown report and its relative graphic/CSV/JSON links. The frozen report has 98 `supplement_report_v1/...` link occurrences and 98 unique targets; the verified publication tree resolves all 98. The remote receipt records downloaded asset hashes, trusted manifest comparisons, complete member restoration, source ZIP checks, and link routing. It verifies transport and complete file restoration only; it does not repeat visual review or establish scientific acceptance.
