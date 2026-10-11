# Shot Pattern Phase-Two Artifact Delivery Plan

Read-only execution preparation; no release or remote state has been changed.

## Bundle API and Local Evidence

The packaging API is `scripts/research/shot_pattern_analysis/archive_data.py`:

- `build_archives(source, output, max_bytes=...)` rejects an existing output, hidden entries, symlinks, unsafe ZIP members, and a source nested under the output; it writes a deterministic ZIP set plus `archive_manifest.json` and calls `verify_archives` before atomically publishing.
- `verify_archives(output)` checks each archive SHA-256 and size, safe/regular member metadata, exact member inventory, and every member's byte count and SHA-256. It returns the number of files.
- There is no public graphic-bundle restore function. The accepted primary/control restoration scripts use `verify_archives`, parse the manifest, extract each manifest-listed ZIP with `ZipFile`, compare complete file inventories/hashes, and for data archives also compare loaded reports. For graphics, use the same archive verifier plus the full extraction inventory check below; this proves transport bytes only, while visual/scientific acceptance remains separate.

The integration checkout was clean at `89fe6d3efa`. Existing bundle receipt hashes match the local ZIPs:

| Bundle | Archive | Bytes | ZIP SHA-256 | Accepted local receipt |
| --- | --- | ---: | --- | --- |
| Primary | `final_figures_v1-001.zip` | 223,739,795 | `6ba794355bf4575e2a6fb24a3dc7473dbd0279a628c3e99bbe191ce2a4785cc2` | `publication_primary_bundles_v1/restoration.json` |
| Control | `control_final_figures_v1-001.zip` | 168,462,247 | `0101c2cbc5cba3ffb0b6bded0979cb63fb28adb52f446fbda1a4384c226e35c0` | `publication_control_bundles_v1/restoration.json` |

These receipts establish previous fresh local restorations, not remote transfer. The local archive manifests currently hash to `3375d756c1aa39fb985ebccfb46600de8e1e07714f3dee5561ec7e5ae6229d44` (primary) and `492962d0dc0a2f43ea95b143d352b4234331a8a2c42d10b33c1f7a55c9dbae9c` (control).

## Build the Complete Supplemental Graphics Bundle

Run only after independent full graphic review and report acceptance. This packages the entire source directory, including its manifest and all graphic formats. Keep the output path new. The script's bound is decimal 500,000,000 bytes per ZIP, including ZIP directory overhead.

```bash
set -euo pipefail
cd /home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift
python -m scripts.research.shot_pattern_analysis.archive_data \
  docs/research/shot_pattern_analysis/phase2/supplement_report_v1 \
  /tmp/shot-pattern-delivery-resume/supplement \
  --max-bytes 500000000
```

After packaging, verify and capture archive sizes/hashes before any upload:

```bash
set -euo pipefail
PYTHONPATH=. python -c 'from pathlib import Path; from scripts.research.shot_pattern_analysis.archive_data import verify_archives; p=Path("/tmp/shot-pattern-delivery-resume/supplement"); print("members", verify_archives(p)); print(*[(x.name, x.stat().st_size) for x in sorted(p.glob("*.zip"))], sep="\\n")'
sha256sum /tmp/shot-pattern-delivery-resume/supplement/*.zip /tmp/shot-pattern-delivery-resume/supplement/archive_manifest.json
```

Do not proceed if any ZIP is over 500,000,000 bytes or verification fails.
Before staging, verify the two accepted local publication bundles and the new supplement bundle against their embedded manifests using the same API:

```bash
set -euo pipefail
cd /home/dieterolson/.codex/worktrees/shot-pattern-main-integration/UpstreamDrift
PYTHONPATH=. python -c 'from pathlib import Path; from scripts.research.shot_pattern_analysis.archive_data import verify_archives; roots=[Path("/tmp/shot-pattern-publication-bundles-v1/primary"),Path("/tmp/shot-pattern-publication-bundles-v1/control"),Path("/tmp/shot-pattern-delivery-resume/supplement")]; print([(str(p),verify_archives(p)) for p in roots])'
```

## Release Naming and Upload

Proposed immutable phase-two tag: `shot-pattern-phase2-2026-10-10`. Create it only after `python -m scripts.pre_pr` passes and the final delivery commit is pushed. Resolve `FINAL_PUSHED_DELIVERY_SHA` from that push; do not point the tag at the current checkpoint. The release is tied to that commit via `--target`.

`gh release upload --help` confirms the syntax is `<tag> <files>...`; suffix `#...` sets only the asset display label. It does not rename the asset basename. Therefore stage byte-identical copies under distinct actual filenames, including a `supplement__` prefix for every supplemental ZIP. Keep the staged manifest copies separately named. First create the release/tag at the already pushed final delivery SHA, then upload the staged files:

```bash
set -euo pipefail
REPO=D-sorganization/UpstreamDrift
TAG=shot-pattern-phase2-2026-10-10
FINAL_PUSHED_DELIVERY_SHA='<FINAL_PUSHED_DELIVERY_SHA>'
STAGE=/tmp/shot-pattern-delivery-resume/release-assets
mkdir "$STAGE"
cp --reflink=auto /tmp/shot-pattern-publication-bundles-v1/primary/final_figures_v1-001.zip "$STAGE/primary__final_figures_v1-001.zip"
cp --reflink=auto /tmp/shot-pattern-publication-bundles-v1/primary/archive_manifest.json "$STAGE/primary__archive_manifest.json"
cp --reflink=auto /tmp/shot-pattern-publication-bundles-v1/control/control_final_figures_v1-001.zip "$STAGE/control__control_final_figures_v1-001.zip"
cp --reflink=auto /tmp/shot-pattern-publication-bundles-v1/control/archive_manifest.json "$STAGE/control__archive_manifest.json"
cp --reflink=auto /tmp/shot-pattern-delivery-resume/supplement/archive_manifest.json "$STAGE/supplement__archive_manifest.json"
for f in /tmp/shot-pattern-delivery-resume/supplement/*.zip; do
  cp --reflink=auto "$f" "$STAGE/supplement__$(basename "$f")"
done
cmp /tmp/shot-pattern-publication-bundles-v1/primary/archive_manifest.json "$STAGE/primary__archive_manifest.json"
cmp /tmp/shot-pattern-publication-bundles-v1/control/archive_manifest.json "$STAGE/control__archive_manifest.json"
cmp /tmp/shot-pattern-delivery-resume/supplement/archive_manifest.json "$STAGE/supplement__archive_manifest.json"
printf '%s  %s\n' \
  3375d756c1aa39fb985ebccfb46600de8e1e07714f3dee5561ec7e5ae6229d44 \
  /tmp/shot-pattern-publication-bundles-v1/primary/archive_manifest.json \
  492962d0dc0a2f43ea95b143d352b4234331a8a2c42d10b33c1f7a55c9dbae9c \
  /tmp/shot-pattern-publication-bundles-v1/control/archive_manifest.json | sha256sum -c -
# Record the just-built supplemental manifest hash in the delivery receipt before upload.
sha256sum /tmp/shot-pattern-delivery-resume/supplement/archive_manifest.json
sha256sum "$STAGE"/* > /tmp/shot-pattern-delivery-resume/release-assets.sha256
gh release create "$TAG" --repo "$REPO" --target "$FINAL_PUSHED_DELIVERY_SHA" \
  --title 'Shot Pattern Phase Two Artifacts' \
  --notes 'Reviewed phase-two shot-pattern graphics and manifests.'
gh release upload "$TAG" --repo "$REPO" "$STAGE"/*
```

The staging copies preserve source bytes and give every remote asset a unique actual basename. The manifest contents continue to list the original ZIP names; the restore mapping below strips exactly one dataset prefix. A release is a remote mutation and remains unperformed in this preparation.

## Download, Map, and Verify Remote Assets

Download into three separate directories. Rename each downloaded manifest back to the API-required `archive_manifest.json` within its own directory. Map the prefixed ZIP asset names back to the exact names listed inside their manifests before calling `verify_archives`:

```bash
set -euo pipefail
REPO=D-sorganization/UpstreamDrift
TAG=shot-pattern-phase2-2026-10-10
RESTORE=/tmp/shot-pattern-delivery-resume/remote-restoration
mkdir -p "$RESTORE/primary" "$RESTORE/control" "$RESTORE/supplement"
gh release download "$TAG" --repo "$REPO" --dir "$RESTORE/primary" --pattern 'primary__*'
gh release download "$TAG" --repo "$REPO" --dir "$RESTORE/control" --pattern 'control__*'
gh release download "$TAG" --repo "$REPO" --dir "$RESTORE/supplement" --pattern 'supplement__*'
```

Before parsing any downloaded manifest, compare it byte for byte with the staged, locally trusted manifest. Then rename each actual prefixed asset basename back to the exact filename listed in the manifest. Run the following from the integration checkout, whose archive API is source-controlled:

```bash
set -euo pipefail
cd /home/dieterolson/.codex/worktrees/shot-pattern-main-integration/UpstreamDrift
cmp /tmp/shot-pattern-delivery-resume/release-assets/primary__archive_manifest.json "$RESTORE/primary/primary__archive_manifest.json"
cmp /tmp/shot-pattern-delivery-resume/release-assets/control__archive_manifest.json "$RESTORE/control/control__archive_manifest.json"
cmp /tmp/shot-pattern-delivery-resume/release-assets/supplement__archive_manifest.json "$RESTORE/supplement/supplement__archive_manifest.json"
python - "$RESTORE" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
for label in ("primary", "control", "supplement"):
    bundle = base / label
    (bundle / f"{label}__archive_manifest.json").rename(bundle / "archive_manifest.json")
    manifest = json.loads((bundle / "archive_manifest.json").read_text())
    for archive in manifest["archives"]:
        (bundle / f"{label}__{archive}").rename(bundle / archive)
PY
PYTHONPATH=. python - "$RESTORE" <<'PY'
import hashlib, json, sys
from pathlib import Path
from zipfile import ZipFile
from scripts.research.shot_pattern_analysis.archive_data import verify_archives

base = Path(sys.argv[1])
for label in ("primary", "control", "supplement"):
    bundle = base / label
    manifest = json.loads((bundle / "archive_manifest.json").read_text())
    count = verify_archives(bundle)
    destination = base / (label + "-restored")
    destination.mkdir(exist_ok=False)
    for archive in manifest["archives"]:
        with ZipFile(bundle / archive) as zf:
            zf.extractall(destination)
    expected = manifest["files"]
    actual_paths = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    assert actual_paths == set(expected), (label, len(actual_paths), len(expected))
    for member, record in expected.items():
        path = destination / member
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.stat().st_size == record["bytes"] and digest == record["sha256"], (label, member)
    assert count == len(expected)
    print(label, "verified and restored", count, "files")
PY
```

This remote check verifies manifest/archive hashes and full restored file inventory/bytes. It does not re-run the visual review or establish scientific acceptance. Preserve the downloaded directories and command output as delivery evidence; do not infer successful restoration from upload success.

## Remaining Delivery Gates

The v6 producer receipt records exit code 0 and `completed_pending_full_independent_acceptance`. Full supplemental visual review, final report/page review, report acceptance, and pre-PR remain outstanding in the evidence inspected. No remote release exists yet, and this plan does not authorize bypassing those gates.
