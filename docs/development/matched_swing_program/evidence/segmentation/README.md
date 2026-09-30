# Real Body and Club Segmentation Evidence Package (MMR-13, #11099)

Contains the model card, the checkpoint-pin runbook, and the bounded benchmark
harness status for automated neural silhouette segmentation. The package is
**honest by construction**: every report entry is produced by
`evaluate_segmentation_benchmark`, which scores the adapter under test against
independently recorded gold artifacts and emits a typed insufficiency result —
never fabricated numbers — whenever evidence is absent.

## Current Benchmark Status: Blocked (no fabricated metrics)

`benchmark_report.json` records `status: "blocked"` with
`reason_code: "missing_evidence"`. The evidence required to score real body and
club segmentation does not exist in this repository:

1. **Pinned checkpoint weights** — no SAM-ViT-B / MobileSAM checkpoint has been
   provisioned offline, and no trusted SHA-256 pin is committed anywhere (the
   hash strings previously claimed here were fabricated placeholders and have
   been removed; see `MODEL_CARD.md`).
2. **Held-out clip footage** — no modern high-speed, 1953-era archive, or
   occluded gallery clip (the categories the original PR body claimed) has
   been ingested into this evidence directory.
3. **Independent gold masks** — no human-reviewed gold label artifacts
   (`gold_masks/*.npy`) with a `clip_manifest.json` binding exist.

Until every item above is provisioned by the operator, the benchmark reports
`missing_evidence` and `metrics_reported: false`; body IoU, club recall,
boundary F1, correction effort, latency and memory are **not reported**. The
`RealSegmentationAdapter` in the same state raises
`SegmentationUnavailableError` instead of returning synthetic masks. The
committed report is regenerable with the reproduction command below; it is not
hand-maintained.

## Independent Gold Mask Contract

Gold evidence lives beside this document and is the only scoring input:

- `clip_manifest.json`: `schema_version`, `benchmark`, and a `clips` list with
  `clip_id`, `clip_type`, `description`, `width_px`, `height_px`,
  `adverse_conditions`, and `gold_masks` rows.
- Each gold row: `frame_id`, `file` (path relative to the evidence dir),
  `sha256` of the `.npy` payload, `pts_ticks`, and the tick `timebase`
  numerator/denominator (plus optional `physical_time_s` +
  `physical_time_reason` recorded from the source clip).
- Gold label files are uint8 label images: `0` background, `1` body, `2` club,
  `3` invalid/occluded. Valid region = `label != 3`.
- Decoded source pixels are supplied by the caller (`frame_source`); the
  benchmark verifies the recorded gold SHA-256 before scoring and fails the
  clip closed on any mismatch (`status: invalid_evidence`).

## Reproduction

The committed `benchmark_report.json` is produced by the library function.
This is exactly the call that regenerated the committed file in the current
repository state — the benchmark blocks on the missing evidence package
before the adapter is used, and records that no checkpoint weights are
pinned:

```bash
python - <<'PY'
import hashlib, json, tempfile
from pathlib import Path
from shared.python.shadow_tracker.model_segmentation import (
    RealSegmentationAdapter,
    evaluate_segmentation_benchmark,
)

# No SAM/MobileSAM weights exist anywhere in this repository; a scratch file
# stands in only so the adapter (which requires an existing checkpoint path)
# can be constructed. The adapter never runs it.
scratch = Path(tempfile.mkdtemp())
checkpoint = scratch / "sam-vit-b-golf-v1.0.0.onnx"
checkpoint.write_bytes(b"no weights provisioned in this repository")
adapter = RealSegmentationAdapter(
    "sam-vit-b-golf",
    checkpoint,
    expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
)

report = evaluate_segmentation_benchmark(
    adapter,
    evidence_dir=Path(
        "docs/development/matched_swing_program/evidence/segmentation"
    ),
)
target = Path(
    "docs/development/matched_swing_program/evidence/segmentation/"
    "benchmark_report.json"
)
target.write_text(json.dumps(report, indent=2), encoding="utf-8")
PY
```

To obtain real metrics, an operator must:

1. Provision an offline checkpoint (TorchScript or ONNX export that the
   installed runtime can execute) and pin its actual SHA-256 at the
   `expected_sha256` argument; supply a calibrated body/club `postprocess`
   decode callable to the adapter.
2. Record held-out clip evidence into this directory per the gold-mask
   contract above (manifest + SHA-256-bound label files).
3. Supply `frame_source(index)` to `evaluate_segmentation_benchmark` with the
   decoded pixels and PTS/timebase metadata for each gold frame (the
   `shadow_tracker.ingestion.OpenCvVideoDecoder` provides PTS-provenanced
   decoding when OpenCV is installed).

Until those inputs exist the benchmark reports `blocked` with
`metrics_reported: false`; there is no offline path that produces numeric
metrics today, and none is claimed in `SPEC.md` or the handoff documents.

## Zero Hidden Downloads Protocol

- No automatic network downloads of model weights during test execution or
  tool initialization.
- Missing checkpoints fail fast with the typed
  `SegmentationUnavailableError` (adapter) or actionable `FileNotFoundError`
  (`verify_checkpoint`).
- Checkpoints that fail the operator-supplied SHA-256 pin fail closed with
  `RuntimeError`; weights-only state dicts that no runtime can execute are
  rejected as typed unavailability.
- All automated inferences that do run (dev-only checkpoint-validation harness
  against operator-pinned artifacts) are tagged as unreviewed drafts and
  registered in `ManualMaskProvider` for human review.