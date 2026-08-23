# Governed Measured-Trajectory Acquisition and Ingestion

## Purpose and Scientific Boundary

This contract defines the minimum evidence needed before a participant golf
trajectory may enter the issue-#9004 qualification pipeline. It extends the
canonical motion-pipeline adapters; it does not introduce another C3D, JSON, or
CSV parser.

Passing this ingestion gate proves only that an immutable, authorized artifact
has the declared provenance, grouping, coordinate, event, channel, and
uncertainty records and that the canonical adapter accepted it. It cannot prove
that the analytic swing is representative, establish a human mechanism, infer
bilateral hand wrenches, identify muscle or scapular intent, or support coaching
guidance.

## Authority Chain

Every admitted trajectory requires three independently validated records:

1. `measured_trajectory_source_registry.json` must identify the source as a
   human-observed qualification candidate with explicit reuse authority,
   participant grouping, body and club trajectories, calibration,
   synchronization metadata, an immutable source-package digest, and no open
   blocker.
2. `measured_trajectory_metric_registration.json` must retain the registered
   participant split, metrics, frames, events, controls, uncertainties, and
   inference boundary. Its authority state must agree with the source registry.
3. A per-trial `measured-trajectory-artifact/v1` manifest must bind the source
   package, decoded trajectory, participant, trial, acquisition processing,
   frames, events, channels, and uncertainties.

The gateway recomputes registry and metric readiness. Stored summaries are not
trusted.

## Artifact Manifest

The manifest is duplicate-key-rejecting JSON with these top-level fields:

| Field                | Requirement                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------- |
| `schema_version`     | Exactly `measured-trajectory-artifact/v1`                                                          |
| `manifest_id`        | Unique lowercase hyphenated identifier                                                             |
| `created_at_utc`     | Immutable creation timestamp                                                                       |
| `source_registry_id` | Exactly `articulated-golf-trajectory-sources-v1`                                                   |
| `source_id`          | Exact source-registry identifier                                                                   |
| `artifact`           | Package path/digest, trajectory path/digest, and canonical adapter hint                            |
| `participant`        | Participant identifier, grouping identifier, and cohort                                            |
| `acquisition`        | Trial, sampling, units, synchronization, filtering, reconstruction, and anthropometric authorities |
| `frames`             | Ordered laboratory, anatomical, model, and club frame records                                      |
| `events`             | Ordered downswing-start and impact records                                                         |
| `channels`           | Sorted unique list of observed or derived channel identifiers                                      |
| `uncertainties`      | Ordered records for all six registered uncertainty analyses                                        |
| `intended_use`       | `pipeline_probe` or `held_out_qualification`                                                       |
| `inference_boundary` | Explicit statement of prohibited human, wrench, and coaching inferences                            |

### Immutable Files

`artifact` records both the original authorized package and the decoded
trajectory. The original package SHA-256 must reproduce the source-registry
digest. The trajectory SHA-256 must reproduce the per-trial manifest. Paths
must be relative to and contained beneath the manifest directory. Pickle,
joblib, absolute, and parent-traversal paths are rejected before a parser runs.

For a single-file distribution, the source-package and trajectory paths may be
the same. For an archive distribution, retain the unmodified archive alongside
the extracted trajectory so both provenance layers remain independently
verifiable.

### Participant and Acquisition Records

`participant_id` identifies the person within the governed source;
`grouping_id` is the indivisible split unit. Trials from one grouping identifier
must never cross training and held-out partitions. Frame-wise random splitting
is prohibited.

The canonical ingestion boundary uses metres, radians, and seconds. Conversion
from source units must remain reproducible from the acquisition and adapter
records. Sampling rate must be positive. Synchronization, filtering, marker or
keypoint reconstruction, and anthropometric sources must be named rather than
left implicit.

## Coordinate and Event Records

The four frame records occur in this fixed order:

1. `lab` — right-handed acquisition frame retaining gravity and floor
   directions;
2. `anatomical` — participant-specific segment frames;
3. `model` — articulated generalized-coordinate and body frames; and
4. `club` — declared club origin, face, shaft, and swing-normal axes.

Each record requires a nonempty definition and transform authority, a SHA-256
for the transform record, and finite nonnegative translation and rotation
uncertainties. A future mapping runner must verify the referenced transform
content before using it; the digest is not a claim that a transform is
biomechanically correct.

The two event records occur in this fixed order:

1. `downswing_start`; and
2. `impact`.

Each requires a time, detector identifier and version, finite nonnegative timing
uncertainty, and `missing_policy=unavailable_not_zero`. Impact must occur after
downswing start. A kinematic proxy must remain labeled as a proxy and retain its
full uncertainty interval.

## Channel Coverage

The gateway compares the declared channel list with every preregistered metric.
A metric is available only when all its required channels are present. Missing
channels are returned explicitly as `missing_channel_ids`; they are never
materialized as zero. This permits partial pipeline probes while preventing a
partial trial from masquerading as complete qualification evidence.

## Uncertainty Coverage

The manifest must retain ordered, finite bounds and a named method for:

1. time alignment;
2. filtering;
3. coordinate mapping;
4. marker reconstruction;
5. event detection; and
6. anthropometric scaling.

The units may differ by method but cannot be implicit. Later analysis must run
the registered perturbations rather than select favorable alternatives after
examining held-out outcomes.

## Fail-Closed Execution Order

`load_governed_trajectory` performs the following sequence:

1. reject duplicate keys and malformed or extra manifest fields;
2. recompute source-registry and metric-registration readiness;
3. require registry and preregistration authority states to agree;
4. require the source to qualify for the declared intended use;
5. contain and hash the source package and trajectory;
6. compare the package digest with both manifest and source registry;
7. compare the trajectory digest with the artifact manifest;
8. delegate to the canonical contract-checked motion-source adapter; and
9. report available metrics, unavailable metrics, and missing channels.

The returned envelope always retains
`human_inference_ready=false` and
`bilateral_wrench_gate_satisfied=false`. Those claims require downstream
participant-held-out results and, for bilateral force allocation, the separate
issue-#8556 synchronized-wrench evidence.

## Current Acquisition Blocker

No current source passes this gate. GolfPose requires author authorization,
verified reuse terms, delivered-file digests, and participant/trial inventory.
KIT motion 1319 lacks explicit file-reuse authority, verified club calibration,
complete synchronization metadata, and enough participants for held-out
qualification. Simulation exports and software fixtures remain inadmissible as
human validation.

When an authorized package becomes available, register its immutable digest,
preserve the raw download, create one artifact manifest per trial, and run the
focused contract tests before any mapping, fitting, or outcome inspection.

```powershell
python -m pytest tests/research/test_measured_trajectory_source_registry.py `
  tests/research/test_measured_trajectory_metric_registration.py `
  tests/research/test_measured_trajectory_ingestion.py -q
```
