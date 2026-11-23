"""Versioned evidence envelope for the restricted research solution prototype.

A v4 base bundle and compressed, size-bounded NPZ stage archive are referenced
by hashes in the v5 envelope. The envelope is atomically linked last; interruption
may leave orphan files, and directory crash durability is not guaranteed.
The archive stores JSON bytes within NPZ, so large runs may consume substantial
RAM during encoding and decoding. An encoding-only synthetic 42,304-stage,
47-channel benchmark used 154.6 MB JSON, 1.27 MB NPZ, 25 s encoding and 403 MiB
traced peak; repeated numbers make compression unrealistically favorable, and
recorder, loading, and replay overhead were excluded. Full-run support is
explicitly rejected above 4,096 stages; typed numeric arrays and an actual
representative encode/decode benchmark are needed before raising that limit. No older bundle can be upgraded without
an actual recording. This is not a contact or post-actuator certificate.
"""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import zipfile
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .solution_bundle_prototype import (
    ReplayComparison,
    SolutionBundle,
    _unique_object,
    load_solution,
    save_solution,
    verify_forward_resimulation,
)
from .stage_archive import (
    compare_stage_archives,
    decode_stage_archive,
    encode_stage_archive,
)
from .stage_trace import StageRecorder

SCHEMA = "research-evidenced-solution/v5"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_STAGES = 4096


def _bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _check_plan(bundle: SolutionBundle, recorder: StageRecorder) -> None:
    if not recorder.capture_ideal_drive:
        raise ValueError("ideal stage evidence required for v5/v6")
    plan = recorder.plan
    runner = bundle.manifest["runner"]
    if (
        plan is None
        or runner.get("mode") != "mujoco_ideal_zoh"
        or plan.requested_steps != len(bundle.arrays["recorded_time_s"]) - 1
        or plan.coordinate_order != tuple(bundle.manifest["conventions"]["joint_order"])
        or plan.dt_s != runner["dt_s"]
        or plan.substeps != runner["substeps"]
        or plan.effort_holds != bundle.manifest["controller"]["holds_per_step"]
        or plan.project != runner["project"]
    ):
        raise ValueError("stage recording and bundle plan mismatch")
    a = bundle.arrays
    for index, step in enumerate(recorder.steps):
        if (
            not step.stages
            or not np.array_equal(step.stages[0].q, a["recorded_q"][index])
            or not np.array_equal(step.stages[0].v, a["recorded_qd"][index])
            or not np.array_equal(step.post_projection_q, a["recorded_q"][index + 1])
            or not np.array_equal(step.post_projection_v, a["recorded_qd"][index + 1])
        ):
            raise ValueError("stage and recorded state mismatch")
        for stage in step.stages:
            hold = index * plan.effort_holds + stage.hold
            if (
                not np.array_equal(stage.tau, a["ideal_input_tau"][hold])
                or stage.applied_drive is None
                or not np.array_equal(stage.applied_drive, a["ideal_input_tau"][hold])
            ):
                raise ValueError("stage and held ideal drive mismatch")
    impact = bundle.manifest["impact"]
    if impact["status"] == "recorded_latched_force":
        event_stages = [
            s
            for step in recorder.steps
            for s in step.stages
            if s.stage == 1 and abs(s.time_s - impact["time_s"]) <= 1e-12
        ]
        if not event_stages or not any(
            np.array_equal(s.q, impact["event_q"])
            and np.array_equal(s.v, impact["event_qd"])
            for s in event_stages
        ):
            raise ValueError("impact event has no matching observed stage state")


@dataclass(frozen=True)
class EvidencedSolution:
    bundle: SolutionBundle
    archive: dict[str, Any]
    archive_sha256: str


def save_evidenced_solution(
    path: Path, bundle: SolutionBundle, recorder: StageRecorder
) -> str:
    """Publish an envelope last; trusted digest must be kept separately."""
    path = Path(path)
    if path.suffix != ".json" or path.exists():
        raise ValueError("a new JSON envelope path is required")
    if sum(len(step.stages) for step in recorder.steps) > MAX_STAGES:
        raise ValueError("full-run stage archive is unsupported; typed arrays required")
    archive = encode_stage_archive(recorder)
    _check_plan(bundle, recorder)
    base = path.with_name(path.stem + ".base.json")
    base_sha = save_solution(base, bundle)
    stage_json = _bytes(archive)
    if len(stage_json) > MAX_ARCHIVE_BYTES:
        raise ValueError("stage archive exceeds research size limit")
    stage_buffer = BytesIO()
    np.savez_compressed(
        stage_buffer, stage_json=np.frombuffer(stage_json, dtype=np.uint8)
    )
    stage_npz = stage_buffer.getvalue()
    if len(stage_npz) > MAX_ARCHIVE_BYTES:
        raise ValueError("compressed stage archive exceeds research size limit")
    stage_path = path.with_name(path.stem + ".stages.npz")
    with stage_path.open("xb") as stream:
        stream.write(stage_npz)
        stream.flush()
        os.fsync(stream.fileno())
    envelope = {
        "schema": SCHEMA,
        "base_manifest_sha256": base_sha,
        "stage_archive_sha256": _hash(stage_npz),
    }
    encoded = _bytes(envelope)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".evidence-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return _hash(encoded)


def load_evidenced_solution(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_model_sha256: str,
    expected_runtime_sha256: str,
    expected_assets: Mapping[str, str] | None = None,
) -> EvidencedSolution:
    """Reject missing, partial, changed, or older evidence, with no migration."""
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError("evidence manifest exceeds size limit")
    raw = path.read_bytes()
    if _hash(raw) != expected_manifest_sha256:
        raise ValueError("evidence manifest hash mismatch")
    try:
        envelope = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid evidence manifest") from exc
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"schema", "base_manifest_sha256", "stage_archive_sha256"}
        or envelope["schema"] != SCHEMA
    ):
        raise ValueError(
            "unsupported or incomplete evidence schema; re-recording required"
        )
    stage_path = path.with_name(path.stem + ".stages.npz")
    if stage_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("compressed stage archive exceeds research size limit")
    stage_npz = stage_path.read_bytes()
    if _hash(stage_npz) != envelope["stage_archive_sha256"]:
        raise ValueError("stage archive hash mismatch")
    try:
        with zipfile.ZipFile(BytesIO(stage_npz)) as zipped:
            entries = zipped.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != "stage_json.npy"
                or entries[0].file_size > MAX_ARCHIVE_BYTES + 4096
            ):
                raise ValueError("unsupported or oversized stage NPZ")
            with zipped.open(entries[0]) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                elif version in ((2, 0), (3, 0)):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("unsupported NPY version")
                if (
                    len(shape) != 1
                    or shape[0] > MAX_ARCHIVE_BYTES
                    or dtype != np.dtype(np.uint8)
                    or fortran
                    or member.tell() + shape[0] != entries[0].file_size
                ):
                    raise ValueError("invalid stage NPY header")
        with np.load(BytesIO(stage_npz), allow_pickle=False) as arrays:
            stage_data = arrays["stage_json"]
            if (
                stage_data.dtype != np.uint8
                or stage_data.ndim != 1
                or stage_data.size > MAX_ARCHIVE_BYTES
            ):
                raise ValueError("invalid stage NPZ array")
            archive = json.loads(stage_data.tobytes(), object_pairs_hook=_unique_object)
    except (KeyError, OSError, zipfile.BadZipFile, UnicodeError) as exc:
        raise ValueError("invalid stage archive NPZ") from exc
    try:
        if sum(len(step["stages"]) for step in archive["steps"]) > MAX_STAGES:
            raise ValueError(
                "full-run stage archive is unsupported; typed arrays required"
            )
    except (TypeError, KeyError) as exc:
        raise ValueError("invalid stage archive") from exc
    recorder = decode_stage_archive(archive)
    bundle = load_solution(
        path.with_name(path.stem + ".base.json"),
        expected_manifest_sha256=envelope["base_manifest_sha256"],
        expected_model_sha256=expected_model_sha256,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_assets=expected_assets,
    )
    _check_plan(bundle, recorder)
    return EvidencedSolution(bundle, archive, _hash(_bytes(archive)))


def verify_evidenced_forward(
    evidence: EvidencedSolution, tolerances: Mapping[str, tuple[float, float]]
) -> ReplayComparison:
    """Run actual supported MuJoCo replay and compare every accepted stage exactly."""
    from .solution_bundle_mujoco_replay import run_mujoco_ideal_with_trace

    if _hash(_bytes(evidence.archive)) != evidence.archive_sha256:
        raise ValueError("stage archive hash mismatch")
    _check_plan(evidence.bundle, decode_stage_archive(evidence.archive))
    playback, recorder = run_mujoco_ideal_with_trace(evidence.bundle)
    result = verify_forward_resimulation(
        evidence.bundle,
        lambda _: playback,
        tolerances,
        runtime_sha256=evidence.bundle.manifest["integrator"]["runtime_sha256"],
    )
    equal = compare_stage_archives(evidence.archive, encode_stage_archive(recorder))
    return ReplayComparison(
        result.passed and equal,
        result.max_absolute_error,
        result.independently_verified,
        result.runtime_identity_is_caller_asserted,
    )
