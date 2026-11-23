"""Version 6 research envelope binding typed numerical stages to a v4 base.

The recorded latched force is replayed, not certified as first contact. Ideal
source drive and external force are numerical inputs, not actuator measurements.
Envelope publication is atomic; interrupted writes can leave orphan base/NPZ
files and directory crash durability is not promised. No implicit migration.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Mapping
import hashlib
import json
import os
import tempfile

import numpy as np

from .evidenced_solution import _check_plan
from .solution_bundle_prototype import (
    ReplayComparison,
    SolutionBundle,
    _unique_object,
    load_solution,
    save_solution,
    verify_forward_resimulation,
)
from .stage_numeric_archive import LIMIT, decode_numeric, encode_numeric
from .stage_trace import StageRecorder

SCHEMA = "research-evidenced-solution/v6"


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class NumericEvidence:
    bundle: SolutionBundle
    stage_npz: bytes
    stage_sha256: str


def save_numeric_solution(
    path: Path, bundle: SolutionBundle, recorder: StageRecorder
) -> str:
    path = Path(path)
    if path.suffix != ".json" or path.exists():
        raise ValueError("a new JSON envelope path is required")
    _check_plan(bundle, recorder)
    stage_npz = encode_numeric(recorder)
    base_sha = save_solution(path.with_name(path.stem + ".base.json"), bundle)
    with path.with_name(path.stem + ".stages.npz").open("xb") as stream:
        stream.write(stage_npz)
        stream.flush()
        os.fsync(stream.fileno())
    envelope = {
        "schema": SCHEMA,
        "base_manifest_sha256": base_sha,
        "stage_archive_sha256": _hash(stage_npz),
    }
    raw = json.dumps(
        envelope, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".evidence-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return _hash(raw)


def load_numeric_solution(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_model_sha256: str,
    expected_runtime_sha256: str,
    expected_assets: Mapping[str, str] | None = None,
) -> NumericEvidence:
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError("evidence manifest exceeds size limit")
    raw = path.read_bytes()
    if _hash(raw) != expected_manifest_sha256:
        raise ValueError("evidence manifest hash mismatch")
    envelope = json.loads(raw, object_pairs_hook=_unique_object)
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"schema", "base_manifest_sha256", "stage_archive_sha256"}
        or envelope["schema"] != SCHEMA
    ):
        raise ValueError("unsupported or incomplete schema; re-recording required")
    stage_path = path.with_name(path.stem + ".stages.npz")
    if stage_path.stat().st_size > LIMIT:
        raise ValueError("stage archive exceeds size limit")
    stage_npz = stage_path.read_bytes()
    if _hash(stage_npz) != envelope["stage_archive_sha256"]:
        raise ValueError("stage archive hash mismatch")
    recorder = decode_numeric(stage_npz)
    bundle = load_solution(
        path.with_name(path.stem + ".base.json"),
        expected_manifest_sha256=envelope["base_manifest_sha256"],
        expected_model_sha256=expected_model_sha256,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_assets=expected_assets,
    )
    _check_plan(bundle, recorder)
    return NumericEvidence(bundle, stage_npz, envelope["stage_archive_sha256"])


def verify_numeric_forward(
    evidence: NumericEvidence, tolerances: Mapping[str, tuple[float, float]]
) -> ReplayComparison:
    """Reintegrate actual supported MuJoCo plant, compare every numerical stage."""
    from .solution_bundle_mujoco_replay import run_mujoco_ideal_with_trace

    if _hash(evidence.stage_npz) != evidence.stage_sha256:
        raise ValueError("stage archive hash mismatch")
    _check_plan(evidence.bundle, decode_numeric(evidence.stage_npz))
    playback, trace = run_mujoco_ideal_with_trace(evidence.bundle)
    result = verify_forward_resimulation(
        evidence.bundle,
        lambda _: playback,
        tolerances,
        runtime_sha256=evidence.bundle.manifest["integrator"]["runtime_sha256"],
    )
    rerun_npz = encode_numeric(trace)
    with (
        np.load(BytesIO(evidence.stage_npz), allow_pickle=False) as expected,
        np.load(BytesIO(rerun_npz), allow_pickle=False) as actual,
    ):
        same = expected.files == actual.files and all(
            np.array_equal(expected[k], actual[k]) for k in expected.files
        )
    return ReplayComparison(
        result.passed and same,
        result.max_absolute_error,
        result.independently_verified,
        result.runtime_identity_is_caller_asserted,
    )
