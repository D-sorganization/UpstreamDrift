"""Typed, bounded RK4 numerical evidence; not a physical contact certificate."""

from dataclasses import asdict
from io import BytesIO
import json
import zipfile

import numpy as np

from .stage_trace import RunPlan, SegmentPlan, Stage, StageRecorder, TracedStep

LIMIT = 256 * 1024 * 1024
MAX_STAGES = 100_000
KEYS = {
    "plan",
    "stage_ids",
    "stage_time",
    "q",
    "v",
    "tau",
    "drive",
    "external",
    "segment_ids",
    "segment_clock",
    "offsets",
    "step_sizes",
    "boundaries",
}
SEMANTICS = "ideal_direct_generalized_no_saturation"


def encode_numeric(recorder: StageRecorder) -> bytes:
    """Encode only a complete ideal-drive trace, preserving numeric float64 values."""
    if (
        not recorder.capture_ideal_drive
        or recorder.plan is None
        or recorder.outcome is None
        or recorder.outcome.status != "complete"
        or recorder._integrity_failure() is not None
    ):
        raise ValueError("complete ideal stage evidence required")
    stages = [s for step in recorder.steps for s in step.stages]
    segments = [s for step in recorder.steps for s in step.segments]
    if not stages or len(stages) > MAX_STAGES:
        raise ValueError("unsupported stage count")
    sizes = [v for segment in segments for v in segment.step_sizes_s]
    offsets = np.r_[0, np.cumsum([len(s.step_sizes_s) for s in segments])].astype(
        np.int64
    )
    plan = json.dumps(asdict(recorder.plan), sort_keys=True, allow_nan=False).encode()
    arrays = {
        "plan": np.frombuffer(plan, dtype=np.uint8),
        "stage_ids": np.asarray(
            [(s.step, s.hold, s.segment, s.substep, s.stage) for s in stages],
            dtype=np.int64,
        ),
        "stage_time": np.asarray([s.time_s for s in stages], dtype=np.float64),
        "segment_ids": np.asarray(
            [
                (s.step, s.hold, s.segment, s.substeps, int(s.force_active))
                for s in segments
            ],
            dtype=np.int64,
        ),
        "segment_clock": np.asarray(
            [(s.start_s, s.duration_s) for s in segments], dtype=np.float64
        ),
        "offsets": offsets,
        "step_sizes": np.asarray(sizes, dtype=np.float64),
        "boundaries": np.asarray(
            [
                [
                    step.pre_projection_q,
                    step.pre_projection_v,
                    step.post_projection_q,
                    step.post_projection_v,
                ]
                for step in recorder.steps
            ],
            dtype=np.float64,
        ),
    }
    for name, attr in (
        ("q", "q"),
        ("v", "v"),
        ("tau", "tau"),
        ("drive", "applied_drive"),
        ("external", "applied_external"),
    ):
        arrays[name] = np.asarray([getattr(s, attr) for s in stages], dtype=np.float64)
    if sum(a.nbytes for a in arrays.values()) > LIMIT:
        raise ValueError("stage arrays exceed size limit")
    out = BytesIO()
    np.savez_compressed(out, **arrays)
    if out.tell() > LIMIT:
        raise ValueError("stage archive exceeds size limit")
    return out.getvalue()


def decode_numeric(raw: bytes) -> StageRecorder:
    """Reject incomplete, oversized, malformed, nonfinite, or inconsistent evidence."""
    if len(raw) > LIMIT:
        raise ValueError("stage archive exceeds size limit")
    try:
        with zipfile.ZipFile(BytesIO(raw)) as z:
            entries = z.infolist()
            if (
                {x.filename for x in entries} != {k + ".npy" for k in KEYS}
                or len(entries) != len(KEYS)
                or sum(x.file_size for x in entries) > LIMIT + len(KEYS) * 4096
            ):
                raise ValueError("invalid stage NPZ inventory or size")
            for entry in entries:
                with z.open(entry) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                            member
                        )
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                            member
                        )
                    else:
                        raise ValueError("unsupported NPY version")
                    expected = np.dtype(
                        np.uint8
                        if entry.filename == "plan.npy"
                        else (
                            np.int64
                            if entry.filename
                            in {"stage_ids.npy", "segment_ids.npy", "offsets.npy"}
                            else np.float64
                        )
                    )
                    count = 1
                    for dim in shape:
                        count *= dim
                        if count * dtype.itemsize > LIMIT:
                            raise ValueError("oversized array")
                    if (
                        dtype != expected
                        or fortran
                        or member.tell() + count * dtype.itemsize != entry.file_size
                    ):
                        raise ValueError("invalid NPY header")
        with np.load(BytesIO(raw), allow_pickle=False) as data:
            a = {k: data[k] for k in KEYS}
        p = json.loads(a["plan"].tobytes())
        if set(p) != {
            "requested_steps",
            "dt_s",
            "substeps",
            "effort_holds",
            "project",
            "coordinate_order",
        }:
            raise ValueError("invalid plan")
        if (
            any(
                type(p[k]) is not int
                for k in ("requested_steps", "substeps", "effort_holds")
            )
            or type(p["project"]) is not bool
        ):
            raise ValueError("invalid plan types")
        plan = RunPlan(**{**p, "coordinate_order": tuple(p["coordinate_order"])})
        n, nv, ns = (
            len(a["stage_ids"]),
            len(plan.coordinate_order),
            len(a["segment_ids"]),
        )
        if (
            not 0 < n <= MAX_STAGES
            or not 0 < nv <= 1000
            or not 0 < plan.requested_steps <= n
            or a["stage_ids"].shape != (n, 5)
            or a["stage_time"].shape != (n,)
            or a["segment_ids"].shape != (ns, 5)
            or a["segment_clock"].shape != (ns, 2)
            or a["offsets"].shape != (ns + 1,)
            or a["boundaries"].shape != (plan.requested_steps, 4, nv)
            or any(
                a[k].shape != (n, nv) for k in ("q", "v", "tau", "drive", "external")
            )
        ):
            raise ValueError("invalid stage array dimensions")
        offsets = a["offsets"]
        if (
            offsets[0] != 0
            or offsets[-1] != len(a["step_sizes"])
            or np.any(np.diff(offsets) <= 0)
            or np.any((a["segment_ids"][:, 4] < 0) | (a["segment_ids"][:, 4] > 1))
        ):
            raise ValueError("invalid segment offsets")
        if any(
            not np.isfinite(a[k]).all()
            for k in (
                "stage_time",
                "segment_clock",
                "step_sizes",
                "boundaries",
                "q",
                "v",
                "tau",
                "drive",
                "external",
            )
        ):
            raise ValueError("nonfinite stage evidence")
        recorder = StageRecorder(capture_ideal_drive=True)
        recorder.start_run(plan)
        stage_steps = a["stage_ids"][:, 0]
        segment_steps = a["segment_ids"][:, 0]
        if (
            np.any(stage_steps < 0)
            or np.any(stage_steps >= plan.requested_steps)
            or np.any(segment_steps < 0)
            or np.any(segment_steps >= plan.requested_steps)
            or np.any(np.diff(stage_steps) < 0)
            or np.any(np.diff(segment_steps) < 0)
        ):
            raise ValueError("out-of-range or reordered step id")
        stage_breaks = np.searchsorted(stage_steps, np.arange(plan.requested_steps + 1))
        segment_breaks = np.searchsorted(
            segment_steps, np.arange(plan.requested_steps + 1)
        )
        for step_id in range(plan.requested_steps):
            segments = []
            for i in range(segment_breaks[step_id], segment_breaks[step_id + 1]):
                ids = a["segment_ids"][i]
                segments.append(
                    SegmentPlan(
                        int(ids[0]),
                        int(ids[1]),
                        int(ids[2]),
                        float(a["segment_clock"][i, 0]),
                        float(a["segment_clock"][i, 1]),
                        int(ids[3]),
                        bool(ids[4]),
                        tuple(a["step_sizes"][offsets[i] : offsets[i + 1]]),
                    )
                )
            stages = []
            for i in range(stage_breaks[step_id], stage_breaks[step_id + 1]):
                stages.append(
                    Stage(
                        int(a["stage_ids"][i, 0]),
                        int(a["stage_ids"][i, 1]),
                        int(a["stage_ids"][i, 2]),
                        int(a["stage_ids"][i, 3]),
                        int(a["stage_ids"][i, 4]),
                        float(a["stage_time"][i]),
                        a["q"][i],
                        a["v"][i],
                        a["tau"][i],
                        a["drive"][i],
                        a["external"][i],
                        SEMANTICS,
                    )
                )
            recorder.steps.append(
                TracedStep(tuple(stages), tuple(segments), *a["boundaries"][step_id])
            )
        recorder.finish()
        if recorder.outcome is None or recorder.outcome.status != "complete":
            raise ValueError("incomplete or inconsistent stage archive")
        return recorder
    except (
        KeyError,
        TypeError,
        IndexError,
        OverflowError,
        zipfile.BadZipFile,
        UnicodeError,
    ) as exc:
        raise ValueError("invalid numeric stage archive") from exc
