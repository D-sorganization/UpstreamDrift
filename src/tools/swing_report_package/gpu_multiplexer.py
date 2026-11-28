"""SRP-8 single-pass multi-camera and phase-still GPU multiplexer (#15735)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from src.shared.python.video_timing.frame_schedule import speed_suffix
from src.tools.native_viewer_export.frame_times import frame_times_path
from src.tools.swing_report_package import composite as comp
from src.tools.swing_report_package import manifest_verify as mv
from src.tools.swing_report_package import runner
from src.tools.swing_report_package.catalog import Job

CHECKPOINT_FILENAME = "multiplex_checkpoint.json"
_ERR = (subprocess.SubprocessError, OSError, ValueError, RuntimeError)
_RKEYS = ("run_id", "model_definition_hash", "subject_profile_hash")
SweepRunner = Callable[..., Mapping[str, Any] | None]


@dataclass(frozen=True)
class MultiplexTarget:
    """One view and overlay target inside a :class:`MultiplexRenderGroup`."""

    job_key: str
    item: str
    view: str
    speed: float
    overlays: Mapping[str, Any] = field(default_factory=dict)
    phase_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiplexRenderGroup:
    """Single-pass multi-camera, multi-overlay, and phase-still batch."""

    group_id: str
    run_id: str
    case_id: str
    speed: float
    engine: str = "mujoco"
    views: tuple[str, ...] = ()
    targets: tuple[MultiplexTarget, ...] = ()
    composite_keys: tuple[str, ...] = ()
    extract_phase_stills: bool = False


@dataclass(frozen=True)
class MultiplexItemOutcome:
    """Execution outcome of one video target or composite job."""

    key: str
    status: str
    outputs: tuple[str, ...] = ()
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    error: str | None = None
    resumed: bool = False


@dataclass(frozen=True)
class MultiplexExecutionReport:
    """Execution report returned by :func:`execute_multiplex_dag`."""

    run_id: str
    groups: tuple[MultiplexRenderGroup, ...] = ()
    outcomes: tuple[MultiplexItemOutcome, ...] = ()
    manifest: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    skipped_from_checkpoint: tuple[str, ...] = ()
    in_memory_stills_count: int = 0


def _sha(src: Path | str) -> str:
    """Return hex SHA-256 digest of file ``src`` or UTF-8 string ``src``."""
    raw = src.read_bytes() if isinstance(src, Path) else src.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically write ``data`` as deterministic sorted JSON to ``path``."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _clip(root: Path, job: Job, swing: str) -> Path:
    """Return deterministic clip path matching SRP-5 ``runner`` layout."""
    eng, sfx = str(job.args.get("engines", "mujoco")), speed_suffix(job.speed)
    return runner.job_dir(root, job) / f"{swing}_{eng}_{job.view}{sfx}.mp4"


def plan_multiplex_groups(
    jobs: Sequence[Job], *, run_id: str = "default", phase_ids: Sequence[str] = ()
) -> tuple[MultiplexRenderGroup, ...]:
    """Fuse SRP-4 ``jobs`` into single-pass :class:`MultiplexRenderGroup` batches."""
    sheets = {n for j in jobs if j.kind == "phase_sheet" for n in j.needs}
    deps: dict[str, list[str]] = {}
    buckets: dict[tuple[str, str, float], list[Job]] = {}
    for j in jobs:
        if j.kind != "video":
            for n in j.needs:
                deps.setdefault(n, []).append(j.key)
            continue
        cid = str(j.args.get("force_part") or j.args.get("ztcf_zvcf_view") or j.item)
        eng = str(j.args.get("engines", "mujoco"))
        buckets.setdefault((cid, eng, j.speed), []).append(j)
    ptup, groups = tuple(phase_ids), []
    for (cid, eng, spd), gj in buckets.items():
        tgts = []
        for j in gj:
            p, ov = ptup if j.key in sheets else (), dict(j.args)
            tgts.append(MultiplexTarget(j.key, j.item, j.view, j.speed, ov, p))
        views = tuple(dict.fromkeys(j.view for j in gj))
        ck = tuple(dict.fromkeys(k for j in gj for k in deps.get(j.key, ())))
        has, gid = any(j.key in sheets for j in gj), f"{run_id}:{cid}:{eng}:{spd:g}x"
        g = MultiplexRenderGroup(
            gid, run_id, cid, spd, eng, views, tuple(tgts), ck, has
        )
        groups.append(g)
    return tuple(groups)


def _cached(job: Job, ctx: Any) -> Any:
    """Return a resumed outcome when checkpointed outputs verify on disk."""
    root, _, ckpt, _, _, swing, _ = ctx
    e = ckpt.get("items", {}).get(job.key)
    if not isinstance(e, Mapping) or e.get("status") != "done":
        return None
    h, outs = e.get("artifact_hashes", {}), tuple(e.get("outputs", ()))
    base = root.resolve()
    if not outs or not isinstance(h, Mapping):
        return None
    for rel, exp in h.items():
        c = (base / rel).resolve()
        if not c.is_relative_to(base) or not c.is_file() or _sha(c) != exp:
            return None
    clip = _clip(root, job, swing) if job.kind == "video" else None
    if clip and (not clip.is_file() or not frame_times_path(clip).is_file()):
        return None
    return MultiplexItemOutcome(job.key, "done", outs, dict(h), resumed=True)


def _save(ctx: Any, key: str, st: str, paths: Sequence[Path], err: str | None) -> None:
    """Hash ``paths``, record outcome, and update deterministic JSON checkpoint."""
    root, ckpt_path, ckpt, outcomes, by_key, _, _ = ctx
    h = {p.relative_to(root).as_posix(): _sha(p) for p in paths}
    for p in paths:
        if by_key[key].kind == "video" and (sc := frame_times_path(p)).is_file():
            h[sc.relative_to(root).as_posix()] = _sha(sc)
    rels = tuple(p.relative_to(root).as_posix() for p in paths)
    outcomes[key] = MultiplexItemOutcome(key, st, rels, h, err)
    row: dict[str, Any] = {"status": st, "outputs": list(rels), "artifact_hashes": h}
    if err is not None:
        row["error"] = err
    ckpt["items"][key] = row
    _write_json(ckpt_path, ckpt)


def _sheet_mem(keys: Any, phases: Sequence[Any], out: Path, stills: Any, s: Any) -> Any:
    """Write ``phase_sheet`` from in-memory stills when all frames are cached."""
    paths, rows = [], []
    for p in phases:
        if any((keys.get(c) or "", p.id) not in stills for c in s.case_order):
            return None
        cells = []
        for c in s.case_order:
            fr = np.asarray(stills[(keys[c], p.id)], dtype=np.uint8)
            sp = out.with_name(f"{c}_{p.id}.png")
            Image.fromarray(fr).save(sp)
            paths.append(sp)
            cells.append(comp._resized(fr, s.cell_width_px))
        rows.append(np.hstack([comp._row_label(cells[0].shape[0], p.label, s), *cells]))
    corner = comp._info_tile((s.banner_height_px, s.row_label_width_px), (), s)
    bans = [comp.case_banner(s.cell_width_px, c, s) for c in s.case_order]
    Image.fromarray(np.vstack([np.hstack([corner, *bans]), *rows])).save(out)
    return [out, *paths]


def _run_videos(grps: Any, ph: Any, fn: SweepRunner, ctx: Any, mem: Any) -> Any:
    """Execute video sweep groups with checkpoint reuse and error isolation."""
    root, _, _, outcomes, by_key, swing, _ = ctx
    clips: dict[str, Path] = {}
    for grp in grps:
        pending: dict[str, Path] = {}
        for t in grp.targets:
            job, clip = by_key[t.job_key], _clip(root, by_key[t.job_key], swing)
            if (c := _cached(job, ctx)) is not None:
                outcomes[job.key], clips[job.key] = c, clip
            else:
                clip.parent.mkdir(parents=True, exist_ok=True)
                pending[job.key] = clip
        if not pending:
            continue
        errs: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                res = fn(grp, pending, ph, Path(tmp))
            except _ERR as exc:
                res, errs = None, {k: str(exc) for k in pending}
        if isinstance(res, Mapping):
            errs.update({str(k): str(v) for k, v in (res.get("errors") or {}).items()})
            for k, fr in (res.get("stills") or {}).items():
                if isinstance(k, tuple) and len(k) == 2:
                    mem[(str(k[0]), str(k[1]))] = np.asarray(fr, dtype=np.uint8)
        for key, clip in pending.items():
            err = errs.get(key)
            if not err and not (clip.is_file() and frame_times_path(clip).is_file()):
                err = f"{key}: missing clip or frame_times sidecar"
            _save(ctx, key, "failed" if err else "done", () if err else [clip], err)
            if not err:
                clips[key] = clip
    return clips


def _run_composites(jobs: Any, clips: Any, ph: Any, mem: Any, ctx: Any) -> Any:
    """Execute grid and phase_sheet composites with checkpointing."""
    root, _, _, outcomes, by_key, _, cfg = ctx
    for job in (j for j in jobs if j.kind != "video"):
        if (c := _cached(job, ctx)) is not None:
            outcomes[job.key] = c
            continue
        if any(n not in clips for n in job.needs):
            _save(ctx, job.key, "blocked", (), None)
            continue
        try:
            cases = {str(by_key[n].args["force_part"]): clips[n] for n in job.needs}
            folder = runner.job_dir(root, job)
            folder.mkdir(parents=True, exist_ok=True)
            if job.kind == "grid":
                out = folder / f"{job.item}.mp4"
                lbl = f"{job.view.replace('_', ' ')}, {job.speed:g}x speed"
                comp.compose_grid_video(
                    cases, out, cfg, info_lines=(job.title, lbl), phases=ph
                )
                outs = [out]
            else:
                out = folder / f"{job.item}.png"
                keys = {str(by_key[n].args["force_part"]): n for n in job.needs}
                st = _sheet_mem(keys, ph, out, mem, cfg)
                outs = st or [out, *comp.phase_sheet(cases, ph, out, cfg)]
            _save(ctx, job.key, "done", outs, None)
        except _ERR as exc:
            _save(ctx, job.key, "failed", (), str(exc))


def execute_multiplex_dag(
    jobs: Sequence[Job],
    root: Path,
    *,
    swing: str = "driver",
    run_id: str = "default",
    sweep_runner: SweepRunner | None = None,
    phases_of: Any = None,
    composite: Any = None,
) -> MultiplexExecutionReport:
    """Execute ``jobs`` as fused sweep groups with checkpoints and parity."""
    if sweep_runner is None:
        raise ValueError("sweep_runner is required")
    root.mkdir(parents=True, exist_ok=True)
    phases = tuple(phases_of(swing)) if phases_of is not None else ()
    pids = [p.id for p in phases]
    groups = plan_multiplex_groups(jobs, run_id=run_id, phase_ids=pids)
    cp = root / CHECKPOINT_FILENAME
    ckpt = json.loads(cp.read_text("utf-8")) if cp.is_file() else {}
    if ckpt.get("run_id") != run_id or ckpt.get("swing") != swing:
        ckpt = {"schema_version": 1, "run_id": run_id, "swing": swing, "items": {}}
    by_key = {j.key: j for j in jobs}
    cfg = composite or comp.load_composite_settings()
    outcomes: dict[str, MultiplexItemOutcome] = {}
    stills: dict[tuple[str, str], np.ndarray] = {}
    ctx = (root, cp, ckpt, outcomes, by_key, swing, cfg)
    clips = _run_videos(groups, phases, sweep_runner, ctx, stills)
    _run_composites(jobs, clips, phases, stills, ctx)
    arts = {r: o.artifact_hashes[r] for o in outcomes.values() for r in o.outputs}
    entries = []
    for j in jobs:
        o = outcomes[j.key]
        d: dict[str, Any] = {"key": j.key, "item": j.item, "kind": j.kind}
        d.update(title=j.title, status=o.status, outputs=list(o.outputs))
        if o.error:
            d["error"] = o.error
        entries.append(d)
    rid = run_id if len(run_id.removeprefix("sha256:")) == 64 else _sha(run_id)
    manifest = {
        "manifest_schema_version": mv.SCHEMA_VERSION,
        "kind": "dev",
        "swing": swing,
        "run": dict.fromkeys((*_RKEYS, "match_artifact_hash"), rid),
        "tensor_hash": _sha(f"t:{run_id}:{swing}"),
        "settings_hash": _sha(f"s:{run_id}:{len(jobs)}"),
        "exporter_code_hash": _sha("srp8"),
        "git": {"commit": "0" * 40, "dirty": False},
        "jobs": entries,
        "phases": [p.to_dict() for p in phases],
        "artifacts": [{"path": p, "sha256": arts[p]} for p in sorted(arts)],
    }
    _write_json(root / mv.MANIFEST, manifest)
    ordered = tuple(outcomes[j.key] for j in jobs)
    sk = tuple(o.key for o in ordered if o.resumed)
    n = len(stills)
    return MultiplexExecutionReport(run_id, groups, ordered, manifest, cp, sk, n)


def verify_serial_parity(
    serial_root: Path,
    multiplex_root: Path,
    serial_manifest: Any = None,
    multiplex_manifest: Any = None,
) -> mv.Verification:
    """Verify SRP-8 outputs and hashes match SRP-5 serial execution."""
    sp, mp = serial_root / mv.MANIFEST, multiplex_root / mv.MANIFEST
    s_doc = serial_manifest or json.loads(sp.read_text("utf-8"))
    m_doc = multiplex_manifest or json.loads(mp.read_text("utf-8"))
    issues = list(mv.verify_manifest(m_doc, multiplex_root).issues)
    sj = {j["key"]: j for j in s_doc.get("jobs", ())}
    mj = {j["key"]: j for j in m_doc.get("jobs", ())}
    if set(sj) != set(mj):
        issues.append(mv.Issue("parity_jobs_mismatch", "jobs", "job keys differ"))
    for key, se in sj.items():
        if (me := mj.get(key)) is None:
            continue
        if se.get("status") != me.get("status"):
            issues.append(mv.Issue("parity_status_mismatch", key, "status differs"))
        so = [Path(p).as_posix() for p in se.get("outputs", ())]
        if so != [Path(p).as_posix() for p in me.get("outputs", ())]:
            issues.append(mv.Issue("parity_outputs_mismatch", key, "outputs differ"))
            continue
        for rel in so:
            sf, mf = (serial_root / rel).resolve(), (multiplex_root / rel).resolve()
            ss, ms = frame_times_path(sf), frame_times_path(mf)
            if not sf.is_file() or not mf.is_file():
                issues.append(mv.Issue("missing_artifact", rel, "missing output"))
            elif _sha(sf) != _sha(mf):
                issues.append(mv.Issue("parity_hash_mismatch", rel, "hash differs"))
            elif ss.is_file() and (not ms.is_file() or _sha(ss) != _sha(ms)):
                issues.append(mv.Issue("parity_hash_mismatch", rel, "sidecar differs"))
    return mv.Verification(tuple(issues))
