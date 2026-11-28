"""SRP-8 multi-camera and phase-still GPU multiplexing tests (#15735)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tools.native_viewer_export.frame_times import write_frame_times
from src.tools.swing_report_package import catalog, composite, runner
import src.tools.swing_report_package.gpu_multiplexer as gm
import src.tools.swing_report_package.manifest_verify as mv
from src.tools.swing_report_package.phases import KeyPhase

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
PHASES = (KeyPhase("top", "Top", 1, 0.1), KeyPhase("impact", "Impact", 2, 0.2))
CFG = composite.load_composite_settings().scaled(tile_width_px=64, cell_width_px=64)


def _jobs(*items: str) -> Any:
    """Plan ``items`` with small test dimensions across default camera views."""
    r = catalog.PackageRequest(
        include=items, default=False, render_overrides={"size": "64x48"}
    )
    return catalog.plan_package(catalog.load_catalog(), r)


def _write_clip(job_key: str, clip: Path) -> list[np.ndarray]:
    """Write a deterministic 4-frame clip and its frame-times sidecar."""
    imageio = pytest.importorskip("imageio.v2")
    pal = (0, 128, 255)
    base = int(hashlib.sha256(job_key.encode("utf-8")).hexdigest()[:4], 16)
    frames = [np.full((48, 64, 3), pal[(base + i) % 3], np.uint8) for i in range(4)]
    imageio.mimwrite(str(clip), frames, fps=10, macro_block_size=8)
    write_frame_times({"x": clip}, (0.0, 0.1, 0.2, 0.3))
    return frames


def _run_dag(jobs: Any, root: Path, calls: list[Any], bad: tuple[str, ...] = ()) -> Any:
    """Execute ``jobs`` via ``execute_multiplex_dag`` with a test sweep runner."""

    def sweep(grp: Any, clips: dict[str, Path], phases: Any, tmp: Path) -> Any:
        """Execute one fused sweep pass across all targets in ``grp``."""
        assert tmp.is_dir()
        calls.append((grp.group_id, tuple(sorted(clips))))
        errors, mem = {}, {}
        ps = phases if grp.extract_phase_stills else ()
        for t in grp.targets:
            if t.job_key in bad:
                errors[t.job_key] = f"overlay failed: {t.job_key}"
            elif t.job_key in clips:
                frames = _write_clip(t.job_key, clips[t.job_key])
                mem.update({(t.job_key, p.id): frames[p.index] for p in ps})
        return {"errors": errors, "stills": mem}

    return gm.execute_multiplex_dag(
        jobs,
        root,
        run_id="srp8",
        sweep_runner=sweep,
        phases_of=lambda _s: PHASES,
        composite=CFG,
    )


def test_plan_multiplex_groups_fuses_sweeps_without_mutating_catalog() -> None:
    """Planner fuses multi-camera + overlay sweeps without mutating SRP-4."""
    cat = catalog.load_catalog()
    snap = json.dumps(dict(cat.render_defaults), sort_keys=True)
    jobs = catalog.plan_package(
        cat, catalog.PackageRequest(render_overrides={"size": "64x48"})
    )
    keys, groups = [j.key for j in jobs], gm.plan_multiplex_groups(jobs, run_id="r")
    assert len(groups) == 3 and len(groups) < sum(j.kind == "video" for j in jobs)
    assert any(g.extract_phase_stills for g in groups)
    assert any(len(g.views) == 2 and len(g.targets) >= 6 for g in groups)
    assert json.dumps(dict(cat.render_defaults), sort_keys=True) == snap
    assert [j.key for j in jobs] == keys
    for obj, attr in ((groups[0], "case_id"), (groups[0].targets[0], "view")):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attr, "mutated")


def test_execute_multiplex_dag_isolates_failure_and_resumes(tmp_path: Path) -> None:
    """One failed overlay stays isolated; deterministic checkpoints allow resume."""
    jobs, bad = _jobs("case_grid", "midhands_total"), "midhands_total@face_on@0.25x"
    c1: list[Any] = []
    c2: list[Any] = []
    r1 = _run_dag(jobs, tmp_path, c1, (bad,))
    by = {o.key: o.status for o in r1.outcomes}
    assert by[bad] == "failed" and by["midhands_total@down_the_line@0.25x"] == "done"
    assert by["case_grid@face_on@0.25x"] == "done" and r1.checkpoint_path is not None
    ck = json.loads(r1.checkpoint_path.read_text("utf-8"))
    assert ck["items"][bad]["status"] == "failed"
    r2 = _run_dag(jobs, tmp_path, c2)
    assert len(c2) == 1 and c2[0][1] == (bad,) and bad not in r2.skipped_from_checkpoint
    assert "case_grid@face_on@0.25x" in r2.skipped_from_checkpoint
    assert mv.verify_manifest(r2.manifest, tmp_path).ok


def test_multiplex_matches_serial_srp5(tmp_path: Path, monkeypatch: Any) -> None:
    """Outputs and hashes match serial SRP-5 and pass manifest verification."""
    jobs = _jobs("case_grid", "phase_sheet")
    s_dir, m_dir = tmp_path / "serial", tmp_path / "mux"
    s_calls: list[str] = []
    m_calls: list[Any] = []

    def ser(job: Any, clip: Path, tmp_dir: Path) -> None:
        """Render one serial job clip into ``clip``."""
        assert tmp_dir.is_dir() and clip.parent.is_dir()
        s_calls.append(job.key)
        _write_clip(job.key, clip)

    s_man = runner.run_package(
        jobs,
        s_dir,
        swing="driver",
        render=ser,
        phases_of=lambda _s: PHASES,
        composite=CFG,
        min_free_gb=0.0,
        preflight=lambda _j: {},
    )

    def forbid(*_a: Any, **_kw: Any) -> Any:
        """Fail if disk-decoding phase_sheet is called when stills are in memory."""
        raise AssertionError("composite.phase_sheet called despite in-memory stills")

    monkeypatch.setattr(composite, "phase_sheet", forbid)
    rep = _run_dag(jobs, m_dir, m_calls)
    assert len(m_calls) == 3 and len(s_calls) == 6 and rep.in_memory_stills_count == 12
    assert gm.verify_serial_parity(s_dir, m_dir, s_man, rep.manifest).ok
    (m_dir / "case_grid" / "face_on_0p25x" / "case_grid.mp4").write_bytes(b"corrupt")
    broken = gm.verify_serial_parity(s_dir, m_dir, s_man, rep.manifest)
    assert not broken.ok and "parity_hash_mismatch" in {i.code for i in broken.issues}
