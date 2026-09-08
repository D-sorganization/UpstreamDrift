"""Which joints can be trusted, from the evidence a session already holds (#9662).

Every ingested observation set (``observations/``, ``observations_<name>/``)
contributes per-joint coverage, mean confidence and jitter through the same
:mod:`.compare` metrics; a reconstruction's ``clean_report.json`` contributes
how often each joint was rejected as an outlier. The report ranks joints by
a plain score so the operator can decide which ones to keep in the fit
(``rig reconstruct --exclude-joints``). It states evidence, not accuracy:
without ground truth, "reliable" means consistently detected, confident and
steady across views and detectors.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.core.contracts import require

from .compare import DEFAULT_MIN_CONFIDENCE, SHARED_JOINTS, joint_metrics, load_series

RELIABILITY_FILE = "reliability.json"
JITTER_REFERENCE = 0.02  # box heights per frame at which the jitter term is 0.5


class JointEvidence(BaseModel):
    """One joint across every (view, detector) set that reports it."""

    model_config = ConfigDict(frozen=True)

    joint: str
    sets: int
    coverage: float | None
    mean_confidence: float | None
    jitter: float | None
    rejection_rate: float | None
    score: float | None

    @property
    def grade(self) -> str:
        if self.score is None:
            return "unknown"
        if self.score >= 0.75:
            return "reliable"
        if self.score >= 0.5:
            return "usable"
        return "weak"


class ReliabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "joint-reliability/1.0.0"
    session: str
    observation_sets: tuple[str, ...]
    clean_report: bool
    joints: tuple[JointEvidence, ...]
    recommended_exclusions: tuple[str, ...] = Field(default_factory=tuple)

    def markdown(self) -> str:
        lines = [
            "| joint | grade | coverage | confidence | jitter | rejected | score |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for j in self.joints:
            cells = [
                j.joint,
                j.grade,
                *(_fmt(v) for v in (j.coverage, j.mean_confidence, j.jitter)),
                _fmt(j.rejection_rate),
                _fmt(j.score),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def score(
    coverage: float | None,
    confidence: float | None,
    jitter: float | None,
    rejection_rate: float | None,
) -> float | None:
    """Mean of the available terms, each in [0, 1]; None without evidence.

    A joint never confidently seen scores 0 whatever the other terms say.
    """
    if coverage is not None and coverage <= 0.0:
        return 0.0
    terms = []
    if coverage is not None:
        terms.append(coverage)
    if confidence is not None:
        terms.append(confidence)
    if jitter is not None:
        terms.append(1.0 / (1.0 + jitter / JITTER_REFERENCE))
    if rejection_rate is not None:
        terms.append(1.0 - rejection_rate)
    return float(np.mean(terms)) if terms else None


def observation_sets(session_dir: Path) -> dict[str, list[Path]]:
    """``{set name: [view files]}`` for every ``observations*`` directory."""
    out: dict[str, list[Path]] = {}
    for d in sorted(session_dir.glob("observations*")):
        if not d.is_dir():
            continue
        files = [
            p
            for p in sorted(d.glob("*.json"))
            if p.name != "observations.json" and p.name != "timing_report.json"
        ]
        if files:
            out[d.name] = files
    return out


def rejection_rates(clean_report: Path) -> dict[str, float] | None:
    """Per-joint rejected / frames over all views, or None without a report."""
    if not clean_report.is_file():
        return None
    reports = json.loads(clean_report.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    frames = 0
    for view in reports.values():
        frames += int(view.get("frames", 0))
        for rejection in view.get("rejected", []):
            name = rejection.get("joint")
            if name:
                counts[name] = counts.get(name, 0) + 1
    if frames == 0:
        return None
    return {k: v / frames for k, v in counts.items()}


def reliability_report(
    session_dir: Path,
    *,
    joints: Sequence[str] = SHARED_JOINTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    weak_threshold: float = 0.5,
) -> ReliabilityReport:
    """Aggregate every observation set and the clean report; rank the joints."""
    require(session_dir.is_dir(), "session must be a directory", str(session_dir))
    sets = observation_sets(session_dir)
    require(bool(sets), "session has no observation sets", str(session_dir))
    rejected = rejection_rates(session_dir / "reconstruct" / "clean_report.json")
    per_joint: dict[str, list[Any]] = {j: [] for j in joints}
    for files in sets.values():
        for path in files:
            series = load_series(path)
            for name in joints:
                if name in series.names:
                    per_joint[name].append(
                        joint_metrics(series, name, min_confidence=min_confidence)
                    )
    evidence = []
    for name in joints:
        rows = per_joint[name]
        cov = _mean([m.coverage for m in rows])
        conf = _mean([m.mean_confidence for m in rows])
        jit = _mean([m.jitter for m in rows])
        rej = None if rejected is None else rejected.get(name, 0.0)
        evidence.append(
            JointEvidence(
                joint=name,
                sets=len(rows),
                coverage=cov,
                mean_confidence=conf,
                jitter=jit,
                rejection_rate=rej,
                score=score(cov, conf, jit, rej),
            )
        )
    evidence.sort(key=lambda e: -1.0 if e.score is None else e.score, reverse=True)
    weak = tuple(
        e.joint for e in evidence if e.score is not None and e.score < weak_threshold
    )
    return ReliabilityReport(
        session=str(session_dir),
        observation_sets=tuple(sets),
        clean_report=rejected is not None,
        joints=tuple(evidence),
        recommended_exclusions=weak,
    )


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def write_reliability(report: ReliabilityReport, session_dir: Path) -> Path:
    path = session_dir / RELIABILITY_FILE
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(report.markdown(), encoding="utf-8")
    return path
