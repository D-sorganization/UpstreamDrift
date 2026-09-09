"""Fit several models to one take and rank them (#9731).

Every registered model is fitted with the same continuous solver to the same
reconstruction; the report says how well each explains the landmarks it
claims (RMS per landmark), how much it rejected, how fast its joints had to
move, and a score that penalises degrees of freedom so a model that fits
only because it has more knobs does not win by default. No claim of
biomechanical truth: this ranks explanatory power on this data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require

from ...provenance import write_stamped
from src.shared.python.logging_pkg.logging_config import get_logger

from .fit import FitOptions
from .registry import get_model, model_names
from .session import MODEL_DIR, fit_session_model

logger = get_logger(__name__)
COMPARISON_FILE = "comparison.json"


class ModelScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    dof: int
    landmarks: int
    frames: int
    rms_mm: float
    rejected: int
    peak_velocity_rad_s: float
    velocity_violations: int
    score: float  # lower is better: log RMS + DOF penalty per observed value


class ModelComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "model-comparison/1.0.0"
    session: str
    ranking: tuple[ModelScore, ...]  # best first

    def markdown(self) -> str:
        lines = [
            "| model | DOF | landmarks | RMS mm | rejected | peak rad/s | violations | score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for s in self.ranking:
            lines.append(
                f"| {s.model} | {s.dof} | {s.landmarks} | {s.rms_mm:.1f} | {s.rejected} | "
                f"{s.peak_velocity_rad_s:.1f} | {s.velocity_violations} | {s.score:.3f} |"
            )
        return "\n".join(lines) + "\n"


def dof_penalised_score(rms_m: float, dof: int, frames: int, landmarks: int) -> float:
    """``ln(rms) + dof / (frames * landmarks * 3)``: an AIC-shaped trade-off.

    Precondition: positive finite rms and counts (callers floor an exact fit
    at one micrometre). The penalty is per observed coordinate, so it fades
    with more data.
    """
    require(
        bool(np.isfinite(rms_m)) and rms_m > 0 and frames > 0 and landmarks > 0,
        "positive finite rms and positive counts",
        (rms_m, frames, landmarks),
    )
    require(dof >= 0, "positive counts", dof)
    return float(np.log(rms_m) + dof / (frames * landmarks * 3))


def compare_models(
    session_dir: Path,
    names: Sequence[str] | None = None,
    *,
    options: FitOptions | None = None,
    fit_lengths: bool = False,
) -> ModelComparison:
    """Fit each named model (default: all) and write ``model/comparison.{json,md}``.

    Each fit's own files land in ``model/<name>/`` so nothing overwrites the
    default model's ``model/`` outputs. Postcondition: ranking is sorted by
    score, best first.
    """
    names = tuple(names or model_names())
    scores = []
    for name in names:
        registered = get_model(name)
        opts = options or FitOptions()
        if fit_lengths:
            opts = FitOptions(
                **{**vars(opts), "fit_lengths": registered.learnable_lengths}
            )
        fit, _ = fit_session_model(
            session_dir,
            registered.spec,
            registered.landmark_map,
            options=opts,
            out_subdir=name,
        )
        landmarks = int(np.isfinite(fit.residual_m).any(axis=0).sum())
        peak = max(fit.peak_velocity_rad_s.values(), default=0.0)
        scores.append(
            ModelScore(
                model=name,
                dof=len(fit.dof_names),
                landmarks=landmarks,
                frames=int(fit.q.shape[0]),
                rms_mm=1000 * fit.rms_m,
                rejected=len(fit.rejected),
                peak_velocity_rad_s=float(peak),
                velocity_violations=fit.velocity_violations,
                score=(
                    dof_penalised_score(
                        max(fit.rms_m, 1e-6),
                        len(fit.dof_names),
                        fit.q.shape[0],
                        landmarks,
                    )
                    if landmarks and np.isfinite(fit.rms_m)
                    else float("inf")  # nothing fitted: ranks last
                ),
            )
        )
        logger.info("compare-models %s: rms %.1f mm", name, 1000 * fit.rms_m)
    ranking = tuple(sorted(scores, key=lambda s: s.score))
    report = ModelComparison(session=str(session_dir), ranking=ranking)
    out_dir = session_dir / MODEL_DIR
    out_dir.mkdir(exist_ok=True)
    angles = [
        session_dir / MODEL_DIR / name / "joint_angles.json"
        for name in names
        if (session_dir / MODEL_DIR / name / "joint_angles.json").is_file()
    ]
    write_stamped(
        out_dir / COMPARISON_FILE,
        report.model_dump(mode="json"),
        schema_version=report.schema_version,
        module=__name__,
        inputs=angles,
        parameters={"models": list(names), "fit_lengths": fit_lengths},
        derived_from=angles,
        base=session_dir,
    )
    (out_dir / "comparison.md").write_text(report.markdown(), encoding="utf-8")
    return report


def load_comparison(session_dir: Path) -> dict[str, Any] | None:
    path = session_dir / MODEL_DIR / COMPARISON_FILE
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None
