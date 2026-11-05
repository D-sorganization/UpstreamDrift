"""Reusable swing-variant artefact with run identity (SHOW-C6, #14562).

A solved shot-shape preset is saved as a JSON "swing variant": the preset, its
target, the solver result (knots, impact, deviation from the fit, binding
joints) and the :class:`~src.sim_run.RunManifest` that identifies the run
(#14436: model and revision, parameters, input data, seed, engine, commit).
Loading re-validates the manifest, so a variant with a missing or mutable
identity is rejected rather than trusted.

Design by Contract:
    Preconditions: the manifest's ``parameters["preset"]`` names the preset.
    Postconditions: ``save`` writes a file that ``load_variant`` accepts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require
from src.shared.python.impact_optimization.shot_presets import ShotPreset
from src.shared.python.impact_optimization.trajectory_optimiser import ImpactSolution
from src.sim_run import RunManifest, validate_run_manifest

SCHEMA = "swing-variant/1"


@dataclass(frozen=True, eq=False)
class SwingVariant:
    """A solved preset plus the identity of the run that produced it."""

    shot: ShotPreset
    solution: ImpactSolution
    manifest: RunManifest

    def to_dict(self) -> dict[str, Any]:
        t, sol = self.shot.target, self.solution
        return {
            "schema": SCHEMA,
            "preset": self.shot.name,
            "assumptions": list(self.shot.assumptions),
            "target": {
                "face_deg": t.face_deg,
                "path_deg": t.path_deg,
                "aoa_deg": t.aoa_deg,
                "speed_mph": t.speed_mph,
            },
            "solution": {
                "knots": sol.knots.tolist(),
                "deviation": sol.deviation,
                "deviation_from_fit": sol.deviation_from_fit,
                "binding_joints": list(sol.binding_joints),
                "max_utilisation": sol.max_utilisation,
                "tolerances_met": sol.tolerances_met,
                "face_to_path_deg": sol.face_to_path_deg,
            },
            "run_manifest": self.manifest.model_dump(mode="json"),
        }

    def save(self, path: Path) -> None:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=True)
        Path(path).write_text(text + "\n", encoding="utf-8")


def build_variant(
    shot: ShotPreset, solution: ImpactSolution, manifest: RunManifest
) -> SwingVariant:
    """Pair a solved preset with its run identity."""
    named = manifest.parameters.get("preset")
    require(
        named == shot.name,
        "run manifest parameters must name the preset it was run for",
        (named, shot.name),
    )
    return SwingVariant(shot, solution, manifest)


def load_variant(path: Path) -> dict[str, Any]:
    """Read a saved variant, re-validating its run identity."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    require(data.get("schema") == SCHEMA, "unknown swing-variant schema")
    validate_run_manifest(data["run_manifest"])
    return data
