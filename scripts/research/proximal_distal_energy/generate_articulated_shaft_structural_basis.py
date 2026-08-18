"""Freeze the first FE bending mode for lean native-engine runtimes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.research.proximal_distal_energy.shaft_beam_reference import (
    BeamReferenceConfig,
    modal_basis,
    model_matrices,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
JSON_PATH = DATA / "articulated_shaft_structural_basis.json"
NPZ_PATH = DATA / "articulated_shaft_structural_basis.npz"
SOURCES = (
    "scripts/research/proximal_distal_energy/generate_articulated_shaft_structural_basis.py",
    "scripts/research/proximal_distal_energy/shaft_beam_reference.py",
    "src/shared/python/physics/flexible_shaft.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    beam = BeamReferenceConfig.publication_default()
    mass, stiffness = model_matrices(beam)
    frequencies, vectors = modal_basis(mass, stiffness, 1)
    shapes = np.concatenate((np.zeros(1), vectors[0::2, 0]))
    slopes = np.concatenate((np.zeros(1), vectors[1::2, 0]))
    tip_scale = float(shapes[-1])
    if abs(tip_scale) <= 1.0e-12:
        raise RuntimeError("finite-element first mode has zero tip scale")
    arrays = {
        "locations_m": np.linspace(0.0, beam.length_m, beam.element_count + 1),
        "tip_normalized_shape": shapes / tip_scale,
        "tip_normalized_slope_per_m": slopes / tip_scale,
        "frequency_hz": np.asarray([frequencies[0]]),
    }
    record = {
        "schema_version": "articulated-shaft-structural-basis/v1",
        "status": "synthetic_structural_reference_not_equipment_calibration",
        "beam_configuration": {
            "length_m": beam.length_m,
            "butt_diameter_m": beam.butt_diameter_m,
            "tip_diameter_m": beam.tip_diameter_m,
            "wall_thickness_m": beam.wall_thickness_m,
            "youngs_modulus_pa": beam.youngs_modulus_pa,
            "element_count": beam.element_count,
            "head_mass_kg": beam.head_mass_kg,
            "head_rotary_inertia_kg_m2": beam.head_rotary_inertia_kg_m2,
        },
        "first_bending_frequency_hz": float(frequencies[0]),
        "normalization": "unit tip translation with nodal rotations in radians",
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCES},
    }
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(NPZ_PATH, **arrays)
    print(JSON_PATH)
    print(NPZ_PATH)


if __name__ == "__main__":
    main()
