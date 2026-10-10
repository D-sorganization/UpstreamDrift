"""Native source-preservation checks for the additional BUET-Hamner candidate."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.native_buet_hamner_assembly import (
    BuetHamnerAssemblyRequest,
    assemble_buet_hamner,
)

pytestmark = pytest.mark.integration


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> tuple[Path, Path]:
    buet_name = os.environ.get("UD_BUET_SOURCE")
    hamner_name = os.environ.get("UD_HAMNER_SOURCE")
    if not buet_name or not hamner_name:
        pytest.skip("owned exact BUET and Hamner source paths are required")
    return Path(buet_name), Path(hamner_name)


def test_exact_source_assembly_preserves_owned_regions_after_reload(
    tmp_path: Path,
) -> None:
    opensim = pytest.importorskip("opensim")
    buet, hamner = _sources()
    output = tmp_path / "owned_assembly.osim"
    receipt = assemble_buet_hamner(
        BuetHamnerAssemblyRequest(buet, _sha(buet), hamner, _sha(hamner), output)
    )
    fresh = opensim.Model(str(output))
    fresh.initSystem()
    assert receipt.derived_sha256 == _sha(output)
    assert fresh.getBodySet().getSize() == 40
    assert fresh.getMuscles().getSize() == 557
    assert receipt.buet_muscle_count == 473
    assert receipt.hamner_muscle_count == 84
    assert receipt.observed_pose_count >= 3
    assert receipt.max_shared_body_transform_error < 1e-10
    assert receipt.max_shared_body_velocity_error < 1e-10
    assert receipt.max_owned_body_transform_error < 1e-10
    assert receipt.max_owned_body_velocity_error < 1e-10
    assert receipt.max_owned_muscle_length_error < 1e-10
    assert receipt.max_owned_muscle_speed_error < 1e-10
    assert receipt.minimum_native_mass_eigenvalue > 0.0
    assert receipt.native_applied_force_norm > 0.0
    assert receipt.max_native_constraint_residual < 1e-8
    assert len(receipt.native_simulation_sha256) == 64
    assert len(receipt.native_simbody_sha256) == 64
    assert len(receipt.native_actuators_sha256) == 64


def test_changed_source_or_existing_output_is_rejected(tmp_path: Path) -> None:
    buet, hamner = _sources()
    output = tmp_path / "owned_assembly.osim"
    with pytest.raises(ValueError, match="source"):
        assemble_buet_hamner(
            BuetHamnerAssemblyRequest(buet, "0" * 64, hamner, _sha(hamner), output)
        )
    output.write_text("already-owned", encoding="utf-8")
    with pytest.raises(FileExistsError):
        assemble_buet_hamner(
            BuetHamnerAssemblyRequest(buet, _sha(buet), hamner, _sha(hamner), output)
        )


def test_new_hash_cannot_silently_admit_unreviewed_source_bytes(tmp_path: Path) -> None:
    buet, hamner = _sources()
    modified = tmp_path / "modified.osim"
    modified.write_bytes(buet.read_bytes() + b"\n<!-- altered-review-source -->\n")
    with pytest.raises(ValueError, match="reviewed source"):
        assemble_buet_hamner(
            BuetHamnerAssemblyRequest(
                modified,
                _sha(modified),
                hamner,
                _sha(hamner),
                tmp_path / "output.osim",
            )
        )
