"""#14727 step 1: the generator reproduces the committed driver v2 bundle.

``trunk_visuals/driver_v2_bundle.npz`` (HRG-D3, #13174) was generated on spec
``4063705f`` from the true inputs: the ``anthro_driver_drake`` same-input
export, trunk-migrated from its v1 source spec onto the v2 spec. Its
``reference_q`` is that run's rollout, not its tracking target, so feeding
the bundle's own ``reference_q`` back in is a second-generation run that
diverges by design (1.7e-5 rad at step 1). On the true inputs the generator
matches to ~1e-13 rad over the first 100 steps; this test pins a short
horizon of that so a generator, plant or closure regression fails here.

The v1 source spec is ``full_body_spec_anthro_driver.json`` as of
``99a4b0c5ac1`` (main has since moved its grip frames), committed beside the
bundle as ``full_body_v1_source_spec.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.full_body_migration import (
    TrunkMigrator,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence"
TRUNK = EVIDENCE / "trunk_visuals"
RUN_DIR = EVIDENCE / "ground_support/anthro_driver_drake"
V1_SOURCE = TRUNK / "full_body_v1_source_spec.json"
V2_SPEC = TRUNK / "full_body_v2_anthro_driver.json"
BUNDLE = TRUNK / "driver_v2_bundle.npz"

#: Horizon of the regenerated run; the last sample is excluded from the
#: comparison because a truncated target changes its reference rate.
HORIZON_S = 0.03
#: Observed 1.4e-13 rad through step 99 (#14727); the bound leaves headroom
#: for platform round-off while still failing on the 1.7e-5 wrong-input path.
TOLERANCE_RAD = 1e-11


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_v1_source_spec_is_the_v2_specs_recorded_source() -> None:
    v2 = _json(V2_SPEC)
    assert canonical_sha256(_json(V1_SOURCE)) == v2["source_spec_sha256"]
    with np.load(BUNDLE) as bundle:
        assert bytes(bundle["spec"]) == V2_SPEC.read_bytes()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_mujoco
@pytest.mark.timeout(900)
def test_generator_reproduces_the_committed_v2_rollout(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    from scripts.same_input_bundle import export
    from src.shared.python.motion_matching.same_input.reference import (
        generate_reference_bundle,
    )

    manifest = export(RUN_DIR, tmp_path / "v1.npz", HORIZON_S)
    with np.load(tmp_path / "v1.npz") as v1:
        v1_q = v1["reference_q"]
    migrator = TrunkMigrator(_json(V1_SOURCE), _json(V2_SPEC))
    q_v2 = migrator.migrate_coordinates(v1_q, manifest["coordinate_order"])
    times = np.arange(q_v2.shape[0]) * manifest["dt_s"]
    regenerated = generate_reference_bundle(
        V2_SPEC.read_bytes(), times, q_v2, duration_s=HORIZON_S
    )
    steps = int(round(HORIZON_S / manifest["dt_s"]))
    with np.load(BUNDLE) as bundle:
        committed = bundle["reference_q"][:steps]
    error = np.abs(regenerated.reference_q[:steps] - committed).max()
    assert error < TOLERANCE_RAD, f"max |dq| {error:.3e} rad over {steps} steps"
