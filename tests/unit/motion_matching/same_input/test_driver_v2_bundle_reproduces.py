"""#14727 step 1: the generator reproduces the committed driver v2 bundle.

``trunk_visuals/driver_v2_bundle.npz`` (HRG-D3, #13174) is generated from the
true inputs: the ``anthro_driver_drake`` same-input export, trunk-migrated
from the run's hip-calibrated, scaled spec onto the v2 spec. Its
``reference_q`` is that run's rollout, not its tracking target, so feeding
the bundle's own ``reference_q`` back in is a second-generation run that
diverges by design. This test pins a short horizon of the true-input run so
a generator, plant or closure regression fails here.

The v2 trunk is derived from ``full_body_v1_source_spec.json`` (the run's
uncalibrated base, ``full_body_spec_anthro_driver.json`` as of
``99a4b0c5ac1``), then carries the run's calibration (#15670). Spec
``4063705f`` skipped that step. Its pelvis sank and slid off the target, and
the golfer fell about 0.8 s in.
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
RUN_SPEC = RUN_DIR / "full_body_spec_hipcal_scaled.json"
V1_SOURCE = TRUNK / "full_body_v1_source_spec.json"
V2_SPEC = TRUNK / "full_body_v2_anthro_driver.json"
BUNDLE = TRUNK / "driver_v2_bundle.npz"

#: Horizon of the regenerated run; the last sample is excluded from the
#: comparison because a truncated target changes its reference rate.
HORIZON_S = 0.03
#: Observed 1.4e-13 rad through step 99 (#14727); the bound leaves headroom
#: for platform round-off while still failing on the 1.7e-5 wrong-input path.
TOLERANCE_RAD = 1e-11
#: Root translation coordinates, first in both the v1 and the v2 order.
ROOT_TRANSLATION = ("TranslationInputX", "TranslationInputY", "TranslationInputZ")
#: Pelvis height and planar slack of the rollout from the run's target (#15670).
#: The calibrated rollout peaks at 23 mm and 59 mm, at impact and on the last
#: follow-through sample; the original MuJoCo pipeline run itself reaches 14 mm
#: and 30 mm. Spec ``4063705f`` missed by 1.74 m (it sank and slid away).
HEIGHT_TOLERANCE_M = 0.03
PLANAR_TOLERANCE_M = 0.08


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_the_run_spec_is_the_v2_specs_recorded_source() -> None:
    v2 = _json(V2_SPEC)
    assert canonical_sha256(_json(RUN_SPEC)) == v2["source_spec_sha256"]
    with np.load(BUNDLE) as bundle:
        assert bytes(bundle["spec"]) == V2_SPEC.read_bytes()


@pytest.mark.unit
def test_the_v2_spec_carries_the_runs_hip_calibration() -> None:
    def joint(spec: dict, name: str) -> dict:
        return next(j for j in spec["joints"] if j["name"] == name)

    v2, run, base = _json(V2_SPEC), _json(RUN_SPEC), _json(V1_SOURCE)
    for name in ("hip_r", "hip_l", "ankle_r", "ankle_l"):
        assert joint(v2, name) == joint(run, name) != joint(base, name)
    assert v2["contact"] == run["contact"]


@pytest.mark.unit
def test_the_rollout_keeps_the_pelvis_on_the_runs_target() -> None:
    """#15670: the demo golfer no longer falls over about 0.8 s in."""
    with np.load(RUN_DIR / "dynamics_record.npz") as record:
        track_t, track_q = record["track_time_s"], record["q_track"]
    with np.load(BUNDLE) as bundle:
        manifest = json.loads(str(bundle["manifest"]))
        rollout = bundle["reference_q"]
    order = manifest["coordinate_order"]
    assert tuple(order[:3]) == ROOT_TRANSLATION
    times = np.arange(rollout.shape[0]) * manifest["dt_s"]
    target = np.column_stack(
        [np.interp(times, track_t - track_t[0], track_q[:, i]) for i in range(3)]
    )
    error = np.abs(rollout[:, :3] - target)
    assert error[:, 2].max() < HEIGHT_TOLERANCE_M, f"height {error[:, 2].max():.3f}"
    assert error[:, :2].max() < PLANAR_TOLERANCE_M, f"planar {error[:, :2].max():.3f}"


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
    migrator = TrunkMigrator(_json(RUN_SPEC), _json(V2_SPEC))
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
