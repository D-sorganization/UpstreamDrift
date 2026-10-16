"""Asset-gated checks of the Bruno full-body map and driving (SPN-9c, #12758).

These need the owner's SimTK download and never run in CI: without the asset
env var (see ``src/config/spine_model_assets.json``) they skip with a reason
starting ``UNQUALIFIED: bruno2015_fullbody_os4_v2``.  A skip is never
acceptance.  Only names, topology and locked flags of the file are compared;
the smoke test reads values at runtime and asserts finiteness and shape, never
a licensed number, and nothing licensed is written to the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.assets import (
    resolve_spine_asset,
)
from src.engines.physics_engines.opensim.python.spine.drive import (
    GOLF_PELVIS,
    DriveOptions,
    drive_segmental_model,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_map import (
    get_fullbody_map,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_native import (
    coordinate_locked_flags,
    coordinate_names,
    probe_axis_dots,
    body_names,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_study import (
    FullbodySpec,
    run_fullbody,
    with_coordinates,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import (
    read_golf_humanoid_motion,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    run_spinal_inverse_dynamics,
)
from src.shared.python.spine.coupling_data import coupling_law
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.requires_opensim, pytest.mark.integration]

BRUNO = "bruno2015_fullbody_os4_v2"
REPO = Path(__file__).resolve().parents[3]
GOLF_MODEL = REPO / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"


def _bruno_path() -> Path:
    request = resolve_spine_asset(BRUNO)
    if request is None or not request.source_path.is_file():
        pytest.skip(f"UNQUALIFIED: {BRUNO} asset env var not set or file absent")
    return request.source_path


@pytest.fixture(scope="module")
def models():
    osim = require_opensim()
    path = _bruno_path()
    return osim.Model(str(GOLF_MODEL)), osim.Model(str(path)), path


def test_every_name_of_the_committed_map_exists_in_both_files(models) -> None:
    golf, target, _ = models
    fmap = get_fullbody_map(BRUNO)
    fmap.check_coverage(set(coordinate_names(golf)), set(coordinate_names(target)))
    golf_bodies, target_bodies = set(body_names(golf)), set(body_names(target))
    for group in fmap.groups:
        assert group.golf_body in golf_bodies and group.target_body in target_bodies
    for golf_hand, target_hand in fmap.hand_bodies.items():
        assert golf_hand in golf_bodies and target_hand in target_bodies
    assert set(fmap.foot_bodies) <= target_bodies


def test_locked_flags_match_the_recorded_treatment(models) -> None:
    _, target, _ = models
    fmap = get_fullbody_map(BRUNO)
    locked = coordinate_locked_flags(target)
    for name, treatment in fmap.target_treatment.items():
        assert locked[name] == (treatment == "locked_in_asset"), name


def test_recorded_signs_match_the_observed_axes(models) -> None:
    golf, target, _ = models
    fmap = get_fullbody_map(BRUNO)
    dots = probe_axis_dots(golf, target, (*fmap.direct, *fmap.pelvis))
    assert set(dots) == {e.golf for e in (*fmap.direct, *fmap.pelvis)}
    for entry in (*fmap.direct, *fmap.pelvis):
        assert entry.sign * dots[entry.golf] >= entry.min_axis_dot, (
            f"{entry.golf} -> {entry.target}: signed dot "
            f"{entry.sign * dots[entry.golf]:.3f} < {entry.min_axis_dot}"
        )


def _write_swing(path: Path, n: int = 41) -> Path:
    """Invented smooth swing-like motion of every golf coordinate (rad, m)."""
    osim = require_opensim()
    model = osim.Model(str(GOLF_MODEL))
    coords = model.getCoordinateSet()
    t = np.linspace(0.0, 0.4, n)
    cols: dict[str, np.ndarray] = {}
    for i in range(coords.getSize()):
        name = coords.get(i).getName()
        if "beta" in name:
            continue
        amp = 0.03 if name.startswith("pelvis_t") else 0.25
        cols[name] = amp * np.sin(5.0 * t + 0.37 * i)
    cols["pelvis_ty"] = cols["pelvis_ty"] + 0.9
    lines = [
        "synthetic",
        "version=1",
        f"nRows={n}",
        f"nColumns={len(cols) + 1}",
        "inDegrees=no",
        "endheader",
        "time\t" + "\t".join(cols),
    ]
    for k, tk in enumerate(t):
        lines.append("\t".join([f"{tk:.12f}"] + [f"{cols[c][k]:.15g}" for c in cols]))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_native_smoke_builds_the_table_and_runs_inverse_dynamics(
    models, tmp_path
) -> None:
    _, target, bruno_path = models
    motion = _write_swing(tmp_path / "swing.sto")
    golf = read_golf_humanoid_motion(motion, GOLF_MODEL)
    drive = drive_segmental_model(
        golf,
        model_path=bruno_path,
        model_id=BRUNO,
        law=coupling_law("bruno2015_regional"),
        options=DriveOptions(pelvis=GOLF_PELVIS),
    )
    fmap = get_fullbody_map(BRUNO)
    trunk = dict(drive.coordinates)
    spec = FullbodySpec(
        fmap=fmap,
        golf_model_path=GOLF_MODEL,
        motion_path=motion,
        target_model_path=bruno_path,
        policy="club_and_grf",
        output_dir=tmp_path / "out",
        stem="smoke",
    )
    result = run_fullbody(spec, golf.time_s, trunk)
    names = set(coordinate_names(target))
    assert set(result.coordinates) <= names
    for name in (
        *fmap.targets_with("driven_direct"),
        *fmap.targets_with("driven_orientation_match"),
    ):
        assert np.all(np.isfinite(result.coordinates[name])), name
    assert result.load_xml is not None and result.load_xml.is_file()
    kinds = {item["kind"] for item in result.record["included"]}
    assert kinds == {"club_hand_wrench", "ground_reaction"}
    merged = with_coordinates(drive, result.coordinates)
    series, budget = run_spinal_inverse_dynamics(
        bruno_path, merged, BRUNO, external_loads=result.load_xml, filter_hz=20.0
    )
    assert np.all(np.isfinite(series.moment_nm))
    print(
        "NATIVE SMOKE pelvis residual force ratio "
        f"{budget.force_ratio:.4g}, moment ratio {budget.moment_ratio:.4g}, "
        f"orientation residual max {result.metrics['orientation_max_residual_rad']:.3g} rad, "
        f"frames {golf.time_s.size}"
    )
