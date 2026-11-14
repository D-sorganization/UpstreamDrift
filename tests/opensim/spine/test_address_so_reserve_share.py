"""Address-event reserve share regression (SPN-D9, #13427).

The native Bruno capture-A anomaly (reserve share 1.344 at address) was traced to
non-physiological trunk IK (#13434).  With a reachable synthetic fixture and
physiological lumbar motion, static optimisation at address must stay below the
declared ``max_share`` gate (0.05) without loosening reserve limits.

Licensed OS-9 capture-A acceptance runs only when OpenSim and the Bruno asset
are available; it records measured reserve shares in the study receipt.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.assets import resolve_spine_asset
from src.engines.physics_engines.opensim.python.spine.drive import FIXTURE_PELVIS
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_level_names,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.inventory import load_level_maps
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    SpineIDOptions,
    spinal_generalized_forces,
)
from src.engines.physics_engines.opensim.python.spine.static_optimization import (
    ReserveSpec,
    SpineSOOptions,
    run_spine_static_optimization,
)
from src.engines.physics_engines.opensim.python.spine.study import (
    SpineStudyRequest,
    run_spine_study,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_swing import write_sto
from src.shared.python.spine.levels import SpinalLevel
from tests.helpers.opensim_support import require_opensim

REPO = Path(__file__).resolve().parents[3]
GOLF_MODEL = REPO / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
OS9_DIR = (
    REPO / "docs/development/opensim_tour_matching/evidence/os9_address_registration"
)
OS9_STO = OS9_DIR / "ik_states_continuous.sto"
OS9_GOLF = OS9_DIR / "golf_humanoid_scaled_tour_markers_address_registered.osim"
OS9_STO_SHA = "f489d41b68e84df3addab2ade825ae0266e672e60ec3abd083275f9f8c76d3f2"
OS9_GOLF_SHA = "56c6c6f0e947ae92d2b6041135e049b68284d46c70fee135339ba3e3f447393e"
BRUNO = "bruno2015_fullbody_os4_v2"

ADDRESS_MAX_SHARE = 0.05
_EVENTS = {"address": 0.0, "top": 0.4, "impact": 0.7}
_TIMES = np.linspace(0.0, 0.7, 8)


def _ramp(lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, _TIMES.size)


def _columns(spec: FixtureSpec, angles: dict[SpinalLevel, tuple]) -> dict:
    cols = {n: np.zeros(_TIMES.size) for n in FIXTURE_PELVIS.rotation}
    cols.update({n: np.zeros(_TIMES.size) for n in FIXTURE_PELVIS.translation})
    for names in fixture_level_names(spec):
        level_angles = angles[names.level]
        for coord, values in zip(names.coordinates, level_angles, strict=True):
            cols[coord] = np.asarray(values, dtype=float)
    return cols


def _efforts(path: Path, spec: FixtureSpec, angles, tmp: Path, level_map):
    mot = write_sto(tmp / "motion.mot", _TIMES, _columns(spec, angles))
    return spinal_generalized_forces(
        path,
        mot,
        "fixture",
        options=SpineIDOptions(pelvis=FIXTURE_PELVIS, level_map=level_map),
    )


@pytest.fixture(scope="module")
def one_level_reachable(tmp_path_factory):
    require_opensim()
    spec = FixtureSpec(levels=(SpinalLevel.L3_L4,))
    tmp = tmp_path_factory.mktemp("addr_so_one")
    muscled = write_fixture_osim(spec, tmp / "muscled.osim", muscles=True)
    full = load_level_maps()["fixture"]
    level_map = replace(
        full, levels={SpinalLevel.L3_L4: full.levels[SpinalLevel.L3_L4]}
    )
    return spec, muscled, level_map, tmp


def _physiological_angles() -> dict:
    return {
        SpinalLevel.L3_L4: (_ramp(0.10, 0.16), _ramp(0.05, 0.02), _ramp(-0.08, 0.0))
    }


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_address_so_reserve_share_below_max_share_on_synthetic_fixture(
    one_level_reachable,
) -> None:
    """Physiological fixture motion keeps address reserve share under 0.05 (#13427)."""
    spec, muscled, level_map, tmp = one_level_reachable
    efforts = _efforts(muscled, spec, _physiological_angles(), tmp, level_map)
    reserves = ReserveSpec(limit_nm=1.0, max_share=ADDRESS_MAX_SHARE)
    options = SpineSOOptions(
        exponent=3,
        events=_EVENTS,
        level_map=level_map,
        reserves=reserves,
    )
    result = run_spine_static_optimization(muscled, efforts, "fixture", options=options)
    assert "address" in result.frame_labels
    idx = list(result.frame_labels).index("address")
    share = float(result.reserve_share[idx])
    assert share < ADDRESS_MAX_SHARE, (
        f"address reserve share {share:.4g} must stay below {ADDRESS_MAX_SHARE}"
    )
    assert not bool(result.flagged[idx])


def _bruno_model_path() -> Path | None:
    request = resolve_spine_asset(BRUNO)
    if request is None or not request.source_path.is_file():
        return None
    return request.source_path


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_os9_capture_address_reserve_share_below_max_share(tmp_path) -> None:
    """OS-9 registered capture-A IK: address reserve share below max_share when qualified."""
    require_opensim()
    if not (OS9_STO.is_file() and OS9_GOLF.is_file()):
        pytest.skip("OS-9 registration evidence files are not present")
    bruno = _bruno_model_path()
    if bruno is None:
        pytest.skip(f"UNQUALIFIED: {BRUNO} asset env var not set or file absent")

    from src.shared.python.pose_estimation.model_files import sha256_of

    assert sha256_of(OS9_STO) == OS9_STO_SHA
    assert sha256_of(OS9_GOLF) == OS9_GOLF_SHA

    request = SpineStudyRequest(
        study_id="spn_d9_os9_address",
        motion_path=OS9_STO,
        golf_model_path=OS9_GOLF,
        model_id=BRUNO,
        asset_id=BRUNO,
        model_path=bruno,
        fullbody_map=BRUNO,
        external_loads_policy="club_and_grf",
        muscle_model_policy="millard_equivalent",
        allow_implausible=True,
        events=(
            ("address", 0.211),
            ("top", 1.103),
            ("impact", 1.319),
        ),
        output_dir=tmp_path / "out",
        reserve_max_share=ADDRESS_MAX_SHARE,
    )
    receipt = run_spine_study(request, code_version="spn_d9_os9")
    if receipt.status == "unqualified" and any(
        "asset" in r for r in receipt.status_reasons
    ):
        pytest.skip("Bruno asset failed admission")

    so_stage = next(s for s in receipt.stages if s.stage == "static_optimization")
    by_label = so_stage.metrics.get("reserve_share_by_label")
    assert isinstance(by_label, dict), so_stage.metrics
    address_share = float(by_label["address"])
    assert address_share < ADDRESS_MAX_SHARE, (
        f"OS-9 address reserve share {address_share:.4g} must stay below "
        f"{ADDRESS_MAX_SHARE} on registered IK (#13427)"
    )
