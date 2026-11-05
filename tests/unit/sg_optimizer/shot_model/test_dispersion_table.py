"""Dispersion tables: deterministic, hashed, convergent, regenerable (#14567)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    DispersionTable,
    build_table,
    Convergence,
    check_convergence,
    main,
    standard_normals,
)
from src.shared.python.sg_optimizer.shot_model.dispersion_lie import (
    lie_modifier_for_table,
)
from src.shared.python.sg_optimizer.shot_model.ground_runout import (
    SKILL_CLUB,
    rest_position,
)
from src.shared.python.sg_optimizer.shot_model.skill_registry import get_skill_family

pytestmark = pytest.mark.unit


def _linear_sim(
    impacts: np.ndarray, skill: str, wind: tuple[float, float], slope: str, lie: str
):
    """Analytic carry + documented runout."""
    along = 2.0 * impacts[:, 0] - wind[0]
    lateral = 3.0 * impacts[:, 1] + wind[1]
    carry = np.stack([along, lateral], axis=1)
    speed = np.clip(2.0 * impacts[:, 0], 10.0, 55.0)
    return rest_position(carry, slope=slope, lie=lie, landing_speed_ms=speed)


def _build(seed: int = 1, n: int = 4000, **kw):
    opts = {"skills": ("driver", "mid_iron"), "n": n, "seed": seed, **kw}
    return build_table(get_skill_family("tour"), _linear_sim, **opts)


def test_standard_normals_prefix_stable() -> None:
    """CRN: the first n draws are identical whatever the total count."""
    np.testing.assert_array_equal(
        standard_normals(7, 100)[:40], standard_normals(7, 40)
    )
    assert not np.array_equal(standard_normals(7, 5), standard_normals(8, 5))


def test_deterministic_per_seed_and_hash_in_provenance() -> None:
    a, b, c = _build(seed=1), _build(seed=1), _build(seed=2)
    assert a.table_hash == b.table_hash == a.provenance["table_hash"]
    assert a.table_hash != c.table_hash
    assert a == b


def test_cell_statistics_match_analytic_moments() -> None:
    t = _build()
    cell = t.lookup("driver", "driver", "fairway", "calm", "flat")
    spec = get_skill_family("tour").skills["driver"].distribution
    carry_mean = 2 * spec.mean[0]
    assert cell.mean[0] > carry_mean + 5.0  # bounce/roll adds downrange rest
    assert cell.sd[0] == pytest.approx(2 * spec.sd[0], rel=0.08)
    assert cell.sd[1] == pytest.approx(3 * spec.sd[1], rel=0.05)
    head = t.lookup("driver", "driver", "fairway", "head_5", "flat")
    assert head.mean[0] == pytest.approx(cell.mean[0] - 5.0, abs=1.0)


def test_unknown_key_fails_closed() -> None:
    with pytest.raises(ContractViolationError, match="no dispersion cell"):
        _build().lookup("driver", "driver", "rough", "calm", "flat")


def test_convergence_check_reports_stability() -> None:
    fam = get_skill_family("tour")
    big = check_convergence(fam, _linear_sim, "mid_iron", n=8000, seed=3)
    small = check_convergence(fam, _linear_sim, "mid_iron", n=40, seed=3)
    assert big.converged and big.max_rel_change < 0.02
    assert small.max_rel_change > big.max_rel_change


def test_runtime_report_present_and_not_hashed() -> None:
    a, b = _build(), _build()
    assert a.runtime["total_s"] >= 0.0 and a.runtime["cells"] == len(a.cells)
    assert a.table_hash == b.table_hash  # wall-clock never enters the hash


def test_json_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    t = _build(n=200)
    p = tmp_path / "t.json"
    t.to_json(p)
    assert DispersionTable.from_json(p) == t
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["cells"][0]["mean"][0] += 1.0
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="hash"):
        DispersionTable.from_json(p)


def test_lie_modifier_hook_enables_other_lies() -> None:
    def shorter(impacts: np.ndarray, lie: str) -> np.ndarray:
        out = impacts.copy()
        out[:, 0] *= 0.9 if lie == "rough" else 1.0
        return out

    t = _build(lies=("fairway", "rough"), lie_modifier=shorter)
    fw = t.lookup("driver", "driver", "fairway", "calm", "flat")
    ro = t.lookup("driver", "driver", "rough", "calm", "flat")
    assert ro.mean[0] < fw.mean[0]


def test_convergence_is_hashed_provenance_and_tamper_evident(tmp_path: Path) -> None:
    conv = Convergence(converged=False, max_rel_change=0.05)
    t = _build(n=200).with_convergence(conv)
    assert t.provenance["convergence"] == {"converged": False, "max_rel_change": 0.05}
    assert t.table_hash != _build(n=200).table_hash
    p = tmp_path / "t.json"
    t.to_json(p)
    assert DispersionTable.from_json(p) == t
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["provenance"]["convergence"]["converged"] = True
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="hash"):
        DispersionTable.from_json(p)


def test_cli_records_convergence_in_provenance_and_loads_back(tmp_path: Path) -> None:
    from src.shared.python.physics.rust_kernel import is_rust_available

    if not is_rust_available():
        pytest.skip("upstream-physics Rust kernel required for table CLI")
    out = tmp_path / "t.json"
    assert main(["--family", "tour", "--n", "8", "--out", str(out)]) == 0
    table = DispersionTable.from_json(out)
    assert table.provenance["convergence"]["max_rel_change"] >= 0.0
    assert isinstance(table.provenance["convergence"]["converged"], bool)
    assert table.provenance["n"] == 8
    assert table.provenance["ground_model"] == "sg_runout_v1"


def test_slope_bins_shift_downrange_mean() -> None:
    t = _build(winds=("calm",), n=3000, seed=4)
    flat = t.lookup("driver", "driver", "fairway", "calm", "flat")
    up = t.lookup("driver", "driver", "fairway", "calm", "uphill")
    down = t.lookup("driver", "driver", "fairway", "calm", "downhill")
    assert up.mean[0] < flat.mean[0] < down.mean[0]


def test_club_column_uses_representative_club() -> None:
    t = _build(skills=("mid_iron",), winds=("calm",), n=100, seed=2)
    assert SKILL_CLUB["mid_iron"] == "7_iron"
    t.lookup("mid_iron", "7_iron", "fairway", "calm", "flat")


def test_table_lie_modifier_lowers_rough_speed() -> None:
    mod = lie_modifier_for_table(9)
    t = _build(lies=("fairway", "rough"), lie_modifier=mod, winds=("calm",), n=4000)
    fw = t.lookup("driver", "driver", "fairway", "calm", "flat")
    ro = t.lookup("driver", "driver", "rough", "calm", "flat")
    assert ro.mean[0] < fw.mean[0]


def test_default_n_convergence_with_runout_linear_sim() -> None:
    fam = get_skill_family("tour")
    conv = check_convergence(fam, _linear_sim, "driver", n=12000, seed=0)
    assert conv.converged, conv.max_rel_change
