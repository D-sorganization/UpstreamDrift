"""Qualified state manifests must survive independent prefix experiments."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.engines.Simscape_Multibody_Models.python.tour_fit_state import (
    qualified_fit_identity,
    transfer_prefix_candidate,
)

pytestmark = pytest.mark.unit
EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/simscape_tour_matching/native_evidence"
)


def _transfer_reports() -> tuple[dict, dict]:
    source = json.loads((EVIDENCE / "prefix_100ms_fit_r2025b.json").read_text())
    target = deepcopy(source)
    target["fit_identity"].update(
        basis="linear-bernstein-6", duration_s=0.2, basis_duration_s=0.2
    )
    target["effort_scales"] = [s for s in source["effort_scales"] for _ in range(2)]
    return source, target


def test_explicit_transfer_preserves_constant_candidate_in_linear_basis() -> None:
    source, target = _transfer_reports()
    expected = [p for p in source["stage"]["parameters"] for _ in range(2)]
    assert transfer_prefix_candidate(source, target) == pytest.approx(expected)


def test_linear_transfer_preserves_physical_slope_when_horizon_grows() -> None:
    source, target = _transfer_reports()
    source["fit_identity"].update(basis="linear-bernstein-6", basis_duration_s=0.1)
    source["effort_scales"] = target["effort_scales"]
    source["stage"]["parameters"] = [1.2, 1.4] * 27
    assert transfer_prefix_candidate(source, target) == pytest.approx([1.2, 1.6] * 27)
    source["stage"]["parameters"] = [1.2, 1.8] * 27
    with pytest.raises(ValueError, match="bounds"):
        transfer_prefix_candidate(source, target)


@pytest.mark.parametrize(
    "field", ["q", "qd", "geometry_in", "coordinate_names", "offsets_m", "body_names"]
)
def test_transfer_rejects_changed_native_identity(field: str) -> None:
    source, target = _transfer_reports()
    target["fit_identity"][field] = []
    with pytest.raises(ValueError, match="identity"):
        transfer_prefix_candidate(source, target)


def test_transfer_rejects_other_capture_and_unfinished_source() -> None:
    source, target = _transfer_reports()
    with pytest.raises(ValueError, match="capture"):
        transfer_prefix_candidate(source, target | {"source_sha256": "other"})
    with pytest.raises(ValueError, match="completed"):
        transfer_prefix_candidate(source | {"status": "failed"}, target)
    target["effort_scales"][0] = 200
    with pytest.raises(ValueError, match="scales"):
        transfer_prefix_candidate(source, target)


def test_qualified_state_preserves_si_rates_geometry_and_fixed_offsets() -> None:
    seed = json.loads(
        (EVIDENCE / "initial_velocity_seed_qualified_r2025b.json").read_text()
    )
    identity = qualified_fit_identity(seed, seed["source_sha256"], 0.1, 1.8138888889)
    for key in (
        "coordinate_names",
        "q",
        "qd",
        "geometry_in",
        "labels",
        "body_names",
        "offsets_m",
    ):
        assert identity[key] == seed[key]
    assert identity["duration_s"] == 0.1
    assert identity["basis_duration_s"] == 1.8138888889
    original = deepcopy(identity)
    seed["q"][0] = 100
    assert identity == original


@pytest.mark.parametrize(
    "changed",
    [
        {"initial_state_verified": False},
        {"status": "replayed"},
        {"q": [0]},
        {"qd": [float("nan")] * 27},
        {"geometry_in": [-1, 12]},
        {"coordinate_names": ["HipInputX"] * 27},
        {"offsets_m": [[0, 0, 0]]},
        {"body_names": ["Hip"]},
    ],
)
def test_invalid_or_unqualified_state_is_rejected(changed: dict) -> None:
    seed = json.loads(
        (EVIDENCE / "initial_velocity_seed_qualified_r2025b.json").read_text()
    )
    with pytest.raises(ValueError):
        qualified_fit_identity(seed | changed, seed["source_sha256"], 0.1, 1.8)


def test_capture_and_horizon_are_bound_to_the_state() -> None:
    seed = json.loads(
        (EVIDENCE / "initial_velocity_seed_qualified_r2025b.json").read_text()
    )
    with pytest.raises(ValueError, match="capture"):
        qualified_fit_identity(seed, "another-capture", 0.1, 1.8)
    for duration in (0, -1, float("nan"), 2):
        with pytest.raises(ValueError, match="duration"):
            qualified_fit_identity(seed, seed["source_sha256"], duration, 1.8)


@pytest.mark.parametrize(
    "before,after", [(a, b) for a in range(1, 8) for b in range(a, 8)]
)
def test_higher_degree_transfer_preserves_physical_polynomial(
    before: int, after: int
) -> None:
    import numpy as np
    from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

    names = dict(
        enumerate(
            (
                "constant",
                "linear",
                "quadratic",
                "cubic",
                "quartic",
                "quintic",
                "sextic",
            ),
            start=1,
        )
    )
    source, target = _transfer_reports()
    source["fit_identity"].update(
        basis=f"{names[before]}-bernstein-6", basis_duration_s=0.1
    )
    target["fit_identity"].update(
        basis=f"{names[after]}-bernstein-6", basis_duration_s=0.12, duration_s=0.12
    )
    original = 1 + np.random.default_rng(9921).uniform(-0.001, 0.001, (27, before))
    source["stage"]["parameters"] = original.ravel().tolist()
    source["effort_scales"] = [200.0] * (27 * before)
    target["effort_scales"] = [200.0] * (27 * after)
    transferred = np.asarray(transfer_prefix_candidate(source, target)).reshape(
        27, after
    )
    old = bernstein_to_simscape(original - 1, duration_s=0.1)
    new = bernstein_to_simscape(transferred - 1, duration_s=0.12)
    for t in np.linspace(0, 0.12, 17):
        assert np.polyval(old.T, t) == pytest.approx(np.polyval(new.T, t), abs=1e-10)
