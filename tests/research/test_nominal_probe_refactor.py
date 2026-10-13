"""Nominal diagnostic extraction preserves solver bounds and residual frames."""

from types import SimpleNamespace

import numpy as np
import pytest

from docs.research.shot_pattern_analysis.phase2.review_followups import (
    probe_control_match_bounds as probe,
)

pytestmark = pytest.mark.unit


def test_multistart_preserves_solver_options_order_and_public_physics_adapter(
    monkeypatch,
):
    reference = probe.matched_preset("driver")
    config = probe.replace(reference, shaft_lean_deg=5.0)
    bounds = probe.MatchBounds(
        -6, 6, reference.club_speed_mps * 0.75, reference.club_speed_mps * 1.25
    )
    starts = []
    delivered = []

    class Physics:
        def simulate(self, *, face_deg, path_deg, config):
            delivered.append((face_deg, path_deg, config.club_speed_mps))
            return SimpleNamespace(carry_x_m=100 + face_deg, carry_y_m=path_deg)

    def fit(residual, initial, **options):
        starts.append(initial)
        assert options == {
            "bounds": ([-6, bounds.speed_min_mps], [6, bounds.speed_max_mps]),
            "max_nfev": 80,
            "xtol": 1e-11,
            "ftol": 1e-11,
            "gtol": 1e-11,
        }
        values = np.array([initial[0], initial[1]])
        np.testing.assert_array_equal(residual(values), [initial[0], initial[0] - 3])
        return SimpleNamespace(
            success=np.bool_(True),
            x=values,
            active_mask=np.array([0, 1]),
            optimality=0.125,
            nfev=4,
        )

    monkeypatch.setattr(probe, "least_squares", fit)
    rows = probe._diagnostic_multistart(
        Physics(), config, reference, bounds, 3.0, 100.0
    )
    assert starts == [[face, reference.club_speed_mps] for face in (-5.0, 0.0, 5.0)]
    assert rows == [
        {
            "success": True,
            "face_deg": face,
            "speed_mps": reference.club_speed_mps,
            "residual_xy_m": [face, face - 3],
            "active_mask": [0, 1],
            "optimality": 0.125,
            "n_evaluations": 4,
        }
        for face in (-5.0, 0.0, 5.0)
    ]
    assert delivered == [
        (face, face - 3, reference.club_speed_mps)
        for face in (-5.0, 0.0, 5.0)
        for _ in range(2)
    ]
