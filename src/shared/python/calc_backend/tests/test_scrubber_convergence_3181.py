"""Scrubber flux/flooding/diameter convergence regression tests (#3181).

Self-consistency of the iterated flux/flooding/diameter solve. Split out of
``test_calc_backend.py`` so the fully annotated regression lives in its own
file (delta-CI mypy clean).
"""

from __future__ import annotations

from typing import Any

import pytest


class TestScrubberFluxConvergence:
    """Self-consistency of the iterated flux/flooding/diameter solve (#3181)."""

    def test_flux_self_consistent_with_solved_area(self) -> None:
        """Recomputing flux at the solved area reproduces the flooding velocity.

        The historical placeholder divided liquid flow against an assumed
        1 m2 area, biasing the flooding velocity whenever the real area != 1.
        After convergence, the flux computed from the solved cross-section
        must reproduce the same flooding velocity (within 1e-6).
        """
        from calc_backend.routers.scrubber import _solve_flux_flooding_diameter
        from sidekick.process_calculators.scrubber_calculator import (
            PACKING_DATABASE,
            WATER_DENSITY,
            WATER_VISCOSITY,
            calculate_flooding_velocity,
            calculate_gas_density,
        )

        packing: Any = PACKING_DATABASE["Metal Pall Rings"]
        gas_density = calculate_gas_density(400.0, 101325.0, 28.0)
        liquid_flow_kg_hr = 5000.0

        flooding_velocity, column_result = _solve_flux_flooding_diameter(
            gas_flow_kg_hr=10000.0,
            liquid_flow_kg_hr=liquid_flow_kg_hr,
            gas_density=gas_density,
            percent_of_flood=70.0,
            packing=packing,
        )

        solved_area = column_result["cross_section_m2"]
        assert solved_area > 0.0
        # The solved area is NOT the 1 m2 seed -> the bug would have biased it.
        assert abs(solved_area - 1.0) > 1e-3

        # Recompute flux from the solved area and re-derive flooding velocity.
        flux_at_solved_area = liquid_flow_kg_hr / (3600.0 * solved_area)
        recomputed_flooding = calculate_flooding_velocity(
            liquid_mass_flux=flux_at_solved_area,
            gas_density=gas_density,
            liquid_density=WATER_DENSITY,
            packing=packing,
            liquid_viscosity=WATER_VISCOSITY,
        )

        assert recomputed_flooding == pytest.approx(flooding_velocity, abs=1e-6)
