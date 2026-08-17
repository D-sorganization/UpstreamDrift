"""Running both tiers on one F0 record (issue #8713, epic #8699).

The arithmetic is covered in ``test_crosstier``. What is pinned here is
the wiring: that the pose F0 recorded reaches F1 unchanged, that the pair
produces the divergence ADR-0033 measured rather than an accidental
agreement, and that the probes land where the comparison says they do.

The F1 bed here is deliberately coarse -- a 6 mm cell on a 50 mm bed --
because a single probe at ADR-0033's resolution costs tens of seconds and
this suite runs on a 60-second timeout. Coarse changes the numbers, so
nothing below is pinned to one: the assertions are on sign, direction and
structure, which is what survives the discretisation. The quotable numbers
are the ones ADR-0033 and SPEC record from a production-resolution run.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.solvers.drft import DRFTSolver, MaterialResponse
from bunkershot3d.solvers.envelope import RefusalPolicy
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver
from src.tools.bunker_shot_gui.bridge import (
    compare_tiers,
    entry_kinematics,
    probe_frames,
    validity_band,
)
from src.tools.bunker_shot_gui.crosstier import ComparedQuantity
from src.tools.bunker_shot_gui.design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel

pytestmark = pytest.mark.unit

COARSE = SolverSetup(
    n_profile_points=12, n_stations=5, playability_points=2, target_carry_m=12.0
)
"""The cheapest head the workbench will loft; F1's cost dominates anyway."""


@pytest.fixture(scope="module")
def bench():  # type: ignore[no-untyped-def]
    """One F0 shot and everything needed to put F1 beside it."""
    model = WorkbenchModel(COARSE)
    design = WedgeDesign(name="cross-tier")
    geometry = design.geometry()
    sand = SandCondition().sand_state()
    swing = SwingSetup()
    build = model.head_build(geometry)
    kinematics = entry_kinematics(build, swing)
    outcome = model.run_shot(geometry, sand, swing)
    permissive = DRFTSolver(
        material=MaterialResponse.from_sand_state(sand),
        dynamic_terms_active=bool(swing.dynamic_terms_active),
        refusal_policy=RefusalPolicy.REPORT,
    )
    return {
        "model": model,
        "geometry": geometry,
        "sand": sand,
        "swing": swing,
        "build": build,
        "kinematics": kinematics,
        "outcome": outcome,
        "f0": permissive,
        "f1": PlaneStrainMPMSolver(
            material=SandContinuum.from_sand_state(sand),
            cell_size_m=0.006,
            effective_width_m=geometry.sole_width_m,
            bed_depth_m=0.050,
            run_in_lengths=0.4,
            refusal_policy=RefusalPolicy.REPORT,
            max_steps=60000,
        ),
    }


@pytest.fixture(scope="module")
def comparison(bench):  # type: ignore[no-untyped-def]
    outcome = bench["outcome"]
    result = bench["model"].shot_result(
        bench["geometry"], bench["sand"], bench["swing"]
    )
    band = validity_band(
        bench["f0"], bench["build"], result, bench["kinematics"].orientation
    )
    assert band is not None
    return compare_tiers(
        f0_solver=bench["f0"],
        f1_solver=bench["f1"],
        build=bench["build"],
        result=result,
        kinematics=bench["kinematics"],
        f0_divot_section_area_m2=outcome.scene.divot.section_area_m2,
        band=band,
        head_mass_kg=bench["geometry"].head_mass_kg,
        bulk_density_kg_m3=bench["sand"].bulk_density_kg_m3,
        n_probes=2,
    )


class TestProbeSelection:
    """Which samples are worth the expense of an F1 march."""

    def test_only_engaged_samples_are_probed(self, bench) -> None:  # type: ignore[no-untyped-def]
        result = bench["model"].shot_result(
            bench["geometry"], bench["sand"], bench["swing"]
        )
        frames = probe_frames(result, 3)
        assert frames
        for frame in frames:
            assert result.active_areas_m2[frame] > 0.0

    def test_the_peak_force_sample_is_always_probed(self, bench) -> None:  # type: ignore[no-untyped-def]
        result = bench["model"].shot_result(
            bench["geometry"], bench["sand"], bench["swing"]
        )
        peak = int(np.argmax(np.linalg.norm(result.forces_n, axis=1)))
        assert peak in probe_frames(result, 2)

    def test_a_non_positive_probe_count_is_refused(self, bench) -> None:  # type: ignore[no-untyped-def]
        result = bench["model"].shot_result(
            bench["geometry"], bench["sand"], bench["swing"]
        )
        with pytest.raises(ValueError, match="n_probes"):
            probe_frames(result, 0)


class TestTheRecordedPoseReachesF1Unchanged:
    """The one row where a disagreement would be a wiring fault."""

    def test_the_two_tiers_report_the_same_sole_depth(self, comparison) -> None:  # type: ignore[no-untyped-def]
        for probe in comparison.shot_probes:
            agreement = probe.agreement(ComparedQuantity.SOLE_DEPTH)
            assert agreement.ratio == pytest.approx(1.0, abs=1e-9), agreement.summary()

    def test_f0s_probe_force_matches_the_force_the_march_recorded(
        self,
        comparison,  # type: ignore[no-untyped-def]
    ) -> None:
        """Rebuilding the state must reproduce the solve, not approximate it."""
        for probe in comparison.shot_probes:
            recorded = float(np.linalg.norm(comparison.f0_force_n[probe.frame]))
            assert probe.f0_force_magnitude_n == pytest.approx(recorded, rel=1e-9)

    def test_every_probe_sits_at_the_sample_it_names(self, comparison) -> None:  # type: ignore[no-untyped-def]
        for probe in comparison.shot_probes:
            assert probe.time_s == pytest.approx(float(comparison.time_s[probe.frame]))


class TestTheDivergenceIsReportedRatherThanSmoothed:
    """Sign and structure, not numbers a coarse bed would move."""

    def test_both_tiers_oppose_the_head(self, comparison) -> None:  # type: ignore[no-untyped-def]
        """The one physical claim a pair of uncalibrated models can support."""
        for probe in comparison.shot_probes:
            assert probe.check.f0_force_n[2] > 0.0, probe.check.summary()
            assert probe.check.f1_force_n[2] > 0.0, probe.check.summary()

    def test_f1_carries_a_divot_f0_cannot(self, comparison) -> None:  # type: ignore[no-untyped-def]
        """The reason ADR-0033 chose a continuum at all."""
        assert comparison.peak_probe.f1_divot_section_area_m2 > 0.0

    def test_f0_credits_more_of_its_force_to_inertia_at_greenside_speed(
        self,
        comparison,  # type: ignore[no-untyped-def]
    ) -> None:
        """The mechanism, at the one speed a greenside shot actually runs at."""
        peak = comparison.peak_probe
        assert peak.speed_m_s > 10.0
        assert peak.f0_inertial_fraction > peak.f1_flux_fraction, peak.check.summary()
        assert peak.inertial_share_gap > 0.0

    def test_the_wrench_agreement_is_reported_with_its_ratio(
        self,
        comparison,  # type: ignore[no-untyped-def]
    ) -> None:
        agreement = comparison.agreement(ComparedQuantity.WRENCH)
        assert np.isfinite(agreement.ratio)
        assert agreement.ratio > 0.0
        assert "declared band" in agreement.summary()

    def test_the_crossover_is_stated_either_way(self, comparison) -> None:  # type: ignore[no-untyped-def]
        text = comparison.crossover_summary()
        assert "yield surface" in text

    def test_the_summary_leads_with_the_licence(self, comparison) -> None:  # type: ignore[no-untyped-def]
        text = comparison.summary()
        assert text.startswith("What this comparison licenses: nothing about sand")
        assert "not validation" in text.lower()
        assert "8733" in text


class TestTheDeclaredSpeedSweep:
    """The crossover needs a range the shot itself does not span."""

    def test_a_sweep_brackets_speeds_the_shot_never_reaches(self, bench) -> None:  # type: ignore[no-untyped-def]
        outcome = bench["outcome"]
        result = bench["model"].shot_result(
            bench["geometry"], bench["sand"], bench["swing"]
        )
        band = validity_band(
            bench["f0"], bench["build"], result, bench["kinematics"].orientation
        )
        assert band is not None
        swept = compare_tiers(
            f0_solver=bench["f0"],
            f1_solver=bench["f1"],
            build=bench["build"],
            result=result,
            kinematics=bench["kinematics"],
            f0_divot_section_area_m2=outcome.scene.divot.section_area_m2,
            band=band,
            head_mass_kg=bench["geometry"].head_mass_kg,
            bulk_density_kg_m3=bench["sand"].bulk_density_kg_m3,
            n_probes=1,
            # Fast on purpose. The approach is a fixed distance and the CFL
            # step is set by the elastic wave speed, so a *slow* probe is the
            # expensive one: the step count goes as ``(c_p + v) / v``, and a
            # 4 m/s probe costs order thirty times a 25 m/s one.
            sweep_speeds_m_s=(22.0, 30.0),
        )
        speeds = sorted(probe.speed_m_s for probe in swept.sweep_probes)
        assert speeds == pytest.approx([22.0, 30.0])
        assert swept.crossover_probes == swept.sweep_probes
