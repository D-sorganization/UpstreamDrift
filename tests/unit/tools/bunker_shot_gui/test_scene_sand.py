"""Carrying a sand volume into the 3-D scene (issue #8729), headless.

The scene has drawn a flat translucent plane since issue #8706, because
F0 resolves no grains and the plane is a boundary condition. Once a tier
does resolve grains the scene must be able to carry them -- and must not
be able to carry somebody else's.

The failure this file exists to prevent is a silent tier substitution: an
F1 sand field drawn over an F0 shot, or the reverse. Both look entirely
plausible. Both are a claim the run behind the picture never made.
"""

from __future__ import annotations

import dataclasses

import pytest

from bunkershot3d.solvers.protocol import FidelityTier
from src.tools.bunker_shot_gui.sandvolume import sand_volume
from src.tools.bunker_shot_gui.shot3d import SandSurface, ShotScene

from .test_slices import analytic_field

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def f1_volume():  # type: ignore[no-untyped-def]
    """A volume from the analytic F1 field the slice tests already use."""
    return sand_volume(analytic_field(), n_sheets=3, max_cells=200)


def f1_scene(nominal_scene: ShotScene) -> ShotScene:
    """The nominal scene re-labelled F1 and given a sand volume."""
    volume = f1_volume()
    return dataclasses.replace(
        nominal_scene,
        fidelity_tier=FidelityTier.F1,
        surface=dataclasses.replace(
            nominal_scene.surface, resolves_grains=True, tier=FidelityTier.F1
        ),
        divot=dataclasses.replace(
            nominal_scene.divot, resolves_grains=True, tier=FidelityTier.F1
        ),
        sand=volume,
    )


class TestTheSceneStatesWhoSolvedTheSand:
    """A picture of moving sand must name the tier that moved it."""

    def test_an_f0_scene_still_says_it_resolves_no_grains(
        self, nominal_scene: ShotScene
    ) -> None:
        assert nominal_scene.sand is None
        assert nominal_scene.surface.resolves_grains is False
        assert "resolves no grains" in nominal_scene.surface.describe()

    def test_an_f0_scene_names_f0_in_its_sand_note(
        self, nominal_scene: ShotScene
    ) -> None:
        assert "F0" in " ".join(nominal_scene.sand_note())

    def test_a_grain_resolving_surface_stops_denying_grains(self) -> None:
        surface = SandSurface(
            height_m=0.0,
            along_extent_m=(-0.1, 0.1),
            across_extent_m=(-0.05, 0.05),
            resolves_grains=True,
            tier=FidelityTier.F1,
        )
        assert "resolves no grains" not in surface.describe()
        assert "F1" in surface.describe()

    def test_the_free_surface_stays_an_input_even_when_grains_are_solved(
        self,
    ) -> None:
        """The height is set up with, not solved for, at every tier."""
        surface = SandSurface(
            height_m=0.0,
            along_extent_m=(-0.1, 0.1),
            across_extent_m=(-0.05, 0.05),
            resolves_grains=True,
            tier=FidelityTier.F1,
        )
        assert "not a result" in surface.describe()

    def test_a_scene_with_sand_says_extruded_in_its_note(
        self, nominal_scene: ShotScene
    ) -> None:
        note = " ".join(f1_scene(nominal_scene).sand_note()).lower()
        assert "extrud" in note

    def test_a_scene_with_sand_still_qualifies_the_divot(
        self, nominal_scene: ShotScene
    ) -> None:
        """The swept envelope is not transported sand at any tier."""
        note = " ".join(f1_scene(nominal_scene).sand_note())
        assert "divot" in note


class TestTierSubstitutionIsRefused:
    """The most persuasive unlabelled picture this epic could produce."""

    def test_an_f1_field_over_an_f0_shot_is_refused(
        self, nominal_scene: ShotScene
    ) -> None:
        assert nominal_scene.fidelity_tier is FidelityTier.F0
        with pytest.raises(ValueError, match="F1"):
            dataclasses.replace(nominal_scene, sand=f1_volume())

    def test_the_refusal_names_both_tiers(self, nominal_scene: ShotScene) -> None:
        with pytest.raises(ValueError) as caught:
            dataclasses.replace(nominal_scene, sand=f1_volume())
        message = str(caught.value)
        assert "F0" in message and "F1" in message

    def test_a_matching_tier_is_accepted(self, nominal_scene: ShotScene) -> None:
        scene = f1_scene(nominal_scene)
        assert scene.sand is not None
        assert scene.sand.fidelity_tier is scene.fidelity_tier

    def test_a_grain_resolving_scene_without_a_field_is_refused(
        self, nominal_scene: ShotScene
    ) -> None:
        """A surface that claims grains with no grains behind it."""
        with pytest.raises(ValueError, match="resolves_grains"):
            dataclasses.replace(
                nominal_scene,
                surface=dataclasses.replace(
                    nominal_scene.surface, resolves_grains=True, tier=FidelityTier.F1
                ),
            )

    def test_a_divot_that_disagrees_with_the_field_is_refused(
        self, nominal_scene: ShotScene
    ) -> None:
        """The divot caption branches on the same flag the surface does."""
        with pytest.raises(ValueError, match="resolves_grains"):
            dataclasses.replace(
                nominal_scene,
                divot=dataclasses.replace(
                    nominal_scene.divot, resolves_grains=True, tier=FidelityTier.F1
                ),
            )

    def test_an_f1_divot_stops_claiming_f0_moves_no_sand(
        self, nominal_scene: ShotScene
    ) -> None:
        """The first rendered frame said 'F0 moves no sand' on an F1 scene."""
        note = " ".join(f1_scene(nominal_scene).sand_note())
        assert "F0 moves no sand" not in note
        assert "F1" in note

    def test_carrying_sand_marks_the_surface_as_resolving_grains(
        self, nominal_scene: ShotScene
    ) -> None:
        """The two must agree, or the caption contradicts the picture."""
        with pytest.raises(ValueError, match="resolves_grains"):
            dataclasses.replace(
                nominal_scene, fidelity_tier=FidelityTier.F1, sand=f1_volume()
            )


class TestThePayloadCarriesTheSand:
    """The backend-neutral payload is what a real 3-D provider would consume."""

    def test_the_payload_reports_no_grains_at_f0(
        self, nominal_scene: ShotScene
    ) -> None:
        from src.tools.bunker_shot_gui.shot3d import viewport_payload

        meta = viewport_payload(nominal_scene).meta
        assert meta["resolves_grains"] is False

    def test_the_payload_reports_the_extrusion_at_f1(
        self, nominal_scene: ShotScene
    ) -> None:
        from src.tools.bunker_shot_gui.shot3d import viewport_payload

        meta = viewport_payload(f1_scene(nominal_scene)).meta
        assert meta["resolves_grains"] is True
        assert meta["sand_fidelity"] == "extruded"

    def test_the_payload_carries_the_source_digest(
        self, nominal_scene: ShotScene
    ) -> None:
        """A drawn volume must be traceable to the arrays behind it."""
        from src.tools.bunker_shot_gui.shot3d import viewport_payload

        meta = viewport_payload(f1_scene(nominal_scene)).meta
        assert len(str(meta["sand_digest"])) == 64
