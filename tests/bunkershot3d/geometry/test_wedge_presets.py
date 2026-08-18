"""Named grind-preset tests (issue #8609).

Every published number carries provenance. The package has had one
honesty failure of exactly this kind already (#7999): a preset that
cannot say where a number came from is a fabrication.
"""

from __future__ import annotations

import pytest

from bunkershot3d.geometry.presets import (
    GRIND_PRESETS,
    GrindPreset,
    ProvenanceKind,
    get_preset,
    preset_names,
)
from bunkershot3d.geometry.wedge import WedgeGeometry

pytestmark = pytest.mark.unit


class TestRegistry:
    def test_registry_is_not_empty(self) -> None:
        assert len(preset_names()) >= 5

    def test_names_are_sorted_and_unique(self) -> None:
        names = preset_names()
        assert list(names) == sorted(names)
        assert len(set(names)) == len(names)

    def test_get_preset_returns_a_preset(self) -> None:
        preset = get_preset(preset_names()[0])
        assert isinstance(preset, GrindPreset)
        assert isinstance(preset.geometry, WedgeGeometry)

    def test_unknown_preset_raises_with_the_known_names(self) -> None:
        with pytest.raises(KeyError, match="unknown grind preset"):
            get_preset("ms")

    def test_registry_mapping_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            GRIND_PRESETS["hack"] = get_preset(preset_names()[0])  # type: ignore[index]


class TestProvenance:
    @pytest.mark.parametrize("name", preset_names())
    def test_every_preset_has_a_description_and_a_source(self, name: str) -> None:
        preset = get_preset(name)
        assert preset.description
        assert preset.provenance

    @pytest.mark.parametrize("name", preset_names())
    def test_provenance_keys_are_real_geometry_fields(self, name: str) -> None:
        preset = get_preset(name)
        fields = set(WedgeGeometry.field_names())
        assert set(preset.provenance) <= fields

    @pytest.mark.parametrize("name", preset_names())
    def test_every_provenance_entry_is_attributed(self, name: str) -> None:
        for record in get_preset(name).provenance.values():
            assert isinstance(record.kind, ProvenanceKind)
            assert record.source.strip()
            if record.kind is not ProvenanceKind.ESTIMATED:
                assert record.citation.strip()

    @pytest.mark.parametrize("name", preset_names())
    def test_unattributed_fields_are_reported_as_estimated(self, name: str) -> None:
        preset = get_preset(name)
        published = set(preset.published_fields())
        estimated = set(preset.estimated_fields())
        assert published.isdisjoint(estimated)
        assert published | estimated == set(WedgeGeometry.field_names())

    @pytest.mark.parametrize("name", preset_names())
    def test_geometry_is_constructible_and_valid(self, name: str) -> None:
        geometry = get_preset(name).geometry
        assert geometry.head_mass_kg > 0.0
        assert geometry.patent_compliance()

    @pytest.mark.parametrize("name", preset_names())
    def test_every_preset_lofts_a_watertight_head(self, name: str) -> None:
        from bunkershot3d.geometry.lofting import build_wedge_mesh
        from bunkershot3d.geometry.mesh import check_mesh_validity

        mesh = build_wedge_mesh(
            get_preset(name).geometry, n_profile_points=24, n_stations=7
        )
        report = check_mesh_validity(mesh)
        assert report.is_watertight_solid
        assert report.euler_characteristic == 2

    def test_a_preset_cannot_claim_a_field_that_does_not_exist(self) -> None:
        from bunkershot3d.geometry.presets import ParameterProvenance

        preset = get_preset(preset_names()[0])
        with pytest.raises(ValueError, match="not a WedgeGeometry field"):
            GrindPreset(
                name="bogus",
                description="bogus",
                geometry=preset.geometry,
                provenance={
                    "sole_shininess": ParameterProvenance(
                        kind=ProvenanceKind.PUBLISHED,
                        source="nowhere",
                        citation="none",
                    )
                },
            )


class TestPublishedNumbers:
    def test_patent_examples_carry_their_measured_bounce(self) -> None:
        expected = {
            "acushnet_example_1": 15.99,
            "acushnet_example_2": 18.42,
            "acushnet_example_3": 20.78,
        }
        for name, bounce in expected.items():
            preset = get_preset(name)
            assert preset.geometry.geometric_bounce.angle_deg == pytest.approx(bounce)
            record = preset.provenance["geometric_bounce"]
            assert record.kind is ProvenanceKind.PATENT
            assert "10143900" in record.citation or "10661131" in record.citation

    def test_published_head_masses(self) -> None:
        assert get_preset("sm9_54_f").geometry.head_mass_kg == pytest.approx(0.304)
        assert get_preset("sm9_58_m").geometry.head_mass_kg == pytest.approx(0.300)
        for name in ("sm9_54_f", "sm9_58_m"):
            assert (
                get_preset(name).provenance["head_mass_kg"].kind
                is ProvenanceKind.PUBLISHED
            )

    def test_sole_geometry_of_retail_presets_is_marked_estimated(self) -> None:
        for name in ("sm9_54_f", "sm9_58_m"):
            estimated = set(get_preset(name).estimated_fields())
            assert "sole_camber_area_m2" in estimated
            assert "leading_edge_radius_m" in estimated

    def test_no_fabricated_ping_ms_grind(self) -> None:
        # Research note: no "MS" Ping grind could be verified in any
        # generation, so the registry must not invent one.
        assert not any(name.split("_")[-1] == "ms" for name in preset_names())
