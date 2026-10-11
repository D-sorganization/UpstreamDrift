"""CMB-1 (#11652): spec-native character parameters compile to full_body_spec."""

from __future__ import annotations

import json

import pytest

from src.shared.python.humanoid_character_builder.spec_params import (
    SpecCharacterParameters,
    compile_full_body_spec,
    params_from_spec,
    serialize_spec,
)
from src.shared.python.motion_matching.full_body_spec import (
    canonical_sha256,
    upper_body_slice,
)

pytestmark = pytest.mark.unit

PARAMS = SpecCharacterParameters(
    stature_m=1.80, mass_kg=84.0, trunk_scale=1.05, arm_scale=0.98
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return compile_full_body_spec(PARAMS)


def _require_real_module(name: str) -> None:
    """Skip unless ``name`` is a genuinely installed package, not a test mock."""
    import importlib
    import sys

    try:
        module = importlib.import_module(name)
    except ImportError:
        pytest.skip(f"{name} is not installed")
    root = sys.modules.get(name.split(".")[0])
    if not (hasattr(module, "__file__") and hasattr(root, "__path__")) or type(
        module
    ).__module__.startswith("unittest.mock"):
        pytest.skip(f"{name} is mocked by another test, not a real install")


class TestParameterContract:
    def test_defaults_are_valid(self) -> None:
        p = SpecCharacterParameters()
        assert p.club == "driver"
        assert p.trunk_scale == 1.0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("stature_m", 0.0),
            ("stature_m", 2.6),
            ("mass_kg", -1.0),
            ("mass_kg", 400.0),
            ("trunk_scale", 0.2),
            ("arm_scale", 2.5),
            ("shoulder_scale", float("nan")),
            ("grip_roll_deg", 400.0),
        ],
    )
    def test_out_of_range_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            SpecCharacterParameters(**{field: value})

    def test_unknown_club_rejected(self) -> None:
        with pytest.raises(ValueError, match="club"):
            SpecCharacterParameters(club="putter9000")

    def test_non_numeric_rejected(self) -> None:
        with pytest.raises(TypeError, match="stature_m"):
            SpecCharacterParameters(stature_m="tall")  # type: ignore[arg-type]

    def test_dict_round_trip_and_unknown_keys(self) -> None:
        assert SpecCharacterParameters.from_dict(PARAMS.to_dict()) == PARAMS
        with pytest.raises(ValueError, match="unknown"):
            SpecCharacterParameters.from_dict({**PARAMS.to_dict(), "bogus": 1})

    def test_from_body_parameters_maps_proportions(self) -> None:
        from src.shared.python.humanoid_character_builder.core.body_parameters import (
            BodyParameters,
        )

        body = BodyParameters(
            height_m=1.9,
            mass_kg=90.0,
            torso_length_factor=1.1,
            arm_length_factor=1.05,
            shoulder_width_factor=0.95,
        )
        p = SpecCharacterParameters.from_body_parameters(body)
        assert (p.stature_m, p.mass_kg) == (1.9, 90.0)
        assert (p.trunk_scale, p.arm_scale, p.shoulder_scale) == (1.1, 1.05, 0.95)


class TestCompile:
    def test_spec_validates(self, spec: dict) -> None:
        # derive_full_body_spec validates before returning; the document also
        # records the identity of its embedded upper-body slice.
        counts = spec["upper_body_counts"]
        assert counts["coordinates"] == 30  # 27 native + 3 neck
        assert len(upper_body_slice(spec)["bodies"]) == counts["bodies"]
        assert len(canonical_sha256(spec)) == 64
        assert spec["schema_version"] == "full-body-v1"
        assert len(spec["coordinate_order"]) == 44

    def test_subject_reflects_inputs(self, spec: dict) -> None:
        assert spec["subject"]["stature_m"] == 1.80
        assert spec["subject"]["mass_kg"] == 84.0
        assert spec["club"]["name"] == "driver"

    def test_total_mass_scales_with_input(self, spec: dict) -> None:
        total = sum(s["mass_kg"] for b in spec["bodies"] for s in b["solids"])
        # De Leva segments plus the copied club, hands and Rajagopal legs.
        assert 60.0 < total < 120.0

    def test_byte_identical_for_same_inputs(self, spec: dict) -> None:
        again = compile_full_body_spec(PARAMS)
        assert serialize_spec(again) == serialize_spec(spec)

    def test_different_inputs_differ(self, spec: dict) -> None:
        other = compile_full_body_spec(
            SpecCharacterParameters(stature_m=1.60, mass_kg=60.0)
        )
        assert serialize_spec(other) != serialize_spec(spec)

    def test_serialize_is_sorted_json_with_newline(self, spec: dict) -> None:
        text = serialize_spec(spec)
        assert text.endswith("\n")
        assert json.loads(text) == spec

    def test_round_trip_spec_to_params_to_spec(self, spec: dict) -> None:
        recovered = params_from_spec(spec)
        assert recovered == PARAMS
        assert serialize_spec(compile_full_body_spec(recovered)) == serialize_spec(spec)

    def test_params_from_spec_rejects_foreign_document(self) -> None:
        with pytest.raises(ValueError, match="subject"):
            params_from_spec({"schema_version": "full-body-v1"})


class TestEngineExporters:
    def test_mujoco_loads(self, spec: dict) -> None:
        mujoco = pytest.importorskip("mujoco")
        from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
            export_full_body_mjcf,
        )

        xml, _meta = export_full_body_mjcf(serialize_spec(spec).encode())
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nq >= 41

    def test_pinocchio_builds(self, spec: dict) -> None:
        _require_real_module("pinocchio")
        from src.engines.physics_engines.pinocchio.python.native_model import (
            build_full_body_pinocchio_model,
        )

        assert build_full_body_pinocchio_model(spec) is not None

    def test_drake_builds(self, spec: dict) -> None:
        _require_real_module("pydrake.all")
        from src.engines.physics_engines.drake.python.full_body_model import (
            FullBodyDrakeModel,
        )

        assert FullBodyDrakeModel(spec).model_sha256

    def test_opensim_exports(self, spec: dict) -> None:
        from src.engines.physics_engines.opensim.python.full_body_osim import (
            export_full_body_osim,
        )

        xml, _meta = export_full_body_osim(spec)
        assert "<OpenSimDocument" in xml
