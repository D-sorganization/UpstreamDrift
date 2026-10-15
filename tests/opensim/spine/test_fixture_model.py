"""Synthetic segmental-spine OpenSim fixture (SPN-7a, #12750).

Pure XML structure tests run without OpenSim.  The native tests load the
generated model and check, to 1e-9, that the thorax-proxy transform in the
pelvis frame equals the ADR-0055 composition
``T_base @ prod(T_i) @ T_terminal`` built by ``SegmentalPose.compose()``.
All fixture numbers are invented (see the module docstring of the fixture).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # test-only parse of generated XML
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    build_fixture_osim,
    fixture_base_offset,
    fixture_level_names,
    fixture_neutral_transforms,
    fixture_terminal_offset,
    write_fixture_osim,
)
from src.shared.python.spine.levels import SpinalLevel, levels_between
from src.shared.python.spine.segmental import LevelTransform, SegmentalPose
from tests.helpers.opensim_support import require_opensim

LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)


def _root(spec: FixtureSpec | None = None) -> ET.Element:
    return ET.fromstring(build_fixture_osim(spec or FixtureSpec()))  # noqa: S314  # nosemgrep  # generated, trusted


# --------------------------------------------------------------------- pure --


@pytest.mark.unit
class TestSpec:
    def test_defaults_are_five_lumbar_levels(self) -> None:
        spec = FixtureSpec()
        assert spec.levels == LUMBAR
        assert len(spec.neutral_lordosis_deg) == 5
        assert all(0.0 <= v <= 10.0 for v in spec.neutral_lordosis_deg.values())

    def test_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            FixtureSpec().thorax_proxy_mass_kg = 1.0  # type: ignore[misc]

    def test_mappings_are_read_only(self) -> None:
        with pytest.raises(TypeError):
            FixtureSpec().segment_mass_kg["L5"] = 1.0  # type: ignore[index]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"levels": ()},
            {"levels": (SpinalLevel.L4_L5, SpinalLevel.L5_S1)},
            {"levels": (SpinalLevel.L5_S1, SpinalLevel.L3_L4)},
            {"thorax_proxy_mass_kg": 0.0},
            {"thorax_proxy_mass_kg": float("nan")},
            {"sacral_slope_deg": 95.0},
            {"neutral_lordosis_deg": {SpinalLevel.L5_S1: 3.0}},
            {"segment_mass_kg": {"L5": 1.0}},
            {"segment_length_m": {"pelvis": -0.1}},
        ],
    )
    def test_invalid_specs_raise_value_error(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            FixtureSpec(**kwargs)

    def test_type_errors(self) -> None:
        with pytest.raises(TypeError):
            FixtureSpec(levels=("L5_S1",))  # type: ignore[arg-type]

    def test_extra_mapping_keys_rejected(self) -> None:
        base = FixtureSpec()
        masses = dict(base.segment_mass_kg)
        masses["bogus"] = 1.0
        with pytest.raises(ValueError, match="bogus"):
            FixtureSpec(segment_mass_kg=masses)


@pytest.mark.unit
class TestNames:
    def test_level_names_pattern_and_order(self) -> None:
        names = fixture_level_names(FixtureSpec())
        assert [n.level for n in names] == list(LUMBAR)
        assert names[0].joint == "L5_S1_jnt"
        assert names[0].coordinates == (
            "l5_s1_flex_ext",
            "l5_s1_lat_bend",
            "l5_s1_axial_rot",
        )
        assert names[0].inferior_body == "sacrum"
        assert names[0].superior_body == "L5"
        assert names[-1].joint == "L1_L2_jnt"
        assert names[-1].superior_body == "L1"
        assert names[1].inferior_body == names[0].superior_body

    def test_subset_of_levels(self) -> None:
        spec = FixtureSpec(
            levels=(SpinalLevel.L4_L5, SpinalLevel.L3_L4),
            neutral_lordosis_deg={SpinalLevel.L4_L5: 4.0, SpinalLevel.L3_L4: 5.0},
            segment_mass_kg={"pelvis": 10.0, "L5": 1.0, "L4": 1.0, "L3": 1.0},
            segment_length_m={"pelvis": 0.1, "L5": 0.04, "L4": 0.04, "L3": 0.04},
        )
        names = fixture_level_names(spec)
        assert [n.joint for n in names] == ["L4_L5_jnt", "L3_L4_jnt"]
        assert names[0].inferior_body == "L5"


@pytest.mark.unit
class TestXml:
    def test_parses_and_has_topology(self) -> None:
        root = _root()
        assert root.tag == "OpenSimDocument"
        bodies = [b.get("name") for b in root.findall(".//BodySet/objects/Body")]
        assert bodies == [
            "pelvis",
            "sacrum",
            "L5",
            "L4",
            "L3",
            "L2",
            "L1",
            "thorax_proxy",
        ]

    def test_one_custom_joint_per_level_caudal_to_cranial(self) -> None:
        root = _root()
        joints = root.findall(".//JointSet/objects/CustomJoint")
        assert [j.get("name") for j in joints] == [
            f"{lv.name}_jnt" for lv in LUMBAR
        ]
        assert len(root.findall(".//JointSet/objects/FreeJoint")) == 1
        assert len(root.findall(".//JointSet/objects/WeldJoint")) == 2

    def test_isb_zxy_spatial_transform(self) -> None:
        root = _root()
        for joint, names in zip(
            root.findall(".//JointSet/objects/CustomJoint"),
            fixture_level_names(FixtureSpec()),
            strict=True,
        ):
            axes = joint.findall("SpatialTransform/TransformAxis")
            assert len(axes) == 6
            rotations = axes[:3]
            assert [a.findtext("axis") for a in rotations] == [
                "0 0 1",
                "1 0 0",
                "0 1 0",
            ]
            assert [a.findtext("coordinates") for a in rotations] == list(
                names.coordinates
            )
            for a in axes[3:]:
                assert a.findtext("coordinates") in (None, "")
                assert float(a.findtext("Constant/value")) == 0.0
            coords = [c.get("name") for c in joint.findall("coordinates/Coordinate")]
            assert coords == list(names.coordinates)

    def test_offset_frames_carry_neutral_lordosis(self) -> None:
        spec = FixtureSpec()
        root = _root(spec)
        joint = root.findall(".//JointSet/objects/CustomJoint")[0]
        frames = {
            f.get("name"): f for f in joint.findall("frames/PhysicalOffsetFrame")
        }
        parent = frames["L5_S1_jnt_parent_offset"]
        orient = [float(v) for v in parent.findtext("orientation").split()]
        lordosis = spec.neutral_lordosis_deg[SpinalLevel.L5_S1]
        assert orient[2] == pytest.approx(np.deg2rad(lordosis))
        assert parent.findtext("socket_parent") == "/bodyset/sacrum"
        child = frames["L5_S1_jnt_child_offset"]
        assert child.findtext("socket_parent") == "/bodyset/L5"

    def test_bodies_have_positive_definite_inertia(self) -> None:
        for body in _root().findall(".//BodySet/objects/Body"):
            assert float(body.findtext("mass")) > 0.0
            i6 = [float(v) for v in body.findtext("inertia").split()]
            mat = np.array(
                [[i6[0], i6[3], i6[4]], [i6[3], i6[1], i6[5]], [i6[4], i6[5], i6[2]]]
            )
            assert np.all(np.linalg.eigvalsh(mat) > 0.0)

    def test_deterministic(self) -> None:
        assert build_fixture_osim(FixtureSpec()) == build_fixture_osim(FixtureSpec())

    def test_write(self, tmp_path: Path) -> None:
        out = write_fixture_osim(FixtureSpec(), tmp_path / "sub" / "m.osim")
        assert out.read_text(encoding="utf-8") == build_fixture_osim(FixtureSpec())

    def test_write_rejects_non_osim_suffix(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="osim"):
            write_fixture_osim(FixtureSpec(), tmp_path / "m.txt")

    def test_neutral_transforms_are_rigid(self) -> None:
        spec = FixtureSpec()
        neutral = fixture_neutral_transforms(spec)
        assert list(neutral) == list(LUMBAR)
        for mat in neutral.values():
            assert mat.shape == (4, 4)
            assert np.allclose(mat[:3, :3].T @ mat[:3, :3], np.eye(3))
        for mat in (fixture_base_offset(spec), fixture_terminal_offset(spec)):
            assert np.isclose(np.linalg.det(mat[:3, :3]), 1.0)


# ------------------------------------------------------------------- native --


@pytest.fixture(scope="module")
def osim_module():
    return require_opensim()


@pytest.fixture(scope="module")
def native_model(osim_module, tmp_path_factory):
    path = write_fixture_osim(
        FixtureSpec(), tmp_path_factory.mktemp("fixture") / "spine_fixture.osim"
    )
    return osim_module.Model(str(path))


def _expected(spec: FixtureSpec, angles: np.ndarray, upto: int | None = None):
    transforms = tuple(
        LevelTransform.from_isb_angles(lv, angles[i], neutral=neutral)
        for i, (lv, neutral) in enumerate(fixture_neutral_transforms(spec).items())
    )
    pose = SegmentalPose(transforms)
    return pose, fixture_base_offset(spec)


@pytest.mark.requires_opensim
class TestNative:
    def test_loads_and_inits(self, native_model) -> None:
        state = native_model.initSystem()
        assert state is not None
        assert native_model.getNumCoordinates() == 6 + 15

    def test_neutral_pose_matches_composition(self, native_model) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            body_transform_in,
            set_coordinate_values,
        )

        spec = FixtureSpec()
        state = native_model.initSystem()
        set_coordinate_values(native_model, state, {})
        pose, base = _expected(spec, np.zeros((5, 3)))
        want = base @ pose.compose() @ fixture_terminal_offset(spec)
        got = body_transform_in(native_model, state, "thorax_proxy", "pelvis")
        np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("seed", range(5))
    def test_random_angles_match_segmental_pose(self, native_model, seed) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            body_transform_in,
            set_coordinate_values,
        )

        spec = FixtureSpec()
        rng = np.random.default_rng(seed)
        angles = rng.uniform(-0.4, 0.4, size=(5, 3))
        state = native_model.initSystem()
        values = {}
        for names, triple in zip(fixture_level_names(spec), angles, strict=True):
            values.update(dict(zip(names.coordinates, triple, strict=True)))
        set_coordinate_values(native_model, state, values)
        pose, base = _expected(spec, angles)
        terminal = fixture_terminal_offset(spec)

        # terminal body
        want = base @ pose.compose() @ terminal
        got = body_transform_in(native_model, state, "thorax_proxy", "pelvis")
        np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)

        # every intermediate vertebra: partial products, caudal factor first
        for names in fixture_level_names(spec):
            want_i = base @ pose.compose(LUMBAR[0], names.level)
            got_i = body_transform_in(
                native_model, state, names.superior_body, "pelvis"
            )
            np.testing.assert_allclose(got_i, want_i, atol=1e-9, rtol=0)

    def test_control_reversed_order_is_detected(self, native_model) -> None:
        """Negative control: a wrong composition order must NOT match."""
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            body_transform_in,
            set_coordinate_values,
        )

        spec = FixtureSpec()
        angles = np.random.default_rng(11).uniform(-0.4, 0.4, size=(5, 3))
        state = native_model.initSystem()
        values = {}
        for names, triple in zip(fixture_level_names(spec), angles, strict=True):
            values.update(dict(zip(names.coordinates, triple, strict=True)))
        set_coordinate_values(native_model, state, values)
        pose, base = _expected(spec, angles)
        reversed_prod = np.eye(4)
        for item in reversed(pose.transforms):
            reversed_prod = reversed_prod @ item.matrix()
        wrong = base @ reversed_prod @ fixture_terminal_offset(spec)
        got = body_transform_in(native_model, state, "thorax_proxy", "pelvis")
        assert np.max(np.abs(got - wrong)) > 1e-3

    def test_unknown_coordinate_raises(self, native_model) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            set_coordinate_values,
        )

        state = native_model.initSystem()
        with pytest.raises(ValueError, match="nope"):
            set_coordinate_values(native_model, state, {"nope": 0.0})
