"""Fixture coupler constraints and lumped trunk muscles (SPN-7b, #12751).

Pure XML tests run without OpenSim; native tests load the coupled model,
prescribe the massless driver coordinates ``phi`` and check, to 1e-9, that the
level coordinates equal ``law.expand(phi)`` and that the thorax-proxy pose
equals the ADR-0055 composition.  Every number is invented.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # test-only parse of generated XML

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fixture_coupling import (
    DRIVER_BODY,
    DRIVER_JOINT,
    fixture_muscle_names,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    build_fixture_osim,
    fixture_base_offset,
    fixture_level_names,
    fixture_neutral_transforms,
    fixture_terminal_offset,
    write_fixture_osim,
)
from src.shared.python.spine.coupling import (
    LinearCouplingLaw,
    StateDependentCouplingLaw,
    level_rotations,
)
from src.shared.python.spine.levels import SpinalLevel, levels_between
from tests.helpers.opensim_support import require_opensim

LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
PHI_NAMES = ("trunk_flex_ext", "trunk_lat_bend", "trunk_axial_rot")


def _law(levels=LUMBAR) -> LinearCouplingLaw:
    """Invented law with cross-axis terms (LB -> FE, AR -> LB)."""
    mats = {}
    for k, lv in enumerate(levels):
        w = 0.1 + 0.05 * k
        mats[lv] = np.array(
            [[w, 0.03 * (k + 1), 0.0], [0.0, w, 0.02 * (k + 1)], [0.0, 0.0, w / 2]]
        )
    return LinearCouplingLaw("invented", "1", mats, PHI_NAMES, "invented, SPN-7b test")


def _root(**kwargs) -> ET.Element:
    text = build_fixture_osim(FixtureSpec(), **kwargs)
    return ET.fromstring(text)  # noqa: S314  # nosemgrep  # generated, trusted


# --------------------------------------------------------------------- pure --


@pytest.mark.unit
def test_tendon_slack_positive_and_matches_analytic_length() -> None:
    from src.engines.physics_engines.opensim.python.spine import fixture_coupling

    spec = FixtureSpec()
    root = ET.fromstring(build_fixture_osim(spec, muscles=True))  # noqa: S314  # nosemgrep  # generated, trusted
    muscles = root.findall(".//Millard2012EquilibriumMuscle")
    assert len(muscles) == 8
    lengths = fixture_coupling.neutral_path_lengths(spec)
    for node in muscles:
        name = node.get("name")
        slack = float(node.findtext("tendon_slack_length"))
        fiber = float(node.findtext("optimal_fiber_length"))
        assert slack > 0.0
        assert slack + fiber == pytest.approx(lengths[name], abs=1e-12)


@pytest.mark.unit
class TestCouplerXml:
    def test_default_has_no_driver_no_constraints_no_muscles(self) -> None:
        root = _root()
        assert root.find(".//ConstraintSet") is None
        assert root.find(f".//Body[@name='{DRIVER_BODY}']") is None
        assert root.find(".//ForceSet") is None

    def test_coupled_fixture_tightens_assembly_accuracy(self) -> None:
        assert _root().find("Model/assembly_accuracy") is None
        accuracy = _root(coupling=_law()).findtext("Model/assembly_accuracy")
        assert accuracy is not None and float(accuracy) <= 1e-12

    def test_driver_body_is_massless_and_inertia_free(self) -> None:
        body = _root(coupling=_law()).find(
            f".//BodySet/objects/Body[@name='{DRIVER_BODY}']"
        )
        assert body is not None
        assert float(body.findtext("mass")) == 0.0
        assert [float(v) for v in body.findtext("inertia").split()] == [0.0] * 6

    def test_driver_joint_carries_independent_coordinates(self) -> None:
        root = _root(coupling=_law())
        joint = root.find(f".//JointSet/objects/CustomJoint[@name='{DRIVER_JOINT}']")
        assert joint is not None
        coords = [c.get("name") for c in joint.findall("coordinates/Coordinate")]
        assert coords == list(PHI_NAMES)
        assert joint.findtext("socket_parent_frame") == f"{DRIVER_JOINT}_parent_offset"
        parents = {
            f.findtext("socket_parent")
            for f in joint.findall("frames/PhysicalOffsetFrame")
        }
        assert parents == {"/bodyset/pelvis", f"/bodyset/{DRIVER_BODY}"}

    def test_one_coupler_per_dependent_coordinate_with_law_coefficients(self) -> None:
        law = _law()
        root = _root(coupling=law)
        couplers = root.findall(".//ConstraintSet/objects/CoordinateCouplerConstraint")
        expected = [
            (c, lv)
            for lv, names in zip(
                LUMBAR, fixture_level_names(FixtureSpec()), strict=True
            )
            for c in names.coordinates
            for lv in (lv,)
        ]
        assert len(couplers) == 15 == len(expected)
        for item, (coord, level) in zip(couplers, expected, strict=True):
            assert item.findtext("dependent_coordinate_name") == coord
            assert item.findtext("independent_coordinate_names").split() == list(
                PHI_NAMES
            )
            row = int(
                [
                    n.coordinates
                    for n in fixture_level_names(FixtureSpec())
                    if n.level is level
                ][0].index(coord)
            )
            coeffs = [
                float(v)
                for v in item.findtext(
                    "coupled_coordinates_function/LinearFunction/coefficients"
                ).split()
            ]
            np.testing.assert_allclose(coeffs[:-1], law.matrices[level][row])
            assert coeffs[-1] == 0.0

    def test_partial_level_coverage_only_couples_law_levels(self) -> None:
        sub = LUMBAR[:2]
        root = _root(coupling=_law(sub))
        assert len(root.findall(".//CoordinateCouplerConstraint")) == 6

    def test_level_joint_coordinates_survive(self) -> None:
        root = _root(coupling=_law())
        joints = root.findall(".//JointSet/objects/CustomJoint")
        assert len(joints) == 5 + 1  # five levels plus the driver joint

    @pytest.mark.parametrize(
        ("bad", "exc"),
        [
            (object(), TypeError),
            (
                StateDependentCouplingLaw(
                    "x",
                    "1",
                    PHI_NAMES,
                    "trunk_flex_ext",
                    np.array([0.0, 1.0]),
                    {lv: np.zeros((2, 3, 3)) for lv in LUMBAR},
                    "invented",
                ),
                TypeError,
            ),
        ],
    )
    def test_unrepresentable_law_raises(self, bad, exc) -> None:
        with pytest.raises(exc, match="LinearCouplingLaw"):
            build_fixture_osim(FixtureSpec(), coupling=bad)

    def test_law_level_outside_spec_raises(self) -> None:
        spec = FixtureSpec(
            levels=LUMBAR[:2],
            neutral_lordosis_deg=dict.fromkeys(LUMBAR[:2], 1.0),
            segment_mass_kg={"pelvis": 10.0, "sacrum": 1.0, "L5": 1.0, "L4": 1.0},
            segment_length_m={"pelvis": 0.1, "sacrum": 0.05, "L5": 0.04, "L4": 0.04},
        )
        with pytest.raises(ValueError, match="not in the fixture"):
            build_fixture_osim(spec, coupling=_law())

    def test_independent_name_clashing_with_level_coordinate_raises(self) -> None:
        mats = {lv: np.eye(3) for lv in LUMBAR}
        clash = LinearCouplingLaw(
            "c", "1", mats, ("l5_s1_flex_ext", "a", "b"), "invented"
        )
        with pytest.raises(ValueError, match="clash"):
            build_fixture_osim(FixtureSpec(), coupling=clash)

    def test_too_many_independent_coordinates_raises(self) -> None:
        names = tuple(f"q{i}" for i in range(7))
        law = LinearCouplingLaw(
            "c", "1", {lv: np.zeros((3, 7)) for lv in LUMBAR}, names, "invented"
        )
        with pytest.raises(ValueError, match="at most 6"):
            build_fixture_osim(FixtureSpec(), coupling=law)

    def test_deterministic(self) -> None:
        a = build_fixture_osim(FixtureSpec(), coupling=_law(), muscles=True)
        b = build_fixture_osim(FixtureSpec(), coupling=_law(), muscles=True)
        assert a == b

    def test_bad_flags_raise(self) -> None:
        with pytest.raises(TypeError, match="muscles"):
            build_fixture_osim(FixtureSpec(), muscles="yes")  # type: ignore[arg-type]


@pytest.mark.unit
class TestMuscleXml:
    def test_eight_bilateral_millard_muscles(self) -> None:
        root = _root(muscles=True)
        muscles = root.findall(".//ForceSet/objects/Millard2012EquilibriumMuscle")
        names = [m.get("name") for m in muscles]
        assert names == list(fixture_muscle_names())
        assert len(names) == 8
        assert sum(n.endswith("_l") for n in names) == 4
        assert sum(n.endswith("_r") for n in names) == 4
        for stem in (
            "erector_spinae",
            "rectus_abdominis",
            "external_oblique",
            "internal_oblique",
        ):
            assert {f"{stem}_l", f"{stem}_r"} <= set(names)

    def test_positive_max_isometric_force_and_path_spans_pelvis_to_thorax(self) -> None:
        for muscle in _root(muscles=True).findall(".//Millard2012EquilibriumMuscle"):
            assert float(muscle.findtext("max_isometric_force")) > 0.0
            points = muscle.findall(".//PathPointSet/objects/PathPoint")
            parents = [p.findtext("socket_parent_frame") for p in points]
            assert parents == [
                "/bodyset/pelvis",
                "/bodyset/L3",
                "/bodyset/thorax_proxy",
            ]

    def test_muscles_mirror_about_the_sagittal_plane(self) -> None:
        root = _root(muscles=True)
        by_name = {
            m.get("name"): m for m in root.findall(".//Millard2012EquilibriumMuscle")
        }
        for stem in (
            "erector_spinae",
            "rectus_abdominis",
            "external_oblique",
            "internal_oblique",
        ):
            left = [
                [float(v) for v in p.findtext("location").split()]
                for p in by_name[f"{stem}_l"].findall(".//PathPoint")
            ]
            right = [
                [float(v) for v in p.findtext("location").split()]
                for p in by_name[f"{stem}_r"].findall(".//PathPoint")
            ]
            for lp, rp in zip(left, right, strict=True):
                assert lp[0] == pytest.approx(rp[0])
                assert lp[1] == pytest.approx(rp[1])
                assert lp[2] == pytest.approx(-rp[2])

    def test_muscles_without_coupling_have_no_driver(self) -> None:
        assert _root(muscles=True).find(f".//Body[@name='{DRIVER_BODY}']") is None


# ------------------------------------------------------------------- native --


@pytest.fixture(scope="module")
def osim_module():
    return require_opensim()


@pytest.fixture(scope="module")
def coupled(osim_module, tmp_path_factory):
    path = tmp_path_factory.mktemp("coupled") / "coupled.osim"
    path.write_text(
        build_fixture_osim(FixtureSpec(), coupling=_law(), muscles=True),
        encoding="utf-8",
    )
    model = osim_module.Model(str(path))
    return model, model.initSystem()


@pytest.fixture(scope="module")
def muscled(osim_module, tmp_path_factory):
    path = write_fixture_osim(
        FixtureSpec(), tmp_path_factory.mktemp("muscled") / "m.osim", muscles=True
    )
    model = osim_module.Model(str(path))
    return model, model.initSystem()


@pytest.mark.requires_opensim
@pytest.mark.integration
class TestNativeCoupling:
    def test_loads_and_counts(self, coupled) -> None:
        model, _ = coupled
        assert model.getNumCoordinates() == 6 + 15 + 3
        assert model.getConstraintSet().getSize() == 15
        assert model.getMuscles().getSize() == 8

    def test_driver_body_is_massless(self, coupled) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            body_mass,
        )

        model, _ = coupled
        assert body_mass(model, DRIVER_BODY) == 0.0

    @pytest.mark.parametrize("seed", range(6))
    def test_levels_equal_law_expand(self, coupled, seed) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            coordinate_value,
            set_independent_values,
        )

        model, state = coupled
        law = _law()
        phi = np.random.default_rng(seed).uniform(-0.5, 0.5, size=3)
        set_independent_values(model, state, dict(zip(PHI_NAMES, phi, strict=True)))
        expand = law.expand(phi)
        for lv, names in zip(LUMBAR, fixture_level_names(FixtureSpec()), strict=True):
            got = [coordinate_value(model, state, n) for n in names.coordinates]
            np.testing.assert_allclose(got, expand[lv], atol=1e-9, rtol=0)
        for name, value in zip(PHI_NAMES, phi, strict=True):
            assert coordinate_value(model, state, name) == pytest.approx(
                value, abs=1e-9
            )

    @pytest.mark.parametrize("seed", range(3))
    def test_thorax_pose_equals_composition(self, coupled, seed) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            body_transform_in,
            set_independent_values,
        )

        model, state = coupled
        spec = FixtureSpec()
        phi = np.random.default_rng(100 + seed).uniform(-0.5, 0.5, size=3)
        set_independent_values(model, state, dict(zip(PHI_NAMES, phi, strict=True)))
        pose = level_rotations(_law(), phi, fixture_neutral_transforms(spec))
        want = (
            fixture_base_offset(spec) @ pose.compose() @ fixture_terminal_offset(spec)
        )
        got = body_transform_in(model, state, "thorax_proxy", "pelvis")
        np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)

    def test_driver_body_does_not_move_the_chain(self, coupled) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            body_transform_in,
            set_independent_values,
        )

        model, state = coupled
        set_independent_values(model, state, {"trunk_flex_ext": 0.3})
        driver = body_transform_in(model, state, DRIVER_BODY, "pelvis")
        assert abs(np.linalg.det(driver[:3, :3]) - 1.0) < 1e-9
        assert np.max(np.abs(driver[:3, :3] - np.eye(3))) > 1e-3  # it does rotate


@pytest.mark.requires_opensim
@pytest.mark.integration
class TestNativeMuscles:
    def test_muscles_present_with_positive_force(self, muscled) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            muscle_max_isometric_force,
        )

        model, _ = muscled
        assert model.getMuscles().getSize() == 8
        for name in fixture_muscle_names():
            assert muscle_max_isometric_force(model, name) > 0.0

    def test_state_derivatives_run_and_are_finite(self, muscled) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            state_derivatives,
        )

        model, state = muscled
        model.equilibrateMuscles(state)
        deriv = state_derivatives(model, state)
        assert deriv.size >= 6 * 2 + 15 * 2 + 8 * 2
        assert np.all(np.isfinite(deriv))

    def test_coupled_model_is_kinematic_only(self, coupled) -> None:
        """The massless driver makes forward dynamics singular (documented)."""
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            state_derivatives,
        )

        model, state = coupled
        assert not np.all(np.isfinite(state_derivatives(model, state)))

    def test_moment_arms_finite_and_bilaterally_consistent(self, muscled) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            muscle_moment_arm,
            set_coordinate_values,
        )

        model, state = muscled
        set_coordinate_values(model, state, {})
        names = fixture_level_names(FixtureSpec())
        for level in names:
            for stem in (
                "erector_spinae",
                "rectus_abdominis",
                "external_oblique",
                "internal_oblique",
            ):
                arms = {
                    side: [
                        muscle_moment_arm(model, state, f"{stem}_{side}", coord)
                        for coord in level.coordinates
                    ]
                    for side in ("l", "r")
                }
                assert np.all(np.isfinite(arms["l"])) and np.all(np.isfinite(arms["r"]))
                # mirror about the sagittal plane: FE even, LB and AR odd
                np.testing.assert_allclose(arms["l"][0], arms["r"][0], atol=1e-9)
                np.testing.assert_allclose(arms["l"][1], -arms["r"][1], atol=1e-9)
                np.testing.assert_allclose(arms["l"][2], -arms["r"][2], atol=1e-9)

    def test_obliques_have_nonzero_opposite_axial_arms(self, muscled) -> None:
        from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
            muscle_moment_arm,
            set_coordinate_values,
        )

        model, state = muscled
        set_coordinate_values(model, state, {})
        coord = fixture_level_names(FixtureSpec())[2].axial_rot
        ext = muscle_moment_arm(model, state, "external_oblique_l", coord)
        internal = muscle_moment_arm(model, state, "internal_oblique_l", coord)
        assert abs(ext) > 1e-5 and abs(internal) > 1e-5
        assert ext * internal < 0.0


@pytest.mark.requires_opensim
@pytest.mark.integration
class TestNeutralMuscleLength:
    def test_native_length_matches_analytic_and_fibre_is_optimal(
        self, muscled
    ) -> None:
        from src.engines.physics_engines.opensim.python.spine import (
            fixture_coupling,
        )

        model, state = muscled
        lengths = fixture_coupling.neutral_path_lengths(FixtureSpec())
        muscles = model.getMuscles()
        for i in range(muscles.getSize()):
            muscle = muscles.get(i)
            assert muscle.getLength(state) == pytest.approx(
                lengths[muscle.getName()], abs=1e-9
            )
            assert muscle.getLength(state) - muscle.get_tendon_slack_length() == (
                pytest.approx(muscle.get_optimal_fiber_length(), abs=1e-9)
            )

    def test_full_activation_gives_active_force_at_neutral(self, muscled) -> None:
        model, state = muscled
        muscles = model.getMuscles()
        for i in range(muscles.getSize()):
            muscle = muscles.get(i)
            muscle.setActivation(state, 1.0)
        model.equilibrateMuscles(state)
        model.realizeDynamics(state)
        for i in range(muscles.getSize()):
            muscle = muscles.get(i)
            assert muscle.getActiveFiberForce(state) > 0.5 * muscle.getMaxIsometricForce()
